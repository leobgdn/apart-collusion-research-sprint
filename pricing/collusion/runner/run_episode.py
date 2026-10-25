"""Run one episode and log every turn.

The same loop drives scripted agents and LLM agents. In both cases the prompt is
rendered and logged, so the turn log is a faithful record of what an agent saw
whether or not the agent actually read it. That is what lets the round-trip test
and the probe layer work off the logs alone.

Turns are buffered and handed to the writer only when the episode finishes, so
the log never holds a partial episode: a wall-clock abort (or a crash) mid-
episode leaves the log valid, and the resumed sweep re-runs that episode from
scratch without duplicating turns.
"""

from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from collusion.agents.parsing import ParsedAction, parse_response
from collusion.envs.pricing import prompts
from collusion.envs.pricing.conditions import Condition
from collusion.envs.pricing.economics import (
    Benchmarks,
    PricingParams,
    best_response_profit,
)
from collusion.envs.pricing.pricing_env import EnvConfig, PricingEnvironment
from collusion.envs.pricing.prompts import RenderedPrompt
from collusion.envs.pricing.state import Observation
from collusion.metrics.action_metrics import DEFAULT_METRIC_WINDOW, metrics_for_env
from collusion.metrics.text_judge import score_episode_text
from collusion.runner.logging_io import (
    EpisodeSummary,
    TurnRecord,
    episode_id_for,
)

# Rounds in the rolling collusion index logged per turn (Track A addendum 4).
ROLLING_WINDOW = 10


class DeadlineReached(RuntimeError):
    """The wall-clock hard stop passed mid-episode; the episode is discarded."""


@dataclass
class AgentReply:
    """Normalised return value of an agent turn."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model_settings: dict[str, Any] = field(default_factory=dict)
    model: str | None = None
    rendered_prompt: str = ""
    call_info: dict[str, Any] = field(default_factory=dict)


class Agent(Protocol):
    """What the runner needs from an agent.

    `act` may return either a plain string (scripted agents) or an AgentReply
    (LLM agents, which also report tokens and serving details).
    """

    agent_id: str
    model_name: str

    def act(
        self,
        obs: Observation,
        condition: Condition,
        rng: np.random.Generator,
        rendered: RenderedPrompt | None = ...,
    ) -> str | AgentReply: ...


def _normalise(reply: str | AgentReply, agent: Any) -> AgentReply:
    if isinstance(reply, AgentReply):
        if reply.model is None:
            reply.model = getattr(agent, "model_name", "unknown")
        return reply
    return AgentReply(text=reply, model=getattr(agent, "model_name", "unknown"))


@dataclass
class EpisodeResult:
    """Everything one episode produced."""

    summary: EpisodeSummary
    turns: list[TurnRecord]
    env: PricingEnvironment


def rolling_index(
    profits: np.ndarray, start: int, stop: int, benchmarks: Benchmarks
) -> float:
    """Collusion index over periods [start, stop) (0-indexed rows of `profits`)."""
    window = profits[start:stop]
    span = benchmarks.profit_monopoly - benchmarks.profit_nash
    return float((window.mean() - benchmarks.profit_nash) / span)


def _price_window(obs: Observation) -> list[dict[str, Any]]:
    """The numeric rows of the market table shown in the prompt, as displayed."""
    return [
        {
            "period": rec.period,
            "own_price": round(rec.prices[obs.seller_index], 3),
            "own_qty": round(rec.quantities[obs.seller_index], 3),
            "own_profit": round(rec.profits[obs.seller_index], 4),
            "rival_prices": [round(rec.prices[i], 3) for i in obs.rival_indices],
        }
        for rec in obs.history
    ]


def _take_turn(
    agent: Any,
    obs: Observation,
    condition: Condition,
    rng: np.random.Generator,
    rendered: RenderedPrompt,
    max_parse_retries: int,
) -> tuple[AgentReply, ParsedAction, int]:
    """One agent's move, retrying only on a genuine parse failure.

    A parsed-but-odd action is the agent's choice and must stand.
    """
    reply: AgentReply | None = None
    action: ParsedAction | None = None
    attempts = 0
    for attempts in range(max_parse_retries + 1):
        reply = _normalise(agent.act(obs, condition, rng, rendered), agent)
        action = parse_response(
            reply.text, expect_message=condition.has_channel, price_grid=obs.price_grid
        )
        if action.ok:
            break
    assert reply is not None and action is not None
    return reply, action, attempts


def run_episode(
    *,
    condition: Condition,
    seed: int,
    agents: list[Any],
    run_id: str = "adhoc",
    params: PricingParams | None = None,
    env_config: EnvConfig | None = None,
    metric_window: int = DEFAULT_METRIC_WINDOW,
    label_threshold: float | None = None,
    max_parse_retries: int = 2,
    turn_writer=None,
    deadline: float | None = None,
    parallel_agents: bool = False,
    batch: int | None = None,
) -> EpisodeResult:
    """Play one episode of the pricing game and return its log and summary.

    Determinism: each agent gets a Generator seeded from (seed, agent_index)
    only -- deliberately *not* from the condition -- so the same seed draws the
    same agent-side randomness in every condition. That makes conditions paired
    across seeds and keeps the between-condition comparison from inheriting
    noise that has nothing to do with the manipulation.

    `deadline` is a `time.time()` epoch. It is checked before every period; once
    passed, DeadlineReached is raised and nothing from this episode is written.

    `parallel_agents` sends the sellers' requests for a period concurrently
    (they move simultaneously, so neither depends on the other). Used for LLM
    agents; scripted agents stay sequential.
    """
    params = params or PricingParams()
    env_config = env_config or EnvConfig()
    if len(agents) != params.n_sellers:
        raise ValueError(
            f"{params.n_sellers} sellers configured but {len(agents)} agents given"
        )

    env = PricingEnvironment(
        condition=condition, params=params, config=env_config, seed=seed
    )
    rngs = [
        np.random.default_rng(np.random.SeedSequence([seed, i]))
        for i in range(len(agents))
    ]

    episode_id = episode_id_for(condition.condition_id, seed)
    channel = "private" if condition.has_channel else "none"
    turns: list[TurnRecord] = []
    parse_failures = 0
    started = time.monotonic()
    pool = ThreadPoolExecutor(max_workers=len(agents)) if parallel_agents else None

    try:
        while not env.done:
            if deadline is not None and time.time() >= deadline:
                raise DeadlineReached(
                    f"{episode_id}: wall-clock deadline reached at period "
                    f"{env.period + 1}; episode discarded"
                )
            observations = env.observations()
            rendered_all: list[RenderedPrompt] = [
                prompts.render(condition, obs) for obs in observations
            ]
            jobs = [
                (agent, obs, condition, rngs[i], rendered, max_parse_retries)
                for i, (agent, obs, rendered) in enumerate(
                    zip(agents, observations, rendered_all)
                )
            ]
            if pool is not None:
                outcomes = list(pool.map(lambda job: _take_turn(*job), jobs))
            else:
                outcomes = [_take_turn(*job) for job in jobs]
            replies = [o[0] for o in outcomes]
            actions = [o[1] for o in outcomes]
            retries_used = [o[2] for o in outcomes]
            parse_failures += sum(1 for a in actions if not a.ok)

            period = env.period + 1
            record = env.step(actions)

            for i, (agent, obs, rendered, reply, action) in enumerate(
                zip(agents, observations, rendered_all, replies, actions)
            ):
                rival_prices = np.array(
                    [record.prices[j] for j in obs.rival_indices], dtype=float
                )
                br_price, br_profit = best_response_profit(rival_prices, i, params)
                own_profit = record.profits[i]
                turns.append(
                    TurnRecord(
                        run_id=run_id,
                        condition_id=condition.condition_id,
                        factors=condition.factors,
                        seed=seed,
                        episode_id=episode_id,
                        turn=(period - 1) * len(agents) + i,
                        agent_id=getattr(agent, "agent_id", f"seller_{i}"),
                        model=reply.model or "unknown",
                        model_settings=reply.model_settings,
                        prompt_messages=rendered.messages,
                        response_text=reply.text,
                        parsed_action=action.as_dict(),
                        channel=channel,
                        message_text=action.message,
                        reward=float(own_profit),
                        regret=float(max(br_profit - own_profit, 0.0)),
                        metric_components={
                            "own_price": record.prices[i],
                            "own_quantity": record.quantities[i],
                            "rival_prices": [
                                record.prices[j] for j in obs.rival_indices
                            ],
                            "best_response_price": br_price,
                            "best_response_profit": br_profit,
                            "price_nash": env.benchmarks.price_nash,
                            "price_monopoly": env.benchmarks.price_monopoly,
                            "window_periods": condition.window_periods,
                            "history_rows_shown": len(obs.history),
                        },
                        tokens_in=reply.tokens_in,
                        tokens_out=reply.tokens_out,
                        cost_usd=reply.cost_usd,
                        agent_index=i,
                        period=period,
                        retries=retries_used[i],
                        rendered_prompt=reply.rendered_prompt,
                        call_info=reply.call_info,
                        price_window=_price_window(obs),
                    )
                )
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    # Per-turn rolling labels need the whole path, so they are filled in here.
    profit_matrix = env.profit_matrix()
    total = env.period
    window = min(metric_window, total)
    for turn in turns:
        t = turn.period  # 1-indexed; row t-1 of profit_matrix
        turn.rolling_delta = rolling_index(
            profit_matrix, max(0, t - ROLLING_WINDOW), t, env.benchmarks
        )
        turn.next_rolling_delta = (
            rolling_index(profit_matrix, t, t + ROLLING_WINDOW, env.benchmarks)
            if t + ROLLING_WINDOW <= total
            else None
        )
        turn.in_metric_window = t > total - window

    metrics = metrics_for_env(env, window=metric_window)
    label = metrics.label(label_threshold) if label_threshold is not None else -1
    text_baseline = score_episode_text(
        (t.parsed_action for t in turns), channel_on=condition.has_channel
    )
    summary = EpisodeSummary(
        episode_id=episode_id,
        condition_id=condition.condition_id,
        factors=condition.factors,
        seed=seed,
        action_collusion_score=metrics.collusion_index,
        action_collusion_label=label,
        # The LLM judge has not run (Phase 1 override 7). The crude keyword
        # baseline is stored separately under `text_baseline`, never here.
        text_collusion_score=None,
        joint_reward=metrics.joint_reward,
        completed=env.done,
        parse_failures=parse_failures,
        run_id=run_id,
        price_index=metrics.price_index,
        category=metrics.category,
        is_pathological=metrics.is_pathological,
        action_metrics=metrics.as_dict(),
        benchmarks=env.benchmarks.as_dict(),
        params=params.as_dict(),
        agent_models=[getattr(a, "model_name", "unknown") for a in agents],
        label_threshold=label_threshold,
        total_cost_usd=sum(t.cost_usd for t in turns),
        total_tokens_in=sum(t.tokens_in for t in turns),
        total_tokens_out=sum(t.tokens_out for t in turns),
        n_periods=env.period,
        wall_seconds=time.monotonic() - started,
        batch=batch,
        env_config=dataclasses.asdict(env_config),
        metric_window=window,
        reasoning_blocks=sum(1 for t in turns if t.parsed_action["has_reasoning_block"]),
        truncated_turns=sum(1 for t in turns if t.parsed_action["truncated_fields"]),
        message_leaks=sum(
            1
            for t in turns
            if t.parsed_action["failure_reason"] == "message_in_no_channel_condition"
        ),
        text_baseline=text_baseline.as_dict(),
    )
    if turn_writer is not None:
        turn_writer.write_many(turns)
    return EpisodeResult(summary=summary, turns=turns, env=env)
