"""Run one episode and log every turn.

The same loop drives scripted agents and (from Stage 2) LLM agents. In both
cases the prompt is rendered and logged, so the turn log is a faithful record of
what an agent saw whether or not the agent actually read it. That is what lets
the round-trip test and the probe layer work off the logs alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from collusion.agents.parsing import ParsedAction, parse_response
from collusion.envs.pricing import prompts
from collusion.envs.pricing.conditions import Condition
from collusion.envs.pricing.economics import PricingParams, best_response_profit
from collusion.envs.pricing.pricing_env import EnvConfig, PricingEnvironment
from collusion.envs.pricing.prompts import RenderedPrompt
from collusion.envs.pricing.state import Observation
from collusion.metrics.action_metrics import DEFAULT_METRIC_WINDOW, metrics_for_env
from collusion.runner.logging_io import (
    EpisodeSummary,
    TurnRecord,
    episode_id_for,
)


@dataclass
class AgentReply:
    """Normalised return value of an agent turn."""

    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    model_settings: dict[str, Any] = field(default_factory=dict)
    model: str | None = None


class Agent(Protocol):
    """What the runner needs from an agent.

    `act` may return either a plain string (scripted agents) or an AgentReply
    (LLM agents, which also report tokens and cost).
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
    text_collusion_score: float | None = None,
) -> EpisodeResult:
    """Play one episode of the pricing game and return its log and summary.

    Determinism: each agent gets a Generator seeded from (seed, agent_index)
    only -- deliberately *not* from the condition -- so the same seed draws the
    same agent-side randomness in every condition. That makes conditions paired
    across seeds and keeps the between-condition comparison from inheriting
    noise that has nothing to do with the manipulation.
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

    while not env.done:
        observations = env.observations()
        rendered_all: list[RenderedPrompt] = [
            prompts.render(condition, obs) for obs in observations
        ]
        replies: list[AgentReply] = []
        actions: list[ParsedAction] = []
        retries_used: list[int] = []

        for i, (agent, obs, rendered) in enumerate(
            zip(agents, observations, rendered_all)
        ):
            reply: AgentReply | None = None
            action: ParsedAction | None = None
            attempts = 0
            # Retry only on a genuine parse failure; a parsed-but-odd action is
            # the agent's choice and must stand.
            for attempts in range(max_parse_retries + 1):
                reply = _normalise(
                    agent.act(obs, condition, rngs[i], rendered), agent
                )
                action = parse_response(
                    reply.text, expect_message=condition.has_channel
                )
                if action.ok:
                    break
            assert reply is not None and action is not None
            if not action.ok:
                parse_failures += 1
            replies.append(reply)
            actions.append(action)
            retries_used.append(attempts)

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
            turn = TurnRecord(
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
                    "rival_prices": [record.prices[j] for j in obs.rival_indices],
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
            )
            turns.append(turn)
            if turn_writer is not None:
                turn_writer.write(turn)

    metrics = metrics_for_env(env, window=metric_window)
    label = (
        metrics.label(label_threshold) if label_threshold is not None else -1
    )
    summary = EpisodeSummary(
        episode_id=episode_id,
        condition_id=condition.condition_id,
        factors=condition.factors,
        seed=seed,
        action_collusion_score=metrics.collusion_index,
        action_collusion_label=label,
        text_collusion_score=text_collusion_score,
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
    )
    return EpisodeResult(summary=summary, turns=turns, env=env)
