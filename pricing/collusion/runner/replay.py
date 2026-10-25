"""Rebuild an episode from its saved log, and verify the log round-trips.

Two jobs:

1. The acceptance test in Section 6.4 ("log round-trip test passes"). We replay
   the logged actions through a fresh environment and re-render each prompt from
   the reconstructed state. If the result is not byte-identical to the
   `prompt_messages` we logged, then the log does not actually describe what the
   agent saw, and both the probe layer and the text judge would be reading
   fiction. Section 12 lists that mismatch as a stop-the-line condition.
2. The input path for `probes/extract.py`, which needs the exact prompt for each
   turn in order to do a teacher-forced forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from collusion.agents.parsing import ParsedAction
from collusion.envs.pricing import prompts
from collusion.envs.pricing.conditions import condition_from_id
from collusion.envs.pricing.economics import PricingParams
from collusion.envs.pricing.pricing_env import EnvConfig, PricingEnvironment
from collusion.envs.pricing.state import Observation


@dataclass
class ReplayedTurn:
    """One turn, as reconstructed from the log."""

    episode_id: str
    turn: int
    period: int
    agent_index: int
    agent_id: str
    model: str
    observation: Observation
    prompt_messages: list[dict[str, str]]
    logged_prompt_messages: list[dict[str, str]]
    response_text: str

    @property
    def matches(self) -> bool:
        return self.prompt_messages == self.logged_prompt_messages


@dataclass
class ReplayMismatch:
    """A turn whose re-rendered prompt differs from what was logged."""

    episode_id: str
    turn: int
    agent_index: int
    detail: str

    def __str__(self) -> str:
        return (
            f"{self.episode_id} turn {self.turn} (agent {self.agent_index}): "
            f"{self.detail}"
        )


def _action_from_logged(payload: dict[str, Any]) -> ParsedAction:
    """Rebuild a ParsedAction from its logged dict form."""
    return ParsedAction(
        price=payload.get("price"),
        plan=payload.get("plan", ""),
        insight=payload.get("insight", ""),
        message=payload.get("message", ""),
        ok=bool(payload.get("ok", False)),
        failure_reason=payload.get("failure_reason"),
    )


def group_turns_by_episode(turns: Iterable[dict]) -> dict[str, list[dict]]:
    """Bucket raw turn records by episode, each sorted by (period, agent_index)."""
    grouped: dict[str, list[dict]] = {}
    for row in turns:
        grouped.setdefault(row["episode_id"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda r: (r.get("period", 0), r.get("agent_index", 0)))
    return grouped


def replay_episode(
    turn_rows: list[dict],
    *,
    params: PricingParams | None = None,
    total_periods: int | None = None,
    demand_noise_sd: float = 0.0,
    price_grid: tuple[float, ...] | None = None,
) -> list[ReplayedTurn]:
    """Replay one episode's logged turns, re-rendering every prompt.

    `turn_rows` must be all turns of a single episode. Parameters default to the
    environment defaults; pass the values from the episode summary when they
    were overridden, otherwise the replay is not comparing like with like.
    """
    if not turn_rows:
        return []
    episode_ids = {r["episode_id"] for r in turn_rows}
    if len(episode_ids) != 1:
        raise ValueError(f"expected one episode, got {sorted(episode_ids)}")

    rows = sorted(turn_rows, key=lambda r: (r.get("period", 0), r.get("agent_index", 0)))
    condition = condition_from_id(rows[0]["condition_id"])
    seed = int(rows[0]["seed"])
    params = params or PricingParams()

    by_period: dict[int, list[dict]] = {}
    for row in rows:
        by_period.setdefault(int(row["period"]), []).append(row)
    periods = sorted(by_period)
    if total_periods is None:
        total_periods = len(periods)

    env = PricingEnvironment(
        condition=condition,
        params=params,
        config=EnvConfig(
            total_periods=total_periods,
            demand_noise_sd=demand_noise_sd,
            price_grid=price_grid,
        ),
        seed=seed,
    )

    out: list[ReplayedTurn] = []
    for period in periods:
        period_rows = sorted(by_period[period], key=lambda r: r["agent_index"])
        if len(period_rows) != params.n_sellers:
            raise ValueError(
                f"period {period} has {len(period_rows)} turns, expected "
                f"{params.n_sellers}"
            )
        observations = env.observations()
        for row in period_rows:
            i = int(row["agent_index"])
            obs = observations[i]
            rendered = prompts.render(condition, obs)
            out.append(
                ReplayedTurn(
                    episode_id=row["episode_id"],
                    turn=int(row["turn"]),
                    period=period,
                    agent_index=i,
                    agent_id=row.get("agent_id", f"seller_{i}"),
                    model=row.get("model", "unknown"),
                    observation=obs,
                    prompt_messages=rendered.messages,
                    logged_prompt_messages=row["prompt_messages"],
                    response_text=row.get("response_text", ""),
                )
            )
        env.step([_action_from_logged(r["parsed_action"]) for r in period_rows])
    return out


def verify_round_trip(
    turns: Iterable[dict],
    *,
    params: PricingParams | None = None,
    episode_periods: dict[str, int] | None = None,
    price_grid: tuple[float, ...] | None = None,
) -> list[ReplayMismatch]:
    """Replay every episode in a log; return the mismatches (empty == pass)."""
    mismatches: list[ReplayMismatch] = []
    for episode_id, rows in group_turns_by_episode(turns).items():
        total = (episode_periods or {}).get(episode_id)
        replayed = replay_episode(
            rows, params=params, total_periods=total, price_grid=price_grid
        )
        for turn in replayed:
            if turn.matches:
                continue
            mismatches.append(
                ReplayMismatch(
                    episode_id=episode_id,
                    turn=turn.turn,
                    agent_index=turn.agent_index,
                    detail=_describe_difference(
                        turn.logged_prompt_messages, turn.prompt_messages
                    ),
                )
            )
    return mismatches


def _describe_difference(
    logged: list[dict[str, str]], rebuilt: list[dict[str, str]]
) -> str:
    """A short, readable account of the first difference found."""
    if len(logged) != len(rebuilt):
        return f"message count {len(logged)} logged vs {len(rebuilt)} rebuilt"
    for idx, (a, b) in enumerate(zip(logged, rebuilt)):
        if a.get("role") != b.get("role"):
            return f"message {idx} role {a.get('role')!r} vs {b.get('role')!r}"
        if a.get("content") != b.get("content"):
            left, right = a.get("content", ""), b.get("content", "")
            for pos, (c1, c2) in enumerate(zip(left, right)):
                if c1 != c2:
                    return (
                        f"message {idx} ({a.get('role')}) first differs at char "
                        f"{pos}: logged {left[pos:pos + 40]!r} vs rebuilt "
                        f"{right[pos:pos + 40]!r}"
                    )
            return (
                f"message {idx} ({a.get('role')}) differs in length: "
                f"{len(left)} logged vs {len(right)} rebuilt"
            )
    return "no difference found"
