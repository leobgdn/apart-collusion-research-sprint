"""Resumable, budget-guarded sweep driver.

Section 3 rules 3 and 5: every run is keyed by (condition_id, seed), completed
runs are skipped on restart, and the driver refuses to proceed if the projected
spend would breach the cap.

At Stage 1 only the scripted agent factory exists, so a sweep run through here
costs nothing. The LLM agent factory is Stage 2 work and is deliberately absent
until Gate 1 is cleared -- `build_agents` raises rather than quietly falling
back to something cheaper, which would make a sweep silently not the experiment
that was approved.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from collusion.agents.scripted_agents import make_scripted_agent
from collusion.envs.pricing.conditions import (
    Condition,
    full_factorial,
    one_factor_at_a_time,
)
from collusion.envs.pricing.economics import Benchmarks, PricingParams, compute_benchmarks
from collusion.envs.pricing.pricing_env import EnvConfig
from collusion.runner.logging_io import (
    TurnLogWriter,
    completed_episode_keys,
    episode_id_for,
    write_episode_summary,
)
from collusion.runner.run_episode import run_episode

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"


class BudgetExceeded(RuntimeError):
    """Raised when the running or projected cost would breach the hard cap."""


@dataclass
class BudgetGuard:
    """Tracks spend and refuses to let a sweep run past the cap."""

    hard_cap_usd: float
    spent_usd: float = 0.0
    episodes_done: int = 0

    def record(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd
        self.episodes_done += 1
        if self.spent_usd > self.hard_cap_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f}, over the ${self.hard_cap_usd:.2f} cap"
            )

    @property
    def mean_cost(self) -> float:
        return self.spent_usd / self.episodes_done if self.episodes_done else 0.0

    def project(self, episodes_remaining: int) -> float:
        return self.spent_usd + self.mean_cost * episodes_remaining

    def check_projection(self, episodes_remaining: int) -> None:
        projected = self.project(episodes_remaining)
        if projected > self.hard_cap_usd:
            raise BudgetExceeded(
                f"projected total ${projected:.4f} for {episodes_remaining} more "
                f"episodes exceeds the ${self.hard_cap_usd:.2f} cap "
                f"(spent ${self.spent_usd:.4f} over {self.episodes_done})"
            )


@dataclass
class SweepConfig:
    """The parsed experiment.yaml."""

    raw: dict[str, Any]
    run_id: str
    params: PricingParams
    env_config: EnvConfig
    conditions: list[Condition]
    seeds: list[int]
    metric_window: int
    label_threshold: float
    hard_cap_usd: float
    max_parse_retries: int
    max_parse_failure_rate: float
    logs_dir: Path
    episodes_dir: Path
    scripted_noise_sd: float = 0.05

    @property
    def benchmarks(self) -> Benchmarks:
        return compute_benchmarks(self.params)

    @property
    def run_episodes_dir(self) -> Path:
        """Episode summaries live under the run_id.

        Resume keys are (condition_id, seed), which are *not* unique across
        sweeps: two runs with different agents or a changed threshold share the
        same cells. Without this namespacing, a second sweep silently skips
        every episode and reports success while reusing the first sweep's
        results.
        """
        return self.episodes_dir / self.run_id

    @property
    def total_episodes(self) -> int:
        return len(self.conditions) * len(self.seeds)

    @property
    def log_path(self) -> Path:
        return self.logs_dir / f"{self.run_id}.jsonl"


def load_config(path: str | Path = DEFAULT_CONFIG, *, pilot: bool = False) -> SweepConfig:
    """Read experiment.yaml into a SweepConfig."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    design = raw["design"]
    plan = "one_factor_at_a_time" if pilot else design.get("plan", "full_factorial")
    conditions = {
        "full_factorial": full_factorial,
        "one_factor_at_a_time": one_factor_at_a_time,
    }[plan]()
    seeds = design["pilot_seeds"] if pilot else design["seeds"]
    paths = raw.get("paths", {})
    return SweepConfig(
        raw=raw,
        run_id=raw.get("run_id", "sweep"),
        params=PricingParams(**raw["params"]),
        env_config=EnvConfig(**raw["env"]),
        conditions=conditions,
        seeds=list(seeds),
        metric_window=raw["metrics"]["window_periods"],
        label_threshold=raw["metrics"]["label_threshold"],
        hard_cap_usd=float(raw["budget"]["hard_cap_usd"]),
        max_parse_retries=int(raw["limits"]["max_parse_retries"]),
        max_parse_failure_rate=float(raw["limits"]["max_parse_failure_rate"]),
        logs_dir=Path(paths.get("logs_dir", "data/logs")),
        episodes_dir=Path(paths.get("episodes_dir", "data/episodes")),
        scripted_noise_sd=float(raw.get("scripted", {}).get("noise_sd", 0.05)),
    )


AgentFactory = Callable[[Condition, int, Benchmarks], list[Any]]


def scripted_agent_factory(
    policy: str, noise_sd: float, n_sellers: int
) -> AgentFactory:
    """Agent factory for the free, scripted controls."""

    def build(condition: Condition, seed: int, benchmarks: Benchmarks) -> list[Any]:
        return [
            make_scripted_agent(
                policy, benchmarks, noise_sd=noise_sd, agent_id=f"seller_{i}"
            )
            for i in range(n_sellers)
        ]

    return build


def llm_agent_factory(*args, **kwargs) -> AgentFactory:
    """Stage 2 work, gated on Gate 1."""
    raise NotImplementedError(
        "The LLM agent wrapper is Stage 2 and requires human approval at Gate 1 "
        "(Section 6). Use --policy for the scripted controls until then."
    )


@dataclass
class SweepReport:
    """What a sweep did, for the gate report."""

    run_id: str
    attempted: int = 0
    skipped: int = 0
    completed: int = 0
    failed: list[tuple[str, int, str]] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_turns: int = 0
    parse_failures: int = 0
    scores: list[float] = field(default_factory=list)
    budget_aborted: bool = False

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.total_turns if self.total_turns else 0.0


def run_sweep(
    config: SweepConfig,
    agent_factory: AgentFactory,
    *,
    resume: bool = True,
    verbose: bool = True,
) -> SweepReport:
    """Run every (condition, seed) cell that is not already finished."""
    benchmarks = config.benchmarks
    guard = BudgetGuard(hard_cap_usd=config.hard_cap_usd)
    episodes_dir = config.run_episodes_dir
    log_path = config.log_path

    if not resume and (log_path.exists() or episodes_dir.exists()):
        # Appending to an existing log would duplicate turns and corrupt every
        # per-turn statistic downstream. Refuse rather than silently overwrite.
        raise FileExistsError(
            f"run_id {config.run_id!r} already has data at {log_path} / "
            f"{episodes_dir}. Pass a new --run-id, or run `python tasks.py clean`."
        )

    done = completed_episode_keys(episodes_dir) if resume else set()
    report = SweepReport(run_id=config.run_id)
    pending = [
        (c, s)
        for c in config.conditions
        for s in config.seeds
        if (c.condition_id, s) not in done
    ]
    report.skipped = config.total_episodes - len(pending)
    if verbose:
        print(
            f"[{config.run_id}] {config.total_episodes} cells x seeds; "
            f"{report.skipped} already complete, {len(pending)} to run"
        )

    with TurnLogWriter(log_path) as writer:
        for index, (condition, seed) in enumerate(pending):
            report.attempted += 1
            remaining = len(pending) - index - 1
            try:
                guard.check_projection(remaining)
            except BudgetExceeded as exc:
                print(f"ABORTING: {exc}", file=sys.stderr)
                break

            try:
                result = run_episode(
                    condition=condition,
                    seed=seed,
                    agents=agent_factory(condition, seed, benchmarks),
                    run_id=config.run_id,
                    params=config.params,
                    env_config=config.env_config,
                    metric_window=config.metric_window,
                    label_threshold=config.label_threshold,
                    max_parse_retries=config.max_parse_retries,
                    turn_writer=writer,
                )
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                report.failed.append((condition.condition_id, seed, repr(exc)))
                print(
                    f"  FAILED {episode_id_for(condition.condition_id, seed)}: {exc!r}",
                    file=sys.stderr,
                )
                continue

            write_episode_summary(result.summary, episodes_dir)
            report.completed += 1
            report.total_turns += len(result.turns)
            report.parse_failures += result.summary.parse_failures
            report.total_cost_usd += result.summary.total_cost_usd
            report.scores.append(result.summary.action_collusion_score)
            try:
                guard.record(result.summary.total_cost_usd)
            except BudgetExceeded as exc:
                # The episode is already paid for and logged, so keep it; just
                # stop before spending more (Section 3 rule 3: abort and report).
                print(f"ABORTING: {exc}", file=sys.stderr)
                report.budget_aborted = True
                break

            if verbose:
                s = result.summary
                print(
                    f"  [{report.completed}/{len(pending)}] {s.episode_id} "
                    f"Delta={s.action_collusion_score:+.3f} cat={s.category} "
                    f"cost=${report.total_cost_usd:.4f}"
                )

            if report.parse_failure_rate > config.max_parse_failure_rate:
                print(
                    f"STOPPING: parse-failure rate {report.parse_failure_rate:.1%} "
                    f"exceeds the {config.max_parse_failure_rate:.1%} limit "
                    "(Section 6.3)",
                    file=sys.stderr,
                )
                break

    if verbose:
        print(
            f"[{config.run_id}] completed {report.completed}, skipped "
            f"{report.skipped}, failed {len(report.failed)}, "
            f"total cost ${report.total_cost_usd:.4f}, "
            f"parse-failure rate {report.parse_failure_rate:.2%}"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument(
        "--policy", default="colluding", help="which scripted policy to run"
    )
    ap.add_argument(
        "--llm",
        action="store_true",
        help="use LLM agents instead of the scripted controls (Stage 2, gated)",
    )
    ap.add_argument("--pilot", action="store_true", help="use the pilot design/seeds")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    config = load_config(args.config, pilot=args.pilot)
    if args.run_id:
        config.run_id = args.run_id
    if args.llm:
        llm_agent_factory()  # raises until Gate 1 is cleared

    report = run_sweep(
        config,
        scripted_agent_factory(
            args.policy, config.scripted_noise_sd, config.params.n_sellers
        ),
        resume=not args.no_resume,
    )
    return 0 if not report.failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
