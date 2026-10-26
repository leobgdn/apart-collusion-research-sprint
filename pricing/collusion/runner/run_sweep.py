"""Resumable, wall-clock-guarded, balanced-batch sweep driver.

* Resumable: every run is keyed by (condition_id, seed)
  and completed runs are skipped on restart. On resume the turn log is first
  compacted to drop turns of any episode without a completed summary, so a
  crash between writing turns and writing the summary cannot duplicate turns.
* Wall-clock guard (Phase 1 override 1, replacing the dollar budget): no new
  episode is launched once it could not finish before the hard stop, and any
  episode still running at the hard stop is discarded whole. Everything already
  logged stays valid.
* Anytime design (Phase 1 override 8): seeds are grouped into balanced batches.
  Batch b holds the same `seeds_per_batch` new seeds for every cell, and work
  is queued batch by batch, so stopping after any completed batch leaves a
  balanced design. `complete_batches` says which batches are usable.
* Stop-the-line monitors (Phase 1 Section 8): parse-failure rate above the
  limit, any reasoning block, or any prompt-token mismatch stops the sweep.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from collusion.agents.scripted_agents import make_scripted_agent
from collusion.envs.pricing.conditions import Condition, pilot_design
from collusion.envs.pricing.economics import Benchmarks, PricingParams, compute_benchmarks
from collusion.envs.pricing.pricing_env import EnvConfig
from collusion.runner.logging_io import (
    TurnLogWriter,
    completed_episode_keys,
    episode_id_for,
    read_episode_summaries,
    read_turn_log,
    write_episode_summary,
)
from collusion.runner.run_episode import DeadlineReached, run_episode

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "config" / "experiment.yaml"


@dataclass
class WallClockGuard:
    """Hard stop for generation. `deadline` is a time.time() epoch (or None)."""

    deadline: float | None = None
    # Safety factor on the typical episode duration when deciding to launch.
    launch_margin: float = 1.2
    durations: list[float] = field(default_factory=list)

    def record(self, seconds: float) -> None:
        self.durations.append(seconds)

    def expected_episode_seconds(self) -> float:
        return statistics.median(self.durations) if self.durations else 0.0

    def passed(self) -> bool:
        return self.deadline is not None and time.time() >= self.deadline

    def may_launch(self) -> bool:
        """False once a new episode could not plausibly finish before the stop."""
        if self.deadline is None:
            return True
        needed = self.expected_episode_seconds() * self.launch_margin
        return time.time() + needed < self.deadline


def seeds_for_batch(batch: int, seeds_per_batch: int) -> list[int]:
    """Batch b owns seeds b*k+1 .. (b+1)*k. Disjoint across batches."""
    start = batch * seeds_per_batch + 1
    return list(range(start, start + seeds_per_batch))


def batch_of_seed(seed: int, seeds_per_batch: int) -> int:
    return (seed - 1) // seeds_per_batch


@dataclass
class SweepConfig:
    """The parsed experiment.yaml (plus CLI overrides)."""

    raw: dict[str, Any]
    run_id: str
    params: PricingParams
    env_config: EnvConfig
    conditions: list[Condition]
    seeds_per_batch: int
    n_batches: int
    metric_window: int
    label_threshold: float | None
    max_parse_retries: int
    max_parse_failure_rate: float
    logs_dir: Path
    episodes_dir: Path
    scripted_noise_sd: float = 0.05
    concurrency: int = 1
    deadline: float | None = None
    # Parse-failure rate is only judged after this many turns, so one bad turn
    # in the first episode cannot stop a sweep.
    min_turns_for_rate: int = 400

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
    def seeds(self) -> list[int]:
        return [
            s for b in range(self.n_batches) for s in seeds_for_batch(b, self.seeds_per_batch)
        ]

    @property
    def total_episodes(self) -> int:
        return len(self.conditions) * len(self.seeds)

    @property
    def log_path(self) -> Path:
        return self.logs_dir / f"{self.run_id}.jsonl"

    def work_queue(self) -> list[tuple[int, Condition, int]]:
        """(batch, condition, seed) in batch-major order."""
        return [
            (b, c, s)
            for b in range(self.n_batches)
            for s in seeds_for_batch(b, self.seeds_per_batch)
            for c in self.conditions
        ]


def load_config(path: str | Path = DEFAULT_CONFIG, *, pilot: bool = True) -> SweepConfig:
    """Read experiment.yaml into a SweepConfig."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    design = raw["design"]
    block = design["pilot"] if pilot else design["main"]
    conditions = pilot_design(include_horizon=bool(design.get("include_horizon", True)))
    paths = raw.get("paths", {})
    limits = raw["limits"]
    return SweepConfig(
        raw=raw,
        run_id=raw.get("run_id", "sweep"),
        params=PricingParams(**raw["params"]),
        env_config=EnvConfig(
            **{
                **raw["env"],
                "total_periods": block["total_periods"],
                "price_grid": (
                    parse_price_grid(raw["env"]["price_grid"])
                    if raw["env"].get("price_grid")
                    else None
                ),
            }
        ),
        conditions=conditions,
        seeds_per_batch=int(block["seeds_per_batch"]),
        n_batches=int(block["n_batches"]),
        metric_window=int(block["metric_window"]),
        label_threshold=raw["metrics"].get("label_threshold"),
        max_parse_retries=int(limits["max_parse_retries"]),
        max_parse_failure_rate=float(limits["max_parse_failure_rate"]),
        logs_dir=Path(paths.get("logs_dir", "data/logs")),
        episodes_dir=Path(paths.get("episodes_dir", "data/episodes")),
        scripted_noise_sd=float(raw.get("scripted", {}).get("noise_sd", 0.05)),
        concurrency=int(limits.get("concurrency", 1)),
    )


AgentFactory = Callable[[Condition, int, Benchmarks], list[Any]]


def scripted_agent_factory(
    policy: str, noise_sd: float, params: PricingParams
) -> AgentFactory:
    """Agent factory for the free, scripted controls."""

    def build(condition: Condition, seed: int, benchmarks: Benchmarks) -> list[Any]:
        return [
            make_scripted_agent(
                policy,
                benchmarks,
                noise_sd=noise_sd,
                agent_id=f"seller_{i}",
                params=params,
            )
            for i in range(params.n_sellers)
        ]

    return build


def llm_agent_factory(
    raw_config: dict[str, Any], n_sellers: int, endpoint: str | None = None
) -> AgentFactory:
    """Agent factory for the local model (Phase 1 Stage 2)."""
    from collusion.agents.llm_agent import (
        ChatTemplate,
        LLMAgent,
        OpenAICompatClient,
        SamplingSettings,
    )

    agent_cfg = raw_config["models"]["agent"]
    model = agent_cfg["name"]
    client = OpenAICompatClient(
        endpoint, model=model, timeout_s=float(agent_cfg.get("timeout_s", 120))
    )
    template = ChatTemplate.for_model(model)
    sampling = SamplingSettings(
        temperature=float(agent_cfg["temperature"]),
        top_p=float(agent_cfg["top_p"]),
        max_tokens=int(agent_cfg["max_new_tokens"]),
        enable_thinking=bool(agent_cfg.get("enable_thinking", False)),
    )

    def build(condition: Condition, seed: int, benchmarks: Benchmarks) -> list[Any]:
        return [
            LLMAgent(
                client=client,
                episode_seed=seed,
                agent_index=i,
                sampling=sampling,
                template=template,
                agent_id=f"seller_{i}",
                model_name=model,
            )
            for i in range(n_sellers)
        ]

    return build


@dataclass
class SweepReport:
    """What a sweep did, for the gate report."""

    run_id: str
    attempted: int = 0
    skipped: int = 0
    completed: int = 0
    failed: list[tuple[str, int, str]] = field(default_factory=list)
    discarded_at_deadline: list[str] = field(default_factory=list)
    not_launched: int = 0
    total_turns: int = 0
    parse_failures: int = 0
    reasoning_blocks: int = 0
    prompt_token_mismatches: int = 0
    scores: list[float] = field(default_factory=list)
    stop_reason: str | None = None
    wall_seconds: float = 0.0

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.total_turns if self.total_turns else 0.0


def compact_turn_log(log_path: Path, episodes_dir: Path) -> int:
    """Drop turns of episodes with no completed summary. Returns rows dropped."""
    if not log_path.exists():
        return 0
    done = {
        episode_id_for(c, s) for c, s in completed_episode_keys(episodes_dir)
    }
    kept, dropped = [], 0
    for row in read_turn_log(log_path):
        if row["episode_id"] in done:
            kept.append(row)
        else:
            dropped += 1
    if dropped:
        tmp = log_path.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8", newline="\n") as fh:
            for row in kept:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(log_path)
    return dropped


def complete_batches(
    summaries: list[dict], conditions: list[Condition], seeds_per_batch: int
) -> list[int]:
    """Batches in which every cell has every one of the batch's seeds completed."""
    have = {
        (r["condition_id"], int(r["seed"])) for r in summaries if r.get("completed")
    }
    batches = sorted({batch_of_seed(s, seeds_per_batch) for _, s in have})
    return [
        b
        for b in batches
        if all(
            (c.condition_id, s) in have
            for c in conditions
            for s in seeds_for_batch(b, seeds_per_batch)
        )
    ]


def run_sweep(
    config: SweepConfig,
    agent_factory: AgentFactory,
    *,
    resume: bool = True,
    verbose: bool = True,
    parallel_agents: bool = False,
) -> SweepReport:
    """Run every (condition, seed) cell that is not already finished."""
    benchmarks = config.benchmarks
    guard = WallClockGuard(deadline=config.deadline)
    episodes_dir = config.run_episodes_dir
    log_path = config.log_path
    sweep_started = time.time()

    if not resume and (log_path.exists() or episodes_dir.exists()):
        # Appending to an existing log would duplicate turns and corrupt every
        # per-turn statistic downstream. Refuse rather than silently overwrite.
        raise FileExistsError(
            f"run_id {config.run_id!r} already has data at {log_path} / "
            f"{episodes_dir}. Pass a new --run-id, or run `python tasks.py clean`."
        )
    if resume:
        dropped = compact_turn_log(log_path, episodes_dir)
        if dropped and verbose:
            print(f"[{config.run_id}] dropped {dropped} turns of unfinished episodes")

    done = completed_episode_keys(episodes_dir) if resume else set()
    report = SweepReport(run_id=config.run_id)
    queue = [
        (b, c, s) for (b, c, s) in config.work_queue() if (c.condition_id, s) not in done
    ]
    report.skipped = config.total_episodes - len(queue)
    if verbose:
        print(
            f"[{config.run_id}] {len(config.conditions)} cells x {len(config.seeds)} "
            f"seeds ({config.n_batches} batches of {config.seeds_per_batch}); "
            f"{report.skipped} already complete, {len(queue)} to run, "
            f"concurrency {config.concurrency}"
        )

    stop = threading.Event()
    lock = threading.Lock()

    def one(batch: int, condition: Condition, seed: int, writer) -> Any:
        return run_episode(
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
            deadline=config.deadline,
            parallel_agents=parallel_agents,
            batch=batch,
        )

    def handle(fut: Future, key: tuple[int, Condition, int]) -> None:
        batch, condition, seed = key
        eid = episode_id_for(condition.condition_id, seed)
        try:
            result = fut.result()
        except DeadlineReached:
            report.discarded_at_deadline.append(eid)
            report.stop_reason = report.stop_reason or "wall-clock deadline reached"
            stop.set()
            return
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            report.failed.append((condition.condition_id, seed, repr(exc)))
            print(f"  FAILED {eid}: {exc!r}", file=sys.stderr)
            return
        s = result.summary
        write_episode_summary(s, episodes_dir)
        guard.record(s.wall_seconds)
        report.completed += 1
        report.total_turns += len(result.turns)
        report.parse_failures += s.parse_failures
        report.reasoning_blocks += s.reasoning_blocks
        report.prompt_token_mismatches += sum(
            1 for t in result.turns if t.call_info.get("prompt_token_mismatch")
        )
        report.scores.append(s.action_collusion_score)
        if verbose:
            print(
                f"  [{report.completed}/{len(queue)}] b{batch} {s.episode_id} "
                f"Delta={s.action_collusion_score:+.3f} cat={s.category} "
                f"pf={s.parse_failures} {s.wall_seconds:.0f}s"
            )
        if report.reasoning_blocks:
            report.stop_reason = "reasoning block found in model output"
            stop.set()
        elif report.prompt_token_mismatches:
            report.stop_reason = "logged prompt does not match server prompt tokens"
            stop.set()
        elif (
            report.total_turns >= min(config.min_turns_for_rate, len(queue) * 2)
            and report.parse_failure_rate > config.max_parse_failure_rate
        ):
            report.stop_reason = (
                f"parse-failure rate {report.parse_failure_rate:.1%} exceeds "
                f"{config.max_parse_failure_rate:.1%}"
            )
            stop.set()

    with TurnLogWriter(log_path) as writer, ThreadPoolExecutor(
        max_workers=max(1, config.concurrency)
    ) as pool:
        in_flight: dict[Future, tuple[int, Condition, int]] = {}
        position = 0
        while position < len(queue) or in_flight:
            while (
                position < len(queue)
                and len(in_flight) < config.concurrency
                and not stop.is_set()
            ):
                if not guard.may_launch():
                    report.stop_reason = report.stop_reason or (
                        "wall-clock: no time left to finish another episode"
                    )
                    stop.set()
                    break
                key = queue[position]
                position += 1
                report.attempted += 1
                in_flight[pool.submit(one, *key, writer)] = key
            if not in_flight:
                break
            finished, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in finished:
                with lock:
                    handle(fut, in_flight.pop(fut))
        report.not_launched = len(queue) - position

    if report.stop_reason:
        print(f"STOPPED: {report.stop_reason}", file=sys.stderr)
    report.wall_seconds = time.time() - sweep_started
    if verbose:
        summaries = read_episode_summaries(episodes_dir)
        batches = complete_batches(summaries, config.conditions, config.seeds_per_batch)
        print(
            f"[{config.run_id}] completed {report.completed}, skipped "
            f"{report.skipped}, failed {len(report.failed)}, discarded at deadline "
            f"{len(report.discarded_at_deadline)}, not launched {report.not_launched}; "
            f"parse-failure rate {report.parse_failure_rate:.2%}; complete balanced "
            f"batches: {batches}; {report.wall_seconds / 60:.1f} min"
        )
    return report


def parse_deadline(text: str | None, minutes: float | None) -> float | None:
    """--deadline 'YYYY-MM-DDTHH:MM' (local time) and/or --minutes N; earliest wins."""
    candidates = []
    if text:
        candidates.append(datetime.fromisoformat(text).timestamp())
    if minutes is not None:
        candidates.append(time.time() + minutes * 60)
    return min(candidates) if candidates else None


def parse_price_grid(spec: str) -> tuple[float, ...]:
    """'lo:hi:step' -> (lo, lo+step, ..., hi), rounded to 2 decimals."""
    lo, hi, step = (float(x) for x in spec.split(":"))
    n = int(round((hi - lo) / step))
    return tuple(round(lo + k * step, 2) for k in range(n + 1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--policy", default="colluding", help="scripted policy to run")
    ap.add_argument("--llm", action="store_true", help="use the local model")
    ap.add_argument("--main", action="store_true", help="main-sweep design (gated)")
    ap.add_argument("--i-have-gate2-approval", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--seeds-per-batch", type=int, default=None)
    ap.add_argument("--n-batches", type=int, default=None)
    ap.add_argument("--total-periods", type=int, default=None)
    ap.add_argument("--no-horizon", action="store_true", help="drop F3 (4 cells + placebo)")
    ap.add_argument("--deadline", default=None, help="hard stop, local ISO time")
    ap.add_argument("--minutes", type=float, default=None, help="hard stop in N minutes")
    ap.add_argument("--thinking", action="store_true",
                    help="Gate 2 diagnostic: thinking ON (overrides the config)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--price-grid", default=None,
                    help="discrete price menu 'lo:hi:step', e.g. 1.00:2.50:0.05")
    args = ap.parse_args(argv)

    if args.main and not args.i_have_gate2_approval:
        print("The main sweep needs Gate 2 approval (Phase 1 Section 7).", file=sys.stderr)
        return 1

    config = load_config(args.config, pilot=not args.main)
    if args.run_id:
        config.run_id = args.run_id
    if args.concurrency:
        config.concurrency = args.concurrency
    if args.seeds_per_batch:
        config.seeds_per_batch = args.seeds_per_batch
    if args.n_batches:
        config.n_batches = args.n_batches
    if args.total_periods:
        config.env_config.total_periods = args.total_periods
    if args.no_horizon:
        config.conditions = pilot_design(include_horizon=False)
    config.deadline = parse_deadline(args.deadline, args.minutes)
    agent_cfg = config.raw["models"]["agent"]
    if args.thinking:
        agent_cfg["enable_thinking"] = True
    if args.max_tokens:
        agent_cfg["max_new_tokens"] = args.max_tokens
    if args.price_grid:
        config.env_config.price_grid = parse_price_grid(args.price_grid)

    if args.llm:
        factory = llm_agent_factory(config.raw, config.params.n_sellers, args.endpoint)
    else:
        factory = scripted_agent_factory(args.policy, config.scripted_noise_sd, config.params)

    report = run_sweep(
        config, factory, resume=not args.no_resume, parallel_agents=args.llm
    )
    return 0 if not report.failed and report.stop_reason in (None,) else 1


if __name__ == "__main__":
    raise SystemExit(main())
