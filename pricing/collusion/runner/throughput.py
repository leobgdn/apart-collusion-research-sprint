"""Dedicated throughput window (Track A addendum Section 2).

    python -m collusion.runner.throughput [--seconds-per-level 180]

Runs real pilot-style LLM episodes at concurrency 8, 16 and 32 for a fixed time
each (at most 10 minutes in total), with ~/collusion_shared/GPU_BUSY present for
exactly the duration of the measurement so the other session stays off the GPU.

Concurrency here means concurrent *episodes*; each sends its two sellers'
requests in parallel, so up to 2x that many requests are in flight.

Measured per level: aggregate prefill and decode tokens/s, requests/s, request
latency, and from those rounds/s, the time per 80-round episode
(80 / (rounds_per_s / concurrency)) and episodes/hour (rounds_per_s * 3600 / 80).
Episodes in flight at the end of each level are discarded by the wall-clock
guard; only the token counters matter here. Output: reports/throughput.json.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from collusion.agents.llm_agent import SHARED_DIR, STATS
from collusion.runner.run_sweep import load_config, llm_agent_factory, run_sweep

ROOT = Path(__file__).resolve().parents[2]
GPU_BUSY = SHARED_DIR / "GPU_BUSY"
LEVELS = (8, 16, 32)
MAX_TOTAL_SECONDS = 600


def gpu_memory() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"


def measure(seconds_per_level: float, periods: int, endpoint: str | None) -> dict:
    if seconds_per_level * len(LEVELS) > MAX_TOTAL_SECONDS:
        raise ValueError("the dedicated window must stay within 10 minutes")
    scratch = ROOT / "data" / "throughput"
    results = {"periods_per_episode": periods, "levels": []}

    SHARED_DIR.mkdir(parents=True, exist_ok=True)
    GPU_BUSY.touch()
    results["gpu_busy_created"] = time.strftime("%Y-%m-%d %H:%M:%S")
    try:
        for level in LEVELS:
            config = load_config(pilot=True)
            config.run_id = f"throughput_c{level}_{int(time.time())}"
            config.logs_dir = scratch / "logs"
            config.episodes_dir = scratch / "episodes"
            config.concurrency = level
            config.n_batches = 50  # more work than can finish; the clock stops it
            config.deadline = time.time() + seconds_per_level
            factory = llm_agent_factory(config.raw, config.params.n_sellers, endpoint)
            STATS.reset()
            run_sweep(config, factory, parallel_agents=True, verbose=False)
            snap = STATS.snapshot()
            rounds_per_s = snap["requests_per_s"] / config.params.n_sellers
            per_episode_rounds_per_s = rounds_per_s / level
            row = {
                "concurrency_episodes": level,
                **{k: round(v, 3) for k, v in snap.items()},
                "rounds_per_s": round(rounds_per_s, 3),
                "seconds_per_episode": round(periods / per_episode_rounds_per_s, 1)
                if per_episode_rounds_per_s
                else None,
                "episodes_per_hour": round(rounds_per_s * 3600 / periods, 1),
                "gpu_memory_after": gpu_memory(),
            }
            results["levels"].append(row)
            print(json.dumps(row))
    finally:
        GPU_BUSY.unlink(missing_ok=True)
        results["gpu_busy_deleted"] = time.strftime("%Y-%m-%d %H:%M:%S")
        shutil.rmtree(scratch, ignore_errors=True)
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds-per-level", type=float, default=180)
    ap.add_argument("--periods", type=int, default=80)
    ap.add_argument("--endpoint", default=None)
    args = ap.parse_args(argv)
    results = measure(args.seconds_per_level, args.periods, args.endpoint)
    out = ROOT / "reports" / "throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
