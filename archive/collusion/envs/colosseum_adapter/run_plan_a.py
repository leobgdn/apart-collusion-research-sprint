"""Budget-guarded, resumable driver for the Colosseum sweep (Plan A).

`run.py` always starts a *new* timestamped output root per invocation; it has
no built-in way to stop partway through a seed list. `resume.py` does the
in-place resuming Section 3 rule 5 asks for, but it still runs its whole given
seed list in one call once started. So the budget guard here works at the
granularity Colosseum actually exposes: escalate the seed list step by step,
check projected cost (from real $ measured on the seeds already run, via
cost_tracking.py) *before* each escalation, and refuse to start the next step
if the projection would breach the cap. This is coarser than true per-episode
interruption, but at the estimated sweep cost (~$1-10, see DECISIONS.md) a
single step overrunning the cap by more than a trivial margin is not a
realistic failure mode -- the point of the guard is to catch a configuration
mistake (wrong model, runaway max_tokens, a pricing assumption that was wrong),
not to shave fractions of a cent.

Usage:
    python run_plan_a.py smoke   --config <path> [--out-dir <override>]
    python run_plan_a.py escalate --root <existing root> --config <path> \\
        --cap-usd <hard cap>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cost_tracking import FIREWORKS_PRICE_PER_MILLION_TOKENS, scan_sweep_cost  # noqa: E402

THIS_DIR = Path(__file__).resolve().parent


class BudgetExceeded(RuntimeError):
    pass


def _colosseum_root() -> Path:
    import os

    env_path = os.environ.get("COLOSSEUM_ROOT")
    root = Path(env_path).expanduser().resolve() if env_path else (
        THIS_DIR.parents[3] / "colosseum"
    )
    if not (root / "experiments" / "collusion" / "run.py").exists():
        raise FileNotFoundError(
            f"Colosseum not found at {root}. Set COLOSSEUM_ROOT or see README.md."
        )
    return root


def _colosseum_python() -> Path:
    root = _colosseum_root()
    for candidate in (root / ".venv" / "Scripts" / "python.exe", root / ".venv" / "bin" / "python"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No .venv python found under {root}")


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}  (cwd={cwd})")
    return subprocess.run(cmd, cwd=cwd, check=False, capture_output=True, text=True)


def _latest_new_root(output_dir: Path, before: set[str]) -> Path:
    """The timestamped root run.py just created under output_dir."""
    after = {p.name for p in output_dir.iterdir()} if output_dir.exists() else set()
    new = sorted(after - before)
    if not new:
        raise RuntimeError(f"no new output root appeared under {output_dir}")
    return output_dir / new[-1]


def _resolve_output_dir(config_path: Path, colosseum_root: Path) -> Path:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    out = raw["experiment"]["output_dir"]
    return colosseum_root / out


def smoke(config_path: Path, *, seed: int = 1) -> Path:
    """Run exactly one seed, across every configured cell, and report real cost.

    This is the only call allowed to run with no prior cost measurement to
    project from -- its own cost (one seed's worth) is the starting estimate
    for every later escalation.
    """
    colosseum_root = _colosseum_root()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_seeds = list(raw["experiment"]["seeds"])
    raw["experiment"]["seeds"] = [seed]
    output_dir = _resolve_output_dir(config_path, colosseum_root)
    before = {p.name for p in output_dir.iterdir()} if output_dir.exists() else set()

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir=THIS_DIR
    ) as tmp:
        yaml.safe_dump(raw, tmp)
        tmp_path = Path(tmp.name)

    try:
        result = _run(
            [
                str(_colosseum_python()),
                "-m",
                "experiments.collusion.run",
                "--config",
                str(tmp_path.relative_to(colosseum_root))
                if tmp_path.is_relative_to(colosseum_root)
                else str(tmp_path),
            ],
            cwd=colosseum_root,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"smoke run failed with exit code {result.returncode}")

    root = _latest_new_root(output_dir, before)
    totals = scan_sweep_cost(root)
    n_cells = _n_cells(raw)
    print(
        f"\nsmoke run complete at {root}\n"
        f"  1 seed x {n_cells} cells = {n_cells} episodes\n"
        f"  tokens: {totals.prompt_tokens:,} in / {totals.completion_tokens:,} out "
        f"({totals.calls} calls)\n"
        f"  cost: ${totals.cost_usd:.4f} (${FIREWORKS_PRICE_PER_MILLION_TOKENS}/M "
        "tokens, Fireworks 4B-16B tier)\n"
        f"  projected for all {len(target_seeds)} target seeds: "
        f"${totals.cost_usd * len(target_seeds):.4f}\n"
        f"  runs missing usage data: {totals.runs_missing_usage}/{totals.runs_scanned}"
    )
    return root


def _n_cells(raw_config: dict) -> int:
    """Count the unique (topology, secret_channel, prompt_variant) cells.

    Mirrors run.py's own de-dup: prompt_variants collapse to "control" when
    secret_channel_enabled is false.
    """
    sweeps = raw_config["experiment"]["sweeps"]
    total = 0
    for sweep in sweeps:
        settings = set()
        for topology in sweep["topologies"]:
            for secret in sweep["secret_channel_enabled"]:
                variants = sweep["prompt_variants"] if secret else ["control"]
                for variant in variants:
                    settings.add((topology, secret, variant))
        total += len(settings) * len(sweep.get("num_agents", [6])) * len(
            sweep.get("colluder_counts", [2])
        )
    return total


def escalate(
    root: Path, config_path: Path, *, cap_usd: float, dry_run: bool = False
) -> None:
    """Resume `root` to cover every seed in `config_path`, if the cost projects safe."""
    colosseum_root = _colosseum_root()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    target_seeds = list(raw["experiment"]["seeds"])
    n_cells = _n_cells(raw)
    target_episodes = n_cells * len(target_seeds)

    totals = scan_sweep_cost(root)
    completed = totals.runs_scanned - totals.runs_missing_usage
    if completed == 0:
        raise RuntimeError(
            f"no priced episodes found under {root}; run `smoke` first so there "
            "is a real cost-per-episode to project from"
        )
    per_episode = totals.cost_usd / completed
    projected_total = per_episode * target_episodes

    print(
        f"measured so far: {completed} episodes, ${totals.cost_usd:.4f} "
        f"(${per_episode:.5f}/episode)\n"
        f"target: {target_episodes} episodes -> projected ${projected_total:.4f} "
        f"(cap ${cap_usd:.2f})"
    )
    if projected_total > cap_usd:
        raise BudgetExceeded(
            f"projected ${projected_total:.4f} exceeds the ${cap_usd:.2f} cap; "
            "reduce seeds or raise the cap before continuing"
        )
    if dry_run:
        print("[dry run] not launching resume.py")
        return

    result = _run(
        [
            str(_colosseum_python()),
            "-m",
            "experiments.collusion.resume",
            "--root",
            str(root.relative_to(colosseum_root))
            if root.is_relative_to(colosseum_root)
            else str(root),
            "--config",
            str(config_path.resolve()),
        ],
        cwd=colosseum_root,
    )
    print(result.stdout[-6000:])
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"resume failed with exit code {result.returncode}")

    final = scan_sweep_cost(root)
    print(
        f"\nafter resume: ${final.cost_usd:.4f} total, "
        f"{final.runs_scanned - final.runs_missing_usage} priced episodes"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_smoke = sub.add_parser("smoke")
    p_smoke.add_argument("--config", required=True, type=Path)
    p_smoke.add_argument("--seed", type=int, default=1)

    p_esc = sub.add_parser("escalate")
    p_esc.add_argument("--root", required=True, type=Path)
    p_esc.add_argument("--config", required=True, type=Path)
    p_esc.add_argument("--cap-usd", required=True, type=float)
    p_esc.add_argument("--dry-run", action="store_true")

    args = ap.parse_args(argv)
    if args.cmd == "smoke":
        smoke(args.config, seed=args.seed)
    elif args.cmd == "escalate":
        escalate(args.root, args.config, cap_usd=args.cap_usd, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
