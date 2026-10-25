"""Task entry points. `make` is not available on Windows, so this file is the
single source of truth and the Makefile / make.cmd just delegate to it.

    python tasks.py test        # the acceptance suite
    python tasks.py pilot       # scripted dry pilot (no model, no cost)
    python tasks.py gate1       # regenerate the Gate 1 artefacts
    python tasks.py results     # Stage 4, gated
    python tasks.py clean       # drop generated logs/episodes/results

    # Plan A (Colosseum + Fireworks-hosted Qwen3-8B), Stage 2+:
    python tasks.py plan_a_smoke                  # 1 seed, all cells, real $ measured
    python tasks.py plan_a_pilot [--cap-usd N]     # escalate to the pilot's 5 seeds
    python tasks.py plan_a_main [--cap-usd N]      # escalate to the main sweep's 20 seeds
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def _run(*args: str) -> int:
    print(f"$ {' '.join(args)}")
    return subprocess.call(args, cwd=ROOT)


def test() -> int:
    """Run the acceptance suite (Section 6.4)."""
    return _run(PY, "-m", "pytest", "-q")


def pilot() -> int:
    """Scripted dry pilot: exercises the whole pipeline at zero cost.

    This is *not* the Stage 2 pilot, which needs AGENT_MODEL and human approval
    at Gate 1. It runs the planted-collusion and competitive controls over the
    pilot design so that resumability, logging and the budget guard are
    exercised end to end before any model is loaded.
    """
    rc = 0
    for policy in ("colluding", "competitive"):
        rc |= _run(
            PY,
            "-m",
            "collusion.runner.run_sweep",
            "--pilot",
            "--policy",
            policy,
            "--run-id",
            f"dry_pilot_{policy}",
        )
    return rc


def gate1() -> int:
    """Regenerate results/gate1_report.md."""
    return _run(PY, "-m", "collusion.runner.gate1_report")


COLOSSEUM_ADAPTER = ROOT / "collusion" / "envs" / "colosseum_adapter"
PILOT_CONFIG = COLOSSEUM_ADAPTER / "configs" / "pilot_fireworks.yaml"
MAIN_CONFIG = COLOSSEUM_ADAPTER / "configs" / "main_sweep_fireworks.yaml"
_SMOKE_ROOT_FILE = COLOSSEUM_ADAPTER / ".smoke_root"  # records the root smoke() created


def plan_a_smoke(argv: list[str] | None = None) -> int:
    """Run 1 seed across every pilot cell and print the real measured cost."""
    from collusion.envs.colosseum_adapter.run_plan_a import smoke

    root = smoke(PILOT_CONFIG)
    _SMOKE_ROOT_FILE.write_text(str(root), encoding="utf-8")
    print(f"\n[root recorded at {_SMOKE_ROOT_FILE} for plan_a_pilot/plan_a_main]")
    return 0


def _escalate(config_path: Path, argv: list[str] | None) -> int:
    from collusion.envs.colosseum_adapter.run_plan_a import escalate

    if not _SMOKE_ROOT_FILE.exists():
        print(
            "No smoke run recorded. Run `python tasks.py plan_a_smoke` first so "
            "there is a real cost-per-episode to project from.",
            file=sys.stderr,
        )
        return 1
    root = Path(_SMOKE_ROOT_FILE.read_text(encoding="utf-8").strip())

    cap_usd = 10.0
    args = list(argv or [])
    if "--cap-usd" in args:
        cap_usd = float(args[args.index("--cap-usd") + 1])
    dry_run = "--dry-run" in args

    escalate(root, config_path, cap_usd=cap_usd, dry_run=dry_run)
    return 0


def plan_a_pilot(argv: list[str] | None = None) -> int:
    """Escalate the smoke run to the pilot's full 5-seeds x 6-cells design."""
    return _escalate(PILOT_CONFIG, argv)


def plan_a_main(argv: list[str] | None = None) -> int:
    """Escalate to the main sweep's 20 seeds. Gated: needs Gate 2/3 sign-off."""
    print(
        "plan_a_main needs Gate 2 (pilot review) and Gate 3 (prereg committed) "
        "sign-off first; see README.md. Pass --i-have-approval to override.",
        file=sys.stderr,
    )
    args = list(argv or [])
    if "--i-have-approval" not in args:
        return 1
    return _escalate(MAIN_CONFIG, [a for a in args if a != "--i-have-approval"])


def results() -> int:
    """Stage 4 analysis. Gated: there is no main-sweep data yet."""
    print(
        "`results` is Stage 4 (Section 6). It needs an approved main sweep, which "
        "requires Gate 2 and Gate 3 sign-off first. Nothing to do.",
        file=sys.stderr,
    )
    return 1


def clean() -> int:
    """Remove generated data. Prompts first, since logs are expensive to rebuild."""
    targets = [ROOT / "data" / "logs", ROOT / "data" / "episodes", ROOT / "results"]
    existing = [
        p
        for p in targets
        if p.exists() and any(c.name != ".gitkeep" for c in p.iterdir())
    ]
    if not existing:
        print("nothing to clean")
        return 0
    print("This will delete generated data in:")
    for p in existing:
        print(f"  {p.relative_to(ROOT)}")
    if input("Type 'yes' to confirm: ").strip().lower() != "yes":
        print("aborted")
        return 1
    for p in existing:
        for child in p.iterdir():
            if child.name == ".gitkeep":
                continue
            shutil.rmtree(child) if child.is_dir() else child.unlink()
    print("cleaned")
    return 0


TARGETS = {
    "test": test,
    "pilot": pilot,
    "gate1": gate1,
    "results": results,
    "clean": clean,
    "plan_a_smoke": plan_a_smoke,
    "plan_a_pilot": plan_a_pilot,
    "plan_a_main": plan_a_main,
}
# Targets whose implementation takes the remaining CLI args (everything after
# the target name), rather than being called with none.
_TAKES_ARGV = {"plan_a_smoke", "plan_a_pilot", "plan_a_main"}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        print("targets: " + ", ".join(TARGETS))
        return 0
    target = argv[0]
    if target not in TARGETS:
        print(f"unknown target {target!r}; try: {', '.join(TARGETS)}", file=sys.stderr)
        return 2
    if target in _TAKES_ARGV:
        return TARGETS[target](argv[1:])
    return TARGETS[target]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
