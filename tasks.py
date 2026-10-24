"""Task entry points. `make` is not available on Windows, so this file is the
single source of truth and the Makefile / make.cmd just delegate to it.

    python tasks.py test        # the acceptance suite
    python tasks.py pilot       # scripted dry pilot (no model, no cost)
    python tasks.py gate1       # regenerate the Gate 1 artefacts
    python tasks.py results     # Stage 4, gated
    python tasks.py clean       # drop generated logs/episodes/results
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
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        print("targets: " + ", ".join(TARGETS))
        return 0
    target = argv[0]
    if target not in TARGETS:
        print(f"unknown target {target!r}; try: {', '.join(TARGETS)}", file=sys.stderr)
        return 2
    return TARGETS[target]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
