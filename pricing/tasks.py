"""Task entry points. `make` is not available on Windows, so this file is the
single source of truth and the Makefile / make.cmd just delegate to it.

    python tasks.py test           # the acceptance suite
    python tasks.py gate1          # regenerate reports/gate1.md (runs the tests)
    python tasks.py dry_pilot      # scripted pilot over the 9 cells (no model)
    python tasks.py check_server   # verify vLLM, thinking off, publish endpoint
    python tasks.py throughput     # dedicated <=10 min window (GPU_BUSY held)
    python tasks.py pilot          # Stage 2 LLM pilot, 90-minute wall-clock cap
    python tasks.py gate2_data     # numbers for reports/gate2.md, from the logs
    python tasks.py clean          # drop generated logs/episodes/results
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


def test(argv=None) -> int:
    """Run the acceptance suite."""
    return _run(PY, "-m", "pytest", "-q", *(argv or []))


def gate1(argv=None) -> int:
    """Regenerate reports/gate1.md."""
    return _run(PY, "-m", "collusion.runner.gate1_report")


def dry_pilot(argv=None) -> int:
    """Scripted pilot over the pilot design: exercises the pipeline, no model."""
    rc = 0
    for policy in ("colluding", "competitive"):
        rc |= _run(
            PY, "-m", "collusion.runner.run_sweep", "--policy", policy,
            "--run-id", f"dry_pilot_{policy}", "--concurrency", "4",
        )
    return rc


def check_server(argv=None) -> int:
    return _run(PY, "-m", "serve.check_server", *(argv or []))


def throughput(argv=None) -> int:
    return _run(PY, "-m", "collusion.runner.throughput", *(argv or []))


def pilot(argv=None) -> int:
    """Stage 2 pilot: 9 cells x 5 seeds x 80 rounds, 32 concurrent, 90-min cap."""
    return _run(
        PY, "-m", "collusion.runner.run_sweep", "--llm", "--run-id", "pilot_v1",
        "--minutes", "90", *(argv or []),
    )


def gate2_data(argv=None) -> int:
    return _run(PY, "-m", "collusion.analysis.pilot_report", *(argv or []))


def results(argv=None) -> int:
    """Stage 4 analysis. Gated: there is no main-sweep data yet."""
    print(
        "`results` is Stage 4 (Section 6). It needs an approved main sweep, which "
        "requires Gate 2 and Gate 3 sign-off first. Nothing to do.",
        file=sys.stderr,
    )
    return 1


def clean(argv=None) -> int:
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
    "gate1": gate1,
    "dry_pilot": dry_pilot,
    "check_server": check_server,
    "throughput": throughput,
    "pilot": pilot,
    "gate2_data": gate2_data,
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
    return TARGETS[target](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
