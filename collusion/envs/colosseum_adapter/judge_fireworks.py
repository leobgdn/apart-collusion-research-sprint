"""Run Colosseum's own blackboard judge, backed by Fireworks instead of OpenAI.

Must be executed with the Colosseum checkout's own interpreter (it needs
`terrarium` and the `experiments` package importable), e.g.:

    <colosseum_root>/.venv/Scripts/python.exe judge_fireworks.py --root <...> ...

`judge_blackboards.py` calls `client = OpenAIClient()` and then only
`client.init_context(...)` / `client.generate_response(...)` -- exactly the
`AbstractClient` interface `FireworksClient` already implements with the same
signatures. So the whole swap is one attribute patch; every other part of
their script (rubric text, JSON parsing/retry, per-blackboard selection, CSV
aggregation) runs completely unmodified. That matters for Section 8: we are
not re-deriving a new, unvalidated rubric under time pressure, we are reusing
the one the Colosseum paper already published, on a different backend.

Section 8 requirements this satisfies unchanged from upstream:
  - fixed rubric, temperature 0 by default (`--temperature` already defaults
    to 0.0 in judge_blackboards.py)
  - judged purely from the blackboard transcript text, never shown the
    action-level score (the rubric prompt only ever receives `transcript`)

The one build-prompt-specific choice made here is the model string, forced to
Fireworks' Qwen3-8B regardless of --judge-model, so Section 0's JUDGE_MODEL
answer cannot drift from what actually runs.
"""

from __future__ import annotations

import sys
from pathlib import Path

JUDGE_MODEL = "accounts/fireworks/models/qwen3-8b"

# This file is expected to be invoked directly (not imported as a package), so
# find the Colosseum repo root relative to our own checkout's known layout:
# <colosseum_root>/experiments/collusion/judge_blackboards.py is the target.
_THIS_DIR = Path(__file__).resolve().parent


def _find_colosseum_root() -> Path:
    """Locate the Colosseum checkout.

    Checked in order: $COLOSSEUM_ROOT, then the documented sibling location
    (see README.md / DECISIONS.md: Colosseum is not vendored into this repo).
    """
    import os

    env_path = os.environ.get("COLOSSEUM_ROOT")
    if env_path:
        root = Path(env_path).expanduser().resolve()
        if (root / "experiments" / "collusion" / "judge_blackboards.py").exists():
            return root
        raise FileNotFoundError(
            f"COLOSSEUM_ROOT={root} does not contain "
            "experiments/collusion/judge_blackboards.py"
        )
    sibling = _THIS_DIR.parents[3] / "colosseum"
    if (sibling / "experiments" / "collusion" / "judge_blackboards.py").exists():
        return sibling.resolve()
    raise FileNotFoundError(
        "Could not find the Colosseum checkout. Set COLOSSEUM_ROOT, or clone it "
        f"to the documented sibling path: {sibling}"
    )


def main() -> int:
    colosseum_root = _find_colosseum_root()
    sys.path.insert(0, str(colosseum_root))

    from terrarium.llm.clients.fireworks_client import FireworksClient

    import experiments.collusion.judge_blackboards as jb

    # The entire backend swap. `FireworksClient()` takes no required args
    # (reads FIREWORKS_API_KEY from the environment), matching `OpenAIClient()`.
    jb.OpenAIClient = FireworksClient

    # Force the judge model regardless of what was passed on the command line,
    # so JUDGE_MODEL (Section 0) is enforced here rather than trusted to the
    # caller. Any --judge-model the caller passed (as "--judge-model X" or
    # "--judge-model=X") is stripped, value included; everything else
    # (--root/--sweep-dir/--temperature/--max-concurrent/...) passes through
    # to jb.main() untouched.
    argv = sys.argv[1:]
    filtered: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--judge-model":
            skip_next = True
            continue
        if arg.startswith("--judge-model="):
            continue
        filtered.append(arg)
    sys.argv = [sys.argv[0], *filtered, "--judge-model", JUDGE_MODEL]

    jb.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
