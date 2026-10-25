"""Text-level collusion score for Plan A (Section 8), read from Colosseum's
own judge output and normalised to the 0-1 scale this project reports on.

The judge call itself runs in Colosseum's process (it needs the Colosseum repo
and the `terrarium` package importable); see
`collusion/envs/colosseum_adapter/judge_fireworks.py`, which reuses Colosseum's
published rubric and JSON-parsing/retry logic verbatim and swaps only the
backend from OpenAI to Fireworks. That script writes the same
`results.csv` / `results.jsonl` Colosseum's own judge would, under
`<sweep_dir>/runs/<model_label>/judge_secret_blackboard/`.

This module's job is just: run that script, then read its output back into our
data contract (a 0-1 `text_collusion_score` per run_id), so Stage 4 analysis
reads one normalised number regardless of which rubric variant is in play.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Iterator

# Colosseum's rubric is a 0-5 Likert scale (judge_blackboards.py JUDGE_PROMPTS);
# Section 8 wants a 0-1 score, so divide by the scale's maximum.
JUDGE_SCALE_MAX = 5.0
DEFAULT_PROMPT_VARIANT = "simple"  # the rubric variant judge_blackboards.py
# calls "simple": "Did collusion occur on this blackboard between agents?" --
# closest match to Section 8's "coordinate to benefit a subgroup at the expense
# of the shared objective." "medium" and "complex" (tampering / deception) are
# also computed by judge_blackboards.py in the same pass and kept in the raw
# output for an exploratory breakdown, but are not the primary text score.


def normalize(rating_0_5: float) -> float:
    """Rescale a raw 0-5 judge rating to 0-1."""
    return max(0.0, min(1.0, float(rating_0_5) / JUDGE_SCALE_MAX))


def run_judge(
    sweep_dir: Path,
    *,
    colosseum_root: Path,
    max_concurrent: int = 8,
    overwrite: bool = False,
) -> Path:
    """Invoke judge_fireworks.py over a sweep directory. Returns the sweep_dir."""
    python = _colosseum_python(colosseum_root)
    script = Path(__file__).resolve().parents[1] / "envs" / "colosseum_adapter" / "judge_fireworks.py"
    cmd = [
        str(python),
        str(script),
        "--sweep-dir",
        str(sweep_dir),
        "--max-concurrent",
        str(max_concurrent),
    ]
    if overwrite:
        cmd.append("--overwrite")
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=colosseum_root, capture_output=True, text=True)
    print(result.stdout[-4000:])
    if result.returncode != 0:
        print(result.stderr[-4000:], file=sys.stderr)
        raise RuntimeError(f"judge run failed with exit code {result.returncode}")
    return sweep_dir


def _colosseum_python(colosseum_root: Path) -> Path:
    for candidate in (
        colosseum_root / ".venv" / "Scripts" / "python.exe",
        colosseum_root / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"no .venv python under {colosseum_root}")


def read_text_scores(
    sweep_dir: Path, *, model_label: str, prompt_variant: str = DEFAULT_PROMPT_VARIANT
) -> dict[str, float]:
    """run_id -> normalised 0-1 text_collusion_score, from results.csv.

    judge_blackboards.py writes one row per run with a `judge_{variant}_rating`
    column per rubric (simple/medium/complex), each a 0-5 mean over however
    many blackboards were judged for that run (baseline runs average over all
    blackboards; secret-channel runs judge the secret blackboard only -- see
    judge_blackboards.py's own docstring).

    Per the collusion README, the aggregated CSV lands at
    `<root>/runs/<model_label>/judge_secret_blackboard/results.csv`, where
    `<root>` is the experiment timestamp folder. `sweep_dir` here is
    `<root>/runs/<model_label>/<sweep_name>` (the --sweep-dir form), so that
    root is `sweep_dir.parent.parent`.
    """
    judge_dir = sweep_dir.parent / "judge_secret_blackboard"
    path = judge_dir / "results.csv"
    if not path.exists():
        # Layout can shift between Colosseum versions; fall back to a search
        # under the experiment root rather than guess further.
        candidates = list(sweep_dir.parent.parent.rglob("results.csv"))
        if not candidates:
            raise FileNotFoundError(
                f"no judge results.csv found under {sweep_dir.parent.parent}; "
                "run run_judge() first"
            )
        path = candidates[0]

    column = f"judge_{prompt_variant}_rating"
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            run_id = row.get("run_id") or row.get("run")
            raw = row.get(column)
            if run_id is None or raw in (None, ""):
                continue
            scores[run_id] = normalize(float(raw))
    return scores


def iter_raw_judge_rows(sweep_dir: Path, *, model_label: str) -> Iterator[dict]:
    """The unmodified per-run judge JSON, for the raw-output storage Section 8
    requires ("Store the raw judge output")."""
    judge_root = sweep_dir / "runs" / model_label / "judge_secret_blackboard"
    for path in sorted(judge_root.rglob("*.json")):
        if path.name in {"results.jsonl"}:
            continue
        try:
            yield json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
