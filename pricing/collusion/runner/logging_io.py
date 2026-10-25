"""The data contract from Section 5, and the readers/writers for it.

Two artefacts per sweep:

* `data/logs/<run_id>.jsonl` -- one record per agent turn, including the exact
  `prompt_messages` list that was sent. The probe layer and the round-trip test
  both depend on this being the real thing rather than a reconstruction.
* `data/episodes/<run_id>/<episode_id>.json` -- one summary per episode.

Both are append-/write-once and keyed by (condition_id, seed) so a sweep can be
resumed without recomputing finished work.
"""

from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 2

# Every field Section 5 requires of a turn record. Asserted by the schema test.
REQUIRED_TURN_FIELDS = (
    "run_id",
    "condition_id",
    "factors",
    "seed",
    "episode_id",
    "turn",
    "agent_id",
    "model",
    "model_settings",
    "prompt_messages",
    "response_text",
    "parsed_action",
    "channel",
    "message_text",
    "reward",
    "regret",
    "metric_components",
    "tokens_in",
    "tokens_out",
    "cost_usd",
    "timestamp",
)

REQUIRED_EPISODE_FIELDS = (
    "episode_id",
    "condition_id",
    "factors",
    "seed",
    "action_collusion_score",
    "action_collusion_label",
    "text_collusion_score",
    "joint_reward",
    "completed",
    "parse_failures",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TurnRecord:
    """One agent turn. Field names and order follow Section 5 exactly."""

    run_id: str
    condition_id: str
    factors: dict[str, str]
    seed: int
    episode_id: str
    turn: int
    agent_id: str
    model: str
    model_settings: dict[str, Any]
    prompt_messages: list[dict[str, str]]
    response_text: str
    parsed_action: dict[str, Any]
    channel: str
    message_text: str
    reward: float
    regret: float
    metric_components: dict[str, Any]
    tokens_in: int
    tokens_out: int
    cost_usd: float
    timestamp: str = field(default_factory=utc_now)
    schema_version: int = SCHEMA_VERSION
    # Index of this agent within the episode, needed to re-derive its view.
    agent_index: int = 0
    period: int = 0
    # How many parse retries this turn consumed (0 = parsed first time).
    retries: int = 0
    # The fully rendered prompt string after the chat template (LLM agents only),
    # so activation extraction can reproduce exactly what the model saw.
    rendered_prompt: str = ""
    # Serving details: request seed, server/local prompt token counts, latency,
    # finish reason, sampling parameters actually sent.
    call_info: dict[str, Any] = field(default_factory=dict)
    # Track A addendum Section 4: labels and inputs for later probe work.
    rolling_delta: float | None = None  # index over last 10 rounds incl. this one
    next_rolling_delta: float | None = None  # index over the next 10 rounds
    in_metric_window: bool = False
    # The numeric rows of the market table shown in this turn's prompt.
    price_window: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class EpisodeSummary:
    """One finished episode. Field names follow Section 5."""

    episode_id: str
    condition_id: str
    factors: dict[str, str]
    seed: int
    action_collusion_score: float
    action_collusion_label: int
    text_collusion_score: float | None
    joint_reward: float
    completed: bool
    parse_failures: int
    # Extra provenance, outside the minimum contract.
    run_id: str = ""
    price_index: float = 0.0
    category: str = ""
    is_pathological: bool = False
    action_metrics: dict[str, Any] = field(default_factory=dict)
    benchmarks: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    agent_models: list[str] = field(default_factory=list)
    label_threshold: float | None = None
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    n_periods: int = 0
    wall_seconds: float = 0.0
    # Phase 1 additions.
    batch: int | None = None
    env_config: dict[str, Any] = field(default_factory=dict)
    metric_window: int = 0
    reasoning_blocks: int = 0
    truncated_turns: int = 0
    message_leaks: int = 0
    # Crude keyword baseline (Phase 1 override 7); not the LLM judge.
    text_baseline: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def as_dict(self) -> dict:
        return asdict(self)


def episode_id_for(condition_id: str, seed: int) -> str:
    """Episodes are keyed by (condition_id, seed); this is that key as a string."""
    return f"{condition_id}#seed={seed}"


_SAFE = re.compile(r"[^A-Za-z0-9._=#-]")


def episode_filename(episode_id: str) -> str:
    """Filesystem-safe filename for an episode id ('#' and '=' are fine on NTFS)."""
    return _SAFE.sub("_", episode_id) + ".json"


class TurnLogWriter:
    """Append-only JSONL writer for turn records.

    Opened in append mode and flushed per write so that a crashed sweep still
    leaves a readable log of everything that completed. Thread-safe: concurrent
    episodes write whole episodes at a time through `write_many`, so one
    episode's turns are never interleaved with another's.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None
        self._lock = threading.Lock()

    def __enter__(self) -> "TurnLogWriter":
        self._fh = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def write(self, record: TurnRecord) -> None:
        self.write_many([record])

    def write_many(self, records) -> None:
        if self._fh is None:
            raise RuntimeError("TurnLogWriter must be used as a context manager")
        lines = []
        for record in records:
            payload = record.as_dict()
            missing = [f for f in REQUIRED_TURN_FIELDS if f not in payload]
            if missing:
                raise ValueError(f"turn record is missing required fields: {missing}")
            lines.append(json.dumps(payload, ensure_ascii=False) + "\n")
        with self._lock:
            self._fh.write("".join(lines))
            self._fh.flush()


def read_turn_log(path: str | os.PathLike) -> Iterator[dict]:
    """Stream turn records from a JSONL log."""
    p = Path(path)
    if not p.exists():
        return
    with p.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{line_no}: malformed JSONL: {exc}") from exc


def write_episode_summary(
    summary: EpisodeSummary, episodes_dir: str | os.PathLike
) -> Path:
    """Write one episode summary; returns the path written."""
    payload = summary.as_dict()
    missing = [f for f in REQUIRED_EPISODE_FIELDS if f not in payload]
    if missing:
        raise ValueError(f"episode summary is missing required fields: {missing}")
    out_dir = Path(episodes_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / episode_filename(summary.episode_id)
    # Write-then-rename so a crash cannot leave a half-written summary that the
    # resume logic would mistake for a finished episode.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return path


def read_episode_summaries(episodes_dir: str | os.PathLike) -> list[dict]:
    """All episode summaries in a directory, sorted by episode_id."""
    out_dir = Path(episodes_dir)
    if not out_dir.exists():
        return []
    rows = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted(out_dir.glob("*.json"))
    ]
    return sorted(rows, key=lambda r: r.get("episode_id", ""))


def completed_episode_keys(episodes_dir: str | os.PathLike) -> set[tuple[str, int]]:
    """(condition_id, seed) pairs already finished, for resumable sweeps."""
    keys = set()
    for row in read_episode_summaries(episodes_dir):
        if row.get("completed"):
            keys.add((row["condition_id"], int(row["seed"])))
    return keys
