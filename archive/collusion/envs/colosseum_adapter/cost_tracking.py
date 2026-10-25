"""Post-hoc $ cost from Colosseum's own logs.

Colosseum has no built-in notion of API dollar cost (its `metrics.total_cost`
field is the Jira task's effort/cost, an environment economics quantity --
unrelated to what we spent on inference). It does, however, save the raw
provider `usage` dict for every LLM call in `<run_dir>/agent_turns.json`
(`run.py` writes `{"usage": response.get("usage"), ...}` per turn). This module
sums those and prices them, so the budget guard (Section 3 rule 3) has
something real to check against before and during a sweep.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Fireworks serverless pricing, 4B-16B parameter tier: flat rate for input and
# output tokens (verified against docs.fireworks.ai/serverless/pricing,
# 2026-10-24). Qwen3-8B falls in this tier.
FIREWORKS_PRICE_PER_MILLION_TOKENS = 0.20


@dataclass
class UsageTotals:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0
    runs_scanned: int = 0
    runs_missing_usage: int = 0

    @property
    def cost_usd(self) -> float:
        return (self.total_tokens / 1_000_000) * FIREWORKS_PRICE_PER_MILLION_TOKENS

    def __iadd__(self, other: "UsageTotals") -> "UsageTotals":
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.calls += other.calls
        self.runs_scanned += other.runs_scanned
        self.runs_missing_usage += other.runs_missing_usage
        return self


def usage_from_agent_turns(path: Path) -> UsageTotals:
    """Sum token usage from one run's agent_turns.json."""
    totals = UsageTotals(runs_scanned=1)
    try:
        turns = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        totals.runs_missing_usage = 1
        return totals
    saw_usage = False
    for turn in turns:
        usage = (turn or {}).get("usage")
        if not usage:
            continue
        saw_usage = True
        totals.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        totals.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        if "total_tokens" in usage:
            totals.total_tokens += int(usage["total_tokens"] or 0)
        else:
            totals.total_tokens += int(usage.get("prompt_tokens", 0) or 0) + int(
                usage.get("completion_tokens", 0) or 0
            )
        totals.calls += 1
    if not saw_usage:
        totals.runs_missing_usage = 1
    return totals


def scan_sweep_cost(output_dir: Path) -> UsageTotals:
    """Sum usage across every run under <output_dir>/runs/**/agent_turns.json."""
    totals = UsageTotals()
    runs_root = Path(output_dir) / "runs"
    for path in sorted(runs_root.glob("*/*/*/agent_turns.json")):
        totals += usage_from_agent_turns(path)
    return totals


def per_episode_cost(output_dir: Path) -> float:
    """Mean $ cost per completed run, for projecting the cost of remaining seeds."""
    totals = scan_sweep_cost(output_dir)
    completed = totals.runs_scanned - totals.runs_missing_usage
    if completed <= 0:
        return 0.0
    return totals.cost_usd / completed
