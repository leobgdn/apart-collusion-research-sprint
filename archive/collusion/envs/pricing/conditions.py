"""Factors, levels, and condition identity for the pricing sweep.

Section 3 rule 7 ("one manipulation per condition") is enforced structurally:
a Condition is nothing but a choice of level for each factor, and every prompt
slot is *owned* by exactly one factor (see OWNED_SLOTS). The diff test in
tests/test_prompt_diff.py uses that ownership map to prove that changing one
factor changes only the slots that factor owns.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, asdict

# Factor name -> ordered levels. The first level of each factor is the baseline.
FACTORS: dict[str, tuple[str, ...]] = {
    "channel": ("none", "private"),
    "history_window": ("long", "short"),
    "horizon": ("long_run", "myopic"),
}

# Which rendered prompt slots each factor is allowed to change.
OWNED_SLOTS: dict[str, frozenset[str]] = {
    "channel": frozenset({"channel_instruction", "message_field", "inbox_section"}),
    "history_window": frozenset({"history_table", "window_phrase"}),
    "horizon": frozenset({"horizon_instruction"}),
}

# Numeric meaning of the history_window levels, in periods.
HISTORY_WINDOW_PERIODS: dict[str, int] = {"long": 20, "short": 3}

# The hypothesis each factor is the primary test of (used in reporting).
PRIMARY_HYPOTHESIS: dict[str, str] = {
    "channel": "H1",
    "history_window": "H2",
    "horizon": "H3",
}

BASELINE_LEVELS: dict[str, str] = {f: levels[0] for f, levels in FACTORS.items()}


@dataclass(frozen=True)
class Condition:
    """One cell of the factorial design."""

    channel: str
    history_window: str
    horizon: str

    def __post_init__(self) -> None:
        for factor, levels in FACTORS.items():
            value = getattr(self, factor)
            if value not in levels:
                raise ValueError(
                    f"factor {factor!r} has no level {value!r}; expected one of {levels}"
                )

    @property
    def factors(self) -> dict[str, str]:
        return asdict(self)

    @property
    def condition_id(self) -> str:
        """Stable, filesystem-safe identifier. Keys are sorted for determinism.

        Factors are joined with "." because both factor names (history_window)
        and levels (long_run) already contain underscores.
        """
        return ".".join(f"{f}={getattr(self, f)}" for f in sorted(FACTORS))

    @property
    def window_periods(self) -> int:
        return HISTORY_WINDOW_PERIODS[self.history_window]

    @property
    def has_channel(self) -> bool:
        return self.channel == "private"

    def replace(self, **changes: str) -> "Condition":
        return Condition(**{**self.factors, **changes})

    def differing_factors(self, other: "Condition") -> set[str]:
        return {f for f in FACTORS if getattr(self, f) != getattr(other, f)}


BASELINE = Condition(**BASELINE_LEVELS)


def full_factorial() -> list[Condition]:
    """All cells, in a deterministic order (sorted factor names, declared levels)."""
    names = sorted(FACTORS)
    return [
        Condition(**dict(zip(names, combo)))
        for combo in itertools.product(*(FACTORS[n] for n in names))
    ]


def one_factor_at_a_time() -> list[Condition]:
    """Baseline plus one condition per non-baseline level (the pilot design)."""
    out = [BASELINE]
    for factor, levels in FACTORS.items():
        for level in levels[1:]:
            out.append(BASELINE.replace(**{factor: level}))
    return out


def condition_from_id(condition_id: str) -> Condition:
    """Inverse of Condition.condition_id."""
    parts = dict(piece.split("=", 1) for piece in condition_id.split("."))
    missing = set(FACTORS) - set(parts)
    if missing:
        raise ValueError(f"condition_id {condition_id!r} is missing factors {missing}")
    return Condition(**parts)
