"""What happened in a period, and what one seller gets to see about it.

Kept in its own module so both the prompt renderer and the environment can
import it without a cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field


SELLER_LABELS = "ABCDEFGH"


def seller_label(index: int) -> str:
    if not 0 <= index < len(SELLER_LABELS):
        raise ValueError(f"seller index {index} out of range")
    return SELLER_LABELS[index]


@dataclass(frozen=True)
class RoundRecord:
    """The realised outcome of one period, from the market's point of view."""

    period: int
    prices: tuple[float, ...]
    quantities: tuple[float, ...]
    profits: tuple[float, ...]
    # Message each seller sent during this period, by seller index. Empty when
    # the condition has no channel.
    messages: dict[int, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Everything seller `seller_index` may condition on when pricing `period`.

    This is the sole input to the prompt renderer, which is what makes the
    "conditions differ only in the manipulated field" test meaningful: hold the
    Observation fixed, vary the Condition, and inspect the rendered slots.
    """

    seller_index: int
    period: int
    total_periods: int
    n_sellers: int
    cost: float
    # Already truncated to the condition's history window, oldest first.
    history: tuple[RoundRecord, ...]
    # How many periods the window covers (for the prompt's wording).
    window_periods: int
    prev_plan: str
    prev_insight: str
    # (sender_label, text) received since the agent last acted; empty with no channel.
    inbox: tuple[tuple[str, str], ...] = ()
    # Allowed prices, if the action space is a discrete menu (else free).
    price_grid: tuple[float, ...] | None = None

    @property
    def label(self) -> str:
        return seller_label(self.seller_index)

    @property
    def rival_indices(self) -> tuple[int, ...]:
        return tuple(i for i in range(self.n_sellers) if i != self.seller_index)

    @property
    def rival_labels(self) -> tuple[str, ...]:
        return tuple(seller_label(i) for i in self.rival_indices)
