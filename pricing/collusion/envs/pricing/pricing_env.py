"""The repeated logit-Bertrand pricing environment (Plan B, Section 7).

Sellers move simultaneously each period. An agent's memory is exactly three
things, all of which the prompt renderer puts in front of it: the rolling
history table, its own previous plan and insight, and (with a channel) the
messages it received. Nothing else carries across periods, which is what makes
the history-window manipulation a real manipulation.

The market itself is deterministic given the prices, so an episode's seed only
drives agent-side randomness (scripted noise, or model sampling) unless
`demand_noise_sd` is set above zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collusion.agents.parsing import ParsedAction
from collusion.envs.pricing.conditions import Condition
from collusion.envs.pricing.economics import (
    Benchmarks,
    PricingParams,
    compute_benchmarks,
    consumer_surplus,
    profits,
    shares,
)
from collusion.envs.pricing.state import Observation, RoundRecord, seller_label


@dataclass
class EnvConfig:
    """Everything about an episode that is not a factor level."""

    total_periods: int = 200
    # Multiplicative lognormal shock on each seller's share. Zero keeps the
    # environment faithful to the reference model (Calvano et al. 2020).
    demand_noise_sd: float = 0.0
    # Price used for a seller whose reply could not be parsed after all retries.
    # Falls back to the seller's own previous price, or the Nash price in period 1.
    fallback_to_previous_price: bool = True
    # Optional discrete price menu (Gate 2 diagnostic). None = any price.
    price_grid: tuple[float, ...] | None = None


@dataclass
class PricingEnvironment:
    """One episode of the repeated pricing game."""

    condition: Condition
    params: PricingParams = field(default_factory=PricingParams)
    config: EnvConfig = field(default_factory=EnvConfig)
    seed: int = 0

    def __post_init__(self) -> None:
        self.benchmarks: Benchmarks = compute_benchmarks(self.params)
        self._rng = np.random.default_rng(self.seed)
        self.period: int = 0
        self.history: list[RoundRecord] = []
        self._plans: list[str] = [""] * self.params.n_sellers
        self._insights: list[str] = [""] * self.params.n_sellers
        # Messages sent in the previous period, waiting to be delivered.
        self._pending_inbox: dict[int, list[tuple[str, str]]] = {
            i: [] for i in range(self.params.n_sellers)
        }

    # --- observation ---------------------------------------------------------

    @property
    def n_sellers(self) -> int:
        return self.params.n_sellers

    @property
    def done(self) -> bool:
        return self.period >= self.config.total_periods

    def observe(self, seller_index: int) -> Observation:
        """What seller `seller_index` sees before pricing the next period."""
        window = self.condition.window_periods
        visible = tuple(self.history[-window:]) if window > 0 else ()
        inbox = (
            tuple(self._pending_inbox[seller_index])
            if self.condition.has_channel
            else ()
        )
        return Observation(
            seller_index=seller_index,
            period=self.period + 1,
            total_periods=self.config.total_periods,
            n_sellers=self.n_sellers,
            cost=self.params.c,
            history=visible,
            window_periods=window,
            prev_plan=self._plans[seller_index],
            prev_insight=self._insights[seller_index],
            inbox=inbox,
            price_grid=self.config.price_grid,
        )

    def observations(self) -> list[Observation]:
        return [self.observe(i) for i in range(self.n_sellers)]

    # --- transition ----------------------------------------------------------

    def resolve_price(self, seller_index: int, action: ParsedAction) -> float:
        """The price actually used for a seller, handling unparseable replies."""
        if action.ok and action.price is not None:
            return float(action.price)
        if self.config.fallback_to_previous_price and self.history:
            return float(self.history[-1].prices[seller_index])
        return float(self.benchmarks.price_nash)

    def step(self, actions: list[ParsedAction]) -> RoundRecord:
        """Advance one period given every seller's action."""
        if self.done:
            raise RuntimeError("episode is already finished")
        if len(actions) != self.n_sellers:
            raise ValueError(
                f"expected {self.n_sellers} actions, got {len(actions)}"
            )

        prices = np.array(
            [self.resolve_price(i, a) for i, a in enumerate(actions)], dtype=float
        )
        quantities = shares(prices, self.params)
        if self.config.demand_noise_sd > 0:
            shock = self._rng.lognormal(
                mean=0.0, sigma=self.config.demand_noise_sd, size=self.n_sellers
            )
            quantities = quantities * shock
        period_profits = (prices - self.params.c) * quantities

        self.period += 1
        record = RoundRecord(
            period=self.period,
            prices=tuple(float(x) for x in prices),
            quantities=tuple(float(x) for x in quantities),
            profits=tuple(float(x) for x in period_profits),
            messages={
                i: a.message
                for i, a in enumerate(actions)
                if self.condition.has_channel and a.message
            },
        )
        self.history.append(record)

        # Carry plan/insight forward; deliver messages to everyone but the sender.
        for i, action in enumerate(actions):
            if action.plan:
                self._plans[i] = action.plan
            if action.insight:
                self._insights[i] = action.insight
        self._pending_inbox = {i: [] for i in range(self.n_sellers)}
        if self.condition.has_channel:
            for sender, text in record.messages.items():
                for recipient in range(self.n_sellers):
                    if recipient != sender:
                        self._pending_inbox[recipient].append(
                            (seller_label(sender), text)
                        )
        return record

    # --- episode-level readouts ---------------------------------------------

    def price_matrix(self) -> np.ndarray:
        """Prices as an array of shape (periods, n_sellers)."""
        if not self.history:
            return np.empty((0, self.n_sellers))
        return np.array([r.prices for r in self.history], dtype=float)

    def profit_matrix(self) -> np.ndarray:
        if not self.history:
            return np.empty((0, self.n_sellers))
        return np.array([r.profits for r in self.history], dtype=float)

    def consumer_surplus_series(self) -> np.ndarray:
        return np.array(
            [consumer_surplus(np.array(r.prices), self.params) for r in self.history],
            dtype=float,
        )

    def nash_profit_check(self) -> float:
        """Per-seller profit at the Nash benchmark, recomputed for assertions."""
        return float(
            profits(np.full(self.n_sellers, self.benchmarks.price_nash), self.params)[0]
        )
