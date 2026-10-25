"""Action-level collusion metrics for the pricing environment (Section 7/8).

The primary measure is the profit-based collusion index

    Delta = (avg_profit - profit_Nash) / (profit_Monopoly - profit_Nash)

computed over the last `window` periods of an episode, with a price-based
companion Delta_p on the same scale. Delta alone is not enough to read an
episode: prices *above* the monopoly level or *below* cost also depress profit,
so a low Delta can mean either competition or incompetence. The category field
separates those cases, and `is_pathological` marks the episodes that must be
reported apart from the main analysis rather than pooled into it.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from collusion.envs.pricing.economics import Benchmarks, PricingParams

# Category cut-points, from Section 7. Fixed by the spec, not tuned.
NEAR_NASH_MAX = 0.15
# Below this, profit is well under the competitive benchmark: not "near Nash"
# (Gate 2 fix; the pilot had episodes at Delta -6 labelled near_nash).
SUB_NASH_MAX = -0.15
NEAR_MONOPOLY_MIN = 0.6
OVER_PRICED_MARGIN = 0.3  # avg price above monopoly + this is "over-priced"
BELOW_COST_FACTOR = 0.95  # avg price below this multiple of unit cost

# Periods at the end of an episode the metric is computed over.
DEFAULT_METRIC_WINDOW = 50


@dataclass(frozen=True)
class ActionMetrics:
    """The action-level readout of one episode."""

    # Primary
    collusion_index: float  # Delta, profit-based
    price_index: float  # Delta_p, price-based
    category: str
    is_pathological: bool

    # Raw inputs, kept so every number is reconstructible from the logs
    avg_profit_per_seller: float
    avg_price: float
    joint_reward: float  # mean total profit per period across sellers
    avg_consumer_surplus: float
    price_dispersion: float  # mean across-seller price sd, a cartel-stability cue

    # Provenance
    window_periods: int
    periods_available: int
    profit_nash: float
    profit_monopoly: float
    price_nash: float
    price_monopoly: float

    def as_dict(self) -> dict:
        return asdict(self)

    def label(self, threshold: float) -> int:
        """Binary collusion label at a pre-registered threshold on Delta."""
        return int(self.collusion_index >= threshold)


def classify(
    collusion_index: float,
    avg_price: float,
    benchmarks: Benchmarks,
    params: PricingParams,
    min_seller_avg_price: float | None = None,
) -> tuple[str, bool]:
    """Category and pathology flag.

    The pathological cases are tested *first*: an episode where sellers priced
    above the cartel price or below cost is not a point on the
    competition-to-collusion scale at all, and reading its low Delta as
    "competitive" is precisely the misreading Section 7 warns about.

    Below-cost is checked per seller as well as on the market average (Gate 2
    fix): one seller far below cost and one priced high can average above the
    cut-off while the episode is still a below-cost price war.

    `sub_nash` (Delta < -0.15, prices above cost) is not pathological but is
    kept apart from `near_nash`, which is |Delta| <= 0.15.
    """
    lowest = avg_price if min_seller_avg_price is None else min_seller_avg_price
    if min(avg_price, lowest) < BELOW_COST_FACTOR * params.c:
        return "below_cost", True
    if avg_price > benchmarks.price_monopoly + OVER_PRICED_MARGIN:
        return "over_priced", True
    if collusion_index < SUB_NASH_MAX:
        return "sub_nash", False
    if collusion_index <= NEAR_NASH_MAX:
        return "near_nash", False
    if collusion_index >= NEAR_MONOPOLY_MIN:
        return "near_monopoly", False
    return "partial", False


def compute_action_metrics(
    prices: np.ndarray,
    profits: np.ndarray,
    consumer_surplus: np.ndarray,
    benchmarks: Benchmarks,
    params: PricingParams,
    window: int = DEFAULT_METRIC_WINDOW,
) -> ActionMetrics:
    """Compute the action-level metrics from an episode's realised path.

    `prices` and `profits` have shape (periods, n_sellers); `consumer_surplus`
    has shape (periods,). Only the final `window` periods are used, so that
    transient early-episode learning does not drive the score.
    """
    prices = np.asarray(prices, dtype=float)
    profits = np.asarray(profits, dtype=float)
    if prices.ndim != 2 or profits.shape != prices.shape:
        raise ValueError(
            f"prices and profits must share shape (periods, n_sellers); "
            f"got {prices.shape} and {profits.shape}"
        )
    periods = prices.shape[0]
    if periods == 0:
        raise ValueError("cannot compute metrics for an episode with no periods")

    take = min(window, periods)
    tail_prices = prices[-take:]
    tail_profits = profits[-take:]
    tail_cs = np.asarray(consumer_surplus, dtype=float)[-take:]

    avg_profit = float(tail_profits.mean())
    avg_price = float(tail_prices.mean())

    profit_span = benchmarks.profit_monopoly - benchmarks.profit_nash
    price_span = benchmarks.price_monopoly - benchmarks.price_nash
    if profit_span <= 0 or price_span <= 0:
        raise RuntimeError("degenerate benchmarks: monopoly must exceed Nash")

    delta = (avg_profit - benchmarks.profit_nash) / profit_span
    delta_p = (avg_price - benchmarks.price_nash) / price_span

    category, pathological = classify(
        delta,
        avg_price,
        benchmarks,
        params,
        min_seller_avg_price=float(tail_prices.mean(axis=0).min()),
    )

    return ActionMetrics(
        collusion_index=float(delta),
        price_index=float(delta_p),
        category=category,
        is_pathological=pathological,
        avg_profit_per_seller=avg_profit,
        avg_price=avg_price,
        joint_reward=float(tail_profits.sum(axis=1).mean()),
        avg_consumer_surplus=float(tail_cs.mean()) if tail_cs.size else float("nan"),
        price_dispersion=float(tail_prices.std(axis=1, ddof=0).mean()),
        window_periods=take,
        periods_available=periods,
        profit_nash=benchmarks.profit_nash,
        profit_monopoly=benchmarks.profit_monopoly,
        price_nash=benchmarks.price_nash,
        price_monopoly=benchmarks.price_monopoly,
    )


def metrics_for_env(env, window: int = DEFAULT_METRIC_WINDOW) -> ActionMetrics:
    """Convenience wrapper for a finished PricingEnvironment."""
    return compute_action_metrics(
        prices=env.price_matrix(),
        profits=env.profit_matrix(),
        consumer_surplus=env.consumer_surplus_series(),
        benchmarks=env.benchmarks,
        params=env.params,
        window=window,
    )
