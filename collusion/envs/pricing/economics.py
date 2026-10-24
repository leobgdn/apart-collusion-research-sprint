"""Logit-Bertrand economics: demand, and the Nash / joint-monopoly benchmarks.

Model (Section 7 of the build prompt, following Calvano et al. 2020):

    s_i = exp((a - p_i) / mu)              # firm i's exponentiated quality-adjusted value
    s_0 = exp(a0 / mu)                     # outside option
    q_i = s_i / (sum_j s_j + s_0)          # logit market share
    pi_i = (p_i - c) * q_i                 # per-period profit

Defaults: a=2, a0=0, c=1, mu=0.25, n=2.

Both benchmarks are solved numerically (never hardcoded) and stored with every
run, because the collusion index is defined relative to them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.optimize import brentq


@dataclass(frozen=True)
class PricingParams:
    """Structural parameters of the logit-Bertrand market."""

    n_sellers: int = 2
    a: float = 2.0
    a0: float = 0.0
    c: float = 1.0
    mu: float = 0.25

    def as_dict(self) -> dict:
        return asdict(self)


def shares(prices: np.ndarray, params: PricingParams) -> np.ndarray:
    """Logit market shares for each seller. Shares sum to < 1 (outside option)."""
    prices = np.asarray(prices, dtype=float)
    if prices.shape != (params.n_sellers,):
        raise ValueError(
            f"expected {params.n_sellers} prices, got shape {prices.shape}"
        )
    # Subtract the max exponent before exponentiating so large (a - p)/mu cannot
    # overflow; the shift cancels in the ratio.
    exps = np.concatenate([(params.a - prices) / params.mu, [params.a0 / params.mu]])
    shift = exps.max()
    weights = np.exp(exps - shift)
    return weights[:-1] / weights.sum()


def profits(prices: np.ndarray, params: PricingParams) -> np.ndarray:
    """Per-period profit for each seller."""
    prices = np.asarray(prices, dtype=float)
    return (prices - params.c) * shares(prices, params)


def consumer_surplus(prices: np.ndarray, params: PricingParams) -> float:
    """Logit consumer surplus (log-sum-exp), a secondary welfare measure."""
    prices = np.asarray(prices, dtype=float)
    exps = np.concatenate([(params.a - prices) / params.mu, [params.a0 / params.mu]])
    shift = exps.max()
    return float(params.mu * (shift + np.log(np.exp(exps - shift).sum())))


def _symmetric_share(price: float, params: PricingParams) -> float:
    """Share of one seller when all n sellers charge `price`."""
    return float(shares(np.full(params.n_sellers, price), params)[0])


def nash_price(params: PricingParams) -> float:
    """Symmetric one-shot Bertrand-Nash price.

    First-order condition for seller i, with dq_i/dp_i = -q_i (1 - q_i) / mu:

        q_i + (p_i - c) * (-q_i (1 - q_i) / mu) = 0
        =>  p_i = c + mu / (1 - q_i)

    At a symmetric equilibrium q_i = q(p), so we root-find on
    g(p) = p - c - mu / (1 - q(p)), which is strictly increasing in p.
    """

    def g(p: float) -> float:
        return p - params.c - params.mu / (1.0 - _symmetric_share(p, params))

    return _bracket_and_solve(g, params, name="Nash")


def monopoly_price(params: PricingParams) -> float:
    """Symmetric joint-profit-maximising (cartel) price.

    Joint profit with a common price p is (p - c) * Q(p) where
    Q(p) = n s / (n s + s_0). The FOC reduces to

        p = c + mu * (n s + s_0) / s_0 = c + mu / (1 - Q(p))

    i.e. the same form as Nash but with the *industry* share Q in place of the
    individual share q -- which is exactly why the cartel price is higher.
    """

    def g(p: float) -> float:
        total = _symmetric_share(p, params) * params.n_sellers
        return p - params.c - params.mu / (1.0 - total)

    return _bracket_and_solve(g, params, name="monopoly")


def _bracket_and_solve(g, params: PricingParams, name: str) -> float:
    """Root-find g on an expanding bracket above marginal cost."""
    lo = params.c + 1e-9
    if g(lo) > 0:
        raise RuntimeError(f"{name} FOC has no root above marginal cost")
    hi = params.c + params.mu
    for _ in range(200):
        if g(hi) > 0:
            return float(brentq(g, lo, hi, xtol=1e-12, rtol=1e-14))
        lo, hi = hi, params.c + 2.0 * (hi - params.c)
    raise RuntimeError(f"could not bracket the {name} price")


@dataclass(frozen=True)
class Benchmarks:
    """The two reference points the collusion index is normalised against."""

    price_nash: float
    price_monopoly: float
    profit_nash: float
    profit_monopoly: float
    cs_nash: float
    cs_monopoly: float

    def as_dict(self) -> dict:
        return asdict(self)


def compute_benchmarks(params: PricingParams) -> Benchmarks:
    """Solve both benchmarks and the per-seller profit at each."""
    p_n = nash_price(params)
    p_m = monopoly_price(params)
    if not p_m > p_n:
        raise RuntimeError(
            f"monopoly price {p_m} must exceed Nash price {p_n}; check parameters"
        )
    pi_n = float(profits(np.full(params.n_sellers, p_n), params)[0])
    pi_m = float(profits(np.full(params.n_sellers, p_m), params)[0])
    if not pi_m > pi_n:
        raise RuntimeError("monopoly profit must exceed Nash profit")
    return Benchmarks(
        price_nash=p_n,
        price_monopoly=p_m,
        profit_nash=pi_n,
        profit_monopoly=pi_m,
        cs_nash=consumer_surplus(np.full(params.n_sellers, p_n), params),
        cs_monopoly=consumer_surplus(np.full(params.n_sellers, p_m), params),
    )


def best_response_price(
    rival_prices: np.ndarray, seller_index: int, params: PricingParams
) -> float:
    """Seller `seller_index`'s profit-maximising price, holding rivals fixed.

    Same first-order condition as `nash_price` (p = c + mu / (1 - q_i)) but with
    the rivals' actual prices rather than a symmetric guess, so it gives the
    per-turn best-response regret used in the turn log.
    """
    rival_prices = np.asarray(rival_prices, dtype=float)

    def own_share(p: float) -> float:
        full = np.insert(rival_prices, seller_index, p)
        return float(shares(full, params)[seller_index])

    def g(p: float) -> float:
        return p - params.c - params.mu / (1.0 - own_share(p))

    return _bracket_and_solve(g, params, name="best-response")


def best_response_profit(
    rival_prices: np.ndarray, seller_index: int, params: PricingParams
) -> tuple[float, float]:
    """(best-response price, profit at it) for one seller against fixed rivals."""
    p_star = best_response_price(rival_prices, seller_index, params)
    full = np.insert(np.asarray(rival_prices, dtype=float), seller_index, p_star)
    return p_star, float(profits(full, params)[seller_index])
