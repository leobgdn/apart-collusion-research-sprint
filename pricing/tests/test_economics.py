"""The benchmarks the collusion index is normalised against must be right.

If `price_nash` or `price_monopoly` is wrong then every Delta in the project is
wrong by the same amount, so these are checked against published values and
against their own defining properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from collusion.envs.pricing.economics import (
    PricingParams,
    best_response_price,
    best_response_profit,
    compute_benchmarks,
    consumer_surplus,
    monopoly_price,
    nash_price,
    profits,
    shares,
)
from collusion.metrics.action_metrics import compute_action_metrics
from conftest import CALVANO_PRICE_MONOPOLY, CALVANO_PRICE_NASH


def test_benchmarks_match_published_values(params, benchmarks):
    """External validation against Calvano et al. (2020) for n=2."""
    assert benchmarks.price_nash == pytest.approx(CALVANO_PRICE_NASH, abs=1e-3)
    assert benchmarks.price_monopoly == pytest.approx(CALVANO_PRICE_MONOPOLY, abs=1e-3)


def test_monopoly_strictly_above_nash(benchmarks):
    assert benchmarks.price_monopoly > benchmarks.price_nash
    assert benchmarks.profit_monopoly > benchmarks.profit_nash


def test_shares_are_a_proper_distribution_with_outside_option(params):
    s = shares(np.array([1.5, 1.6]), params)
    assert (s > 0).all()
    assert s.sum() < 1.0  # some customers take the outside option


def test_shares_are_monotone_in_own_price(params):
    """Raising your price can only lose you share."""
    cheap = shares(np.array([1.4, 1.6]), params)[0]
    dear = shares(np.array([1.8, 1.6]), params)[0]
    assert cheap > dear


def test_nash_is_a_fixed_point_of_best_response(params, benchmarks):
    """The defining property: at Nash, nobody wants to move."""
    br = best_response_price(np.array([benchmarks.price_nash]), 0, params)
    assert br == pytest.approx(benchmarks.price_nash, abs=1e-9)


def test_cartel_price_is_not_an_equilibrium(params, benchmarks):
    """A cartel is individually unstable -- which is why collusion is interesting."""
    br_price, br_profit = best_response_profit(
        np.array([benchmarks.price_monopoly]), 0, params
    )
    assert br_price < benchmarks.price_monopoly  # the deviation is to undercut
    assert br_profit > benchmarks.profit_monopoly


def test_monopoly_price_maximises_joint_profit(params, benchmarks):
    """Grid-search the symmetric joint profit; the solver should find the peak."""
    grid = np.linspace(params.c + 1e-4, 3.0, 20001)
    joint = [profits(np.full(2, p), params).sum() for p in grid]
    assert grid[int(np.argmax(joint))] == pytest.approx(
        benchmarks.price_monopoly, abs=1e-3
    )


def test_consumer_surplus_falls_under_collusion(params, benchmarks):
    """The welfare cost of collusion: higher prices, worse off consumers."""
    cs_nash = consumer_surplus(np.full(2, benchmarks.price_nash), params)
    cs_mono = consumer_surplus(np.full(2, benchmarks.price_monopoly), params)
    assert cs_nash > cs_mono
    assert benchmarks.cs_nash == pytest.approx(cs_nash)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_benchmarks_solve_for_more_sellers(n):
    """The environment is configurable in n; the solvers must follow."""
    p = PricingParams(n_sellers=n)
    b = compute_benchmarks(p)
    assert b.price_monopoly > b.price_nash
    assert b.profit_monopoly > b.profit_nash
    br = best_response_price(np.full(n - 1, b.price_nash), 0, p)
    assert br == pytest.approx(b.price_nash, abs=1e-9)


def test_more_sellers_compete_prices_down():
    """Sanity on the comparative static, which guards against a sign error."""
    prices = [compute_benchmarks(PricingParams(n_sellers=n)).price_nash for n in (2, 3, 4)]
    assert prices[0] > prices[1] > prices[2]


def test_index_is_zero_at_nash_and_one_at_monopoly(params, benchmarks):
    """The normalisation anchors: Delta must be 0 at Nash and 1 at the cartel."""
    for price, expected in (
        (benchmarks.price_nash, 0.0),
        (benchmarks.price_monopoly, 1.0),
    ):
        path = np.full((50, 2), price)
        m = compute_action_metrics(
            prices=path,
            profits=np.array([profits(row, params) for row in path]),
            consumer_surplus=np.array(
                [consumer_surplus(row, params) for row in path]
            ),
            benchmarks=benchmarks,
            params=params,
        )
        assert m.collusion_index == pytest.approx(expected, abs=1e-9)
        assert m.price_index == pytest.approx(expected, abs=1e-9)


def test_solvers_are_deterministic(params):
    assert nash_price(params) == nash_price(params)
    assert monopoly_price(params) == monopoly_price(params)
