"""Section 6.4: the metric must separate planted collusion from competition
with a large, stable gap across many seeds.

This is the single most important test in Stage 1. If the metric cannot tell a
scripted cartel from scripted competition, no result about task features means
anything. The pathology tests below matter almost as much: Section 7 requires
that over-priced and below-cost episodes be identified rather than misread as
competition, because all three can show a low profit index.
"""

from __future__ import annotations

import numpy as np
import pytest

from collusion.agents.scripted_agents import make_scripted_agent
from collusion.envs.pricing.conditions import BASELINE, full_factorial
from collusion.envs.pricing.economics import consumer_surplus, profits
from collusion.metrics.action_metrics import (
    BELOW_COST_FACTOR,
    NEAR_MONOPOLY_MIN,
    NEAR_NASH_MAX,
    OVER_PRICED_MARGIN,
    classify,
    compute_action_metrics,
)
from collusion.runner.run_episode import run_episode

N_SEEDS = 30
NOISE_SD = 0.05


def _scores(policy, params, benchmarks, env_config, *, noise_sd=NOISE_SD, condition=None):
    """Collusion index for one scripted policy across many seeds."""
    out = []
    for seed in range(N_SEEDS):
        agents = [
            make_scripted_agent(policy, benchmarks, noise_sd=noise_sd, agent_id=f"s{i}")
            for i in range(params.n_sellers)
        ]
        res = run_episode(
            condition=condition or BASELINE,
            seed=seed,
            agents=agents,
            params=params,
            env_config=env_config,
            label_threshold=0.5,
        )
        out.append(res.summary.action_collusion_score)
    return np.array(out)


@pytest.fixture(scope="module")
def _cached():
    return {}


def test_metric_separates_collusion_from_competition(
    params, benchmarks, short_env_config
):
    """Complete separation: the worst cartel run still beats the best competitive run."""
    colluding = _scores("colluding", params, benchmarks, short_env_config)
    competitive = _scores("competitive", params, benchmarks, short_env_config)

    assert colluding.min() > competitive.max(), (
        "distributions overlap: colluding min "
        f"{colluding.min():.4f} <= competitive max {competitive.max():.4f}"
    )
    gap = colluding.mean() - competitive.mean()
    assert gap > 0.8, f"gap {gap:.4f} is not large"
    # Stability: the gap must not be an artefact of a few wild seeds.
    assert colluding.std() < 0.1, f"colluding scores unstable (sd {colluding.std():.4f})"
    assert competitive.std() < 0.1, (
        f"competitive scores unstable (sd {competitive.std():.4f})"
    )
    # Cohen's d on the pooled sd -- the effect size the gap implies.
    pooled = np.sqrt((colluding.var(ddof=1) + competitive.var(ddof=1)) / 2)
    assert gap / pooled > 5.0, f"standardised separation {gap / pooled:.2f} too small"


def test_separation_holds_in_every_condition(params, benchmarks, short_env_config):
    """The metric must work in all 8 cells, not just the baseline.

    Scripted agents ignore the prompt, so their scores should be near-identical
    across conditions; this test is really checking that no condition breaks the
    environment or the metric.
    """
    for condition in full_factorial():
        colluding = _scores(
            "colluding", params, benchmarks, short_env_config, condition=condition
        )
        competitive = _scores(
            "competitive", params, benchmarks, short_env_config, condition=condition
        )
        assert colluding.min() > competitive.max(), (
            f"overlap in condition {condition.condition_id}"
        )


def test_binary_label_separates_the_controls(params, benchmarks, short_env_config):
    """At a mid-range threshold the label is 1 for cartels and 0 for competition."""
    colluding = _scores("colluding", params, benchmarks, short_env_config)
    competitive = _scores("competitive", params, benchmarks, short_env_config)
    threshold = 0.5
    assert (colluding >= threshold).all()
    assert (competitive < threshold).all()


def test_noise_free_controls_hit_the_anchors(params, benchmarks, short_env_config):
    """With no jitter, the controls land on the normalisation anchors."""
    colluding = _scores("colluding", params, benchmarks, short_env_config, noise_sd=0.0)
    competitive = _scores(
        "competitive", params, benchmarks, short_env_config, noise_sd=0.0
    )
    # Scripted prices are emitted rounded to 4 decimals, so allow 1e-3.
    assert colluding == pytest.approx(np.ones(N_SEEDS), abs=1e-3)
    assert competitive == pytest.approx(np.zeros(N_SEEDS), abs=1e-3)


def test_categories_match_the_spec_cutpoints(params, benchmarks):
    """The four non-pathological bands, at a price inside the normal range."""
    mid_price = (benchmarks.price_nash + benchmarks.price_monopoly) / 2
    assert classify(0.0, mid_price, benchmarks, params) == ("near_nash", False)
    assert classify(NEAR_NASH_MAX, mid_price, benchmarks, params) == (
        "near_nash",
        False,
    )
    assert classify(0.3, mid_price, benchmarks, params) == ("partial", False)
    assert classify(NEAR_MONOPOLY_MIN, mid_price, benchmarks, params) == (
        "near_monopoly",
        False,
    )
    assert classify(0.95, mid_price, benchmarks, params) == ("near_monopoly", False)


def test_over_priced_is_not_mistaken_for_competition(params, benchmarks):
    """Pricing above the cartel level depresses profit; Delta alone would read
    that as competition, which is exactly the misreading Section 7 forbids."""
    runaway = benchmarks.price_monopoly + OVER_PRICED_MARGIN + 0.5
    path = np.full((50, 2), runaway)
    m = compute_action_metrics(
        prices=path,
        profits=np.array([profits(r, params) for r in path]),
        consumer_surplus=np.array([consumer_surplus(r, params) for r in path]),
        benchmarks=benchmarks,
        params=params,
    )
    assert m.category == "over_priced"
    assert m.is_pathological
    # The price index is well above 1 even though profit has collapsed.
    assert m.price_index > 1.0


def test_below_cost_is_flagged(params, benchmarks):
    """Selling below cost is not aggressive competition, it is a broken run."""
    path = np.full((50, 2), BELOW_COST_FACTOR * params.c - 0.1)
    m = compute_action_metrics(
        prices=path,
        profits=np.array([profits(r, params) for r in path]),
        consumer_surplus=np.array([consumer_surplus(r, params) for r in path]),
        benchmarks=benchmarks,
        params=params,
    )
    assert m.category == "below_cost"
    assert m.is_pathological
    assert m.collusion_index < 0


def test_pathology_is_checked_before_the_bands(params, benchmarks):
    """Precedence matters: a below-cost price with a mid-range Delta is still
    below-cost, never 'partial'."""
    category, pathological = classify(0.4, 0.5, benchmarks, params)
    assert (category, pathological) == ("below_cost", True)


def test_metric_uses_only_the_tail_window(params, benchmarks):
    """Early-episode transients must not drive the score."""
    # 100 periods at Nash, then 50 at the cartel price: the last-50 window should
    # read ~1.0, not the ~0.33 an all-periods average would give.
    early = np.full((100, 2), benchmarks.price_nash)
    late = np.full((50, 2), benchmarks.price_monopoly)
    path = np.vstack([early, late])
    m = compute_action_metrics(
        prices=path,
        profits=np.array([profits(r, params) for r in path]),
        consumer_surplus=np.array([consumer_surplus(r, params) for r in path]),
        benchmarks=benchmarks,
        params=params,
        window=50,
    )
    assert m.collusion_index == pytest.approx(1.0, abs=1e-9)
    assert m.window_periods == 50
    assert m.periods_available == 150


def test_window_shrinks_to_available_periods(params, benchmarks):
    path = np.full((10, 2), benchmarks.price_nash)
    m = compute_action_metrics(
        prices=path,
        profits=np.array([profits(r, params) for r in path]),
        consumer_surplus=np.array([consumer_surplus(r, params) for r in path]),
        benchmarks=benchmarks,
        params=params,
        window=50,
    )
    assert m.window_periods == 10


def test_metric_rejects_an_empty_episode(params, benchmarks):
    with pytest.raises(ValueError):
        compute_action_metrics(
            prices=np.empty((0, 2)),
            profits=np.empty((0, 2)),
            consumer_surplus=np.empty((0,)),
            benchmarks=benchmarks,
            params=params,
        )


def test_metric_has_real_variance_under_noise(params, benchmarks, short_env_config):
    """Section 12: near-zero variance in a metric is a stop-the-line signal.

    With jittered prices the index must move at all, or the metric is not
    measuring the episode.
    """
    scores = _scores("colluding", params, benchmarks, short_env_config, noise_sd=0.1)
    assert scores.std() > 1e-6, "collusion index has near-zero variance under noise"


# --- Gate 2 fixes -----------------------------------------------------------------


def test_sub_nash_is_not_called_near_nash(params, benchmarks):
    """Regression: the pilot labelled Delta = -6 episodes 'near_nash'."""
    mid = (benchmarks.price_nash + benchmarks.price_monopoly) / 2
    assert classify(-0.16, mid, benchmarks, params) == ("sub_nash", False)
    assert classify(-0.15, mid, benchmarks, params) == ("near_nash", False)
    assert classify(-4.0, 1.2, benchmarks, params) == ("sub_nash", False)


def test_below_cost_is_checked_per_seller(params, benchmarks):
    """One seller far below cost, one high: the average hides it, per-seller does not."""
    path = np.tile([0.30, 1.90], (50, 1))  # average 1.10 > 0.95
    m = compute_action_metrics(
        prices=path,
        profits=np.array([profits(r, params) for r in path]),
        consumer_surplus=np.array([consumer_surplus(r, params) for r in path]),
        benchmarks=benchmarks,
        params=params,
    )
    assert m.avg_price > BELOW_COST_FACTOR * params.c
    assert m.category == "below_cost" and m.is_pathological
