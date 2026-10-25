"""Section 6.4: the same seed must give exactly the same result for scripted agents.

Exact, not approximate. If a scripted episode is not bit-reproducible then a
resumed sweep is not the same experiment as an uninterrupted one, and no
analysis can be reproduced from the logs.
"""

from __future__ import annotations

import numpy as np
import pytest

from collusion.agents.scripted_agents import make_scripted_agent
from collusion.envs.pricing.conditions import BASELINE, full_factorial
from collusion.runner.run_episode import run_episode
from conftest import strip_volatile

NOISE_SD = 0.05


def _run(seed, params, benchmarks, env_config, *, condition=None, noise_sd=NOISE_SD):
    agents = [
        make_scripted_agent("colluding", benchmarks, noise_sd=noise_sd, agent_id=f"s{i}")
        for i in range(params.n_sellers)
    ]
    return run_episode(
        condition=condition or BASELINE,
        seed=seed,
        agents=agents,
        params=params,
        env_config=env_config,
        label_threshold=0.5,
    )


def test_same_seed_gives_identical_turns(params, benchmarks, short_env_config):
    a = _run(11, params, benchmarks, short_env_config)
    b = _run(11, params, benchmarks, short_env_config)
    assert len(a.turns) == len(b.turns)
    for ta, tb in zip(a.turns, b.turns):
        assert strip_volatile(ta.as_dict()) == strip_volatile(tb.as_dict())


def test_same_seed_gives_identical_scores(params, benchmarks, short_env_config):
    a = _run(11, params, benchmarks, short_env_config).summary
    b = _run(11, params, benchmarks, short_env_config).summary
    # Exact equality, not approx.
    assert a.action_collusion_score == b.action_collusion_score
    assert a.price_index == b.price_index
    assert a.joint_reward == b.joint_reward
    assert a.action_collusion_label == b.action_collusion_label


def test_noise_free_runs_are_identical_across_seeds(
    params, benchmarks, short_env_config
):
    """With no agent-side randomness and no demand shocks, the market is
    deterministic, so the seed must not change anything. This documents that
    seeds in this environment vary *agent* randomness only."""
    a = _run(1, params, benchmarks, short_env_config, noise_sd=0.0).summary
    b = _run(2, params, benchmarks, short_env_config, noise_sd=0.0).summary
    assert a.action_collusion_score == b.action_collusion_score


def test_different_seeds_differ_under_noise(params, benchmarks, short_env_config):
    """The converse check: seeds must do something once agents are stochastic,
    otherwise the sweep's replication is fake."""
    scores = {
        _run(s, params, benchmarks, short_env_config).summary.action_collusion_score
        for s in range(5)
    }
    assert len(scores) == 5


def test_seed_pairs_conditions_rather_than_decorrelating_them(
    params, benchmarks, short_env_config
):
    """A seed must draw the same agent randomness in every condition.

    Scripted agents ignore the prompt entirely, so at a fixed seed every
    condition must produce the identical price path. That is what makes the
    between-condition comparison paired across seeds (see run_episode's
    docstring) and keeps condition contrasts from inheriting unrelated noise.
    """
    reference = None
    for condition in full_factorial():
        result = _run(5, params, benchmarks, short_env_config, condition=condition)
        prices = result.env.price_matrix()
        if reference is None:
            reference = prices
        else:
            assert np.array_equal(prices, reference), (
                f"condition {condition.condition_id} drew different agent "
                "randomness at the same seed"
            )


def test_benchmarks_are_stable_across_instantiations(params, benchmarks):
    from collusion.envs.pricing.economics import compute_benchmarks

    again = compute_benchmarks(params)
    assert again.as_dict() == benchmarks.as_dict()


def test_rng_streams_are_independent_per_agent(params, benchmarks, short_env_config):
    """Both sellers must not draw the identical jitter sequence, which would
    make their prices move in lockstep and fake coordination."""
    result = _run(3, params, benchmarks, short_env_config)
    prices = result.env.price_matrix()
    assert not np.allclose(prices[:, 0], prices[:, 1])


def test_condition_id_is_stable_ordering():
    """Condition ids must not depend on dict ordering, or resume keys break."""
    ids = [c.condition_id for c in full_factorial()]
    assert ids == sorted(ids)
    assert len(set(ids)) == len(ids)
