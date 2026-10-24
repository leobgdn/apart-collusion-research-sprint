"""The sweep driver's guards: resumability, run isolation, and the budget cap.

The run-isolation test is a regression test for a bug found during Stage 1: the
resume key is (condition_id, seed), which is not unique across sweeps, so a
second sweep skipped all of its episodes and reported success while silently
reusing the first sweep's results.
"""

from __future__ import annotations

import numpy as np
import pytest

from collusion.envs.pricing.conditions import BASELINE, one_factor_at_a_time
from collusion.envs.pricing.economics import PricingParams, compute_benchmarks
from collusion.envs.pricing.pricing_env import EnvConfig
from collusion.runner.logging_io import read_episode_summaries, read_turn_log
from collusion.runner.run_episode import AgentReply
from collusion.runner.run_sweep import (
    BudgetExceeded,
    BudgetGuard,
    SweepConfig,
    load_config,
    llm_agent_factory,
    run_sweep,
    scripted_agent_factory,
)


@pytest.fixture
def config(tmp_path, params) -> SweepConfig:
    return SweepConfig(
        raw={},
        run_id="unit",
        params=params,
        env_config=EnvConfig(total_periods=12),
        conditions=one_factor_at_a_time()[:2],
        seeds=[1, 2],
        metric_window=10,
        label_threshold=0.5,
        hard_cap_usd=0.0,
        max_parse_retries=2,
        max_parse_failure_rate=0.05,
        logs_dir=tmp_path / "logs",
        episodes_dir=tmp_path / "episodes",
        scripted_noise_sd=0.05,
    )


def _factory(config, policy="colluding"):
    return scripted_agent_factory(policy, config.scripted_noise_sd, config.params.n_sellers)


def test_sweep_runs_every_cell(config):
    report = run_sweep(config, _factory(config), verbose=False)
    assert report.completed == config.total_episodes == 4
    assert report.failed == []
    assert report.total_cost_usd == 0.0


def test_resume_skips_completed_work(config):
    run_sweep(config, _factory(config), verbose=False)
    again = run_sweep(config, _factory(config), verbose=False)
    assert again.completed == 0
    assert again.skipped == config.total_episodes


def test_two_run_ids_do_not_share_results(config):
    """Regression: a second sweep must not inherit the first sweep's episodes."""
    run_sweep(config, _factory(config, "colluding"), verbose=False)

    config.run_id = "unit_competitive"
    second = run_sweep(config, _factory(config, "competitive"), verbose=False)

    assert second.completed == config.total_episodes, (
        "second sweep skipped work belonging to a different run_id"
    )
    rows = read_episode_summaries(config.episodes_dir / "unit_competitive")
    assert rows
    assert all("competitive" in m for r in rows for m in r["agent_models"])
    # And the two runs' scores must actually differ.
    assert np.mean(second.scores) < 0.5


def test_no_resume_refuses_to_append_to_existing_data(config):
    """Appending would duplicate turns and corrupt per-turn statistics."""
    run_sweep(config, _factory(config), verbose=False)
    with pytest.raises(FileExistsError, match="already has data"):
        run_sweep(config, _factory(config), resume=False, verbose=False)


def test_turn_log_is_written_per_run(config):
    run_sweep(config, _factory(config), verbose=False)
    rows = list(read_turn_log(config.log_path))
    # 4 episodes x 12 periods x 2 sellers
    assert len(rows) == 4 * 12 * 2
    assert {r["run_id"] for r in rows} == {"unit"}


def test_episode_summaries_land_under_the_run_id(config):
    run_sweep(config, _factory(config), verbose=False)
    assert (config.episodes_dir / "unit").is_dir()
    assert len(read_episode_summaries(config.episodes_dir / "unit")) == 4


# --- budget guard -------------------------------------------------------------


def test_guard_allows_free_runs():
    guard = BudgetGuard(hard_cap_usd=0.0)
    for _ in range(10):
        guard.record(0.0)
    assert guard.spent_usd == 0.0
    guard.check_projection(1000)  # free work can never breach a zero cap


def test_guard_trips_on_overspend():
    guard = BudgetGuard(hard_cap_usd=1.0)
    guard.record(0.6)
    with pytest.raises(BudgetExceeded, match="over the"):
        guard.record(0.6)


def test_guard_projects_before_spending():
    """Section 3 rule 3: abort if the *projected* total would breach the cap."""
    guard = BudgetGuard(hard_cap_usd=1.0)
    guard.record(0.1)  # mean cost 0.10/episode
    guard.check_projection(5)  # 0.1 + 0.5 = 0.6, fine
    with pytest.raises(BudgetExceeded, match="projected total"):
        guard.check_projection(50)  # 0.1 + 5.0, way over


def test_sweep_aborts_when_a_paid_agent_breaches_a_zero_cap(config):
    """A zero-dollar cap must stop a sweep the moment a model costs money."""

    class PaidAgent:
        agent_id = "paid"
        model_name = "fake-paid-model"

        def act(self, obs, condition, rng, rendered=None):
            return AgentReply(
                text="PLAN: x\nINSIGHT: y\nPRICE: 1.5000",
                tokens_in=100,
                tokens_out=20,
                cost_usd=0.01,
                model_settings={"temperature": 0.7},
            )

    report = run_sweep(
        config,
        lambda condition, seed, benchmarks: [PaidAgent(), PaidAgent()],
        verbose=False,
    )
    # The first episode runs (cost is only known afterwards), then the guard
    # trips and the sweep stops well short of the full design.
    assert report.completed < config.total_episodes
    assert report.total_cost_usd > 0


def test_sweep_stops_on_a_high_parse_failure_rate(config):
    """Section 6.3 / Section 12: stop above a 5% parse-failure rate."""

    class BrokenAgent:
        agent_id = "broken"
        model_name = "fake-unparseable"

        def act(self, obs, condition, rng, rendered=None):
            return "I would rather not give a number."

    report = run_sweep(
        config,
        lambda condition, seed, benchmarks: [BrokenAgent(), BrokenAgent()],
        verbose=False,
    )
    assert report.parse_failure_rate > config.max_parse_failure_rate
    assert report.completed < config.total_episodes


def test_a_failing_episode_is_recorded_not_swallowed(config):
    class ExplodingAgent:
        agent_id = "boom"
        model_name = "fake-exploding"

        def act(self, obs, condition, rng, rendered=None):
            raise RuntimeError("inference backend fell over")

    report = run_sweep(
        config,
        lambda condition, seed, benchmarks: [ExplodingAgent(), ExplodingAgent()],
        verbose=False,
    )
    assert report.completed == 0
    assert len(report.failed) == config.total_episodes
    assert "inference backend fell over" in report.failed[0][2]


# --- gating -------------------------------------------------------------------


def test_llm_agents_are_gated_until_gate_1_clears():
    """Stage 2 must not be reachable by accident."""
    with pytest.raises(NotImplementedError, match="Gate 1"):
        llm_agent_factory()


def test_shipped_config_parses_and_declares_a_zero_cap():
    cfg = load_config()
    assert cfg.hard_cap_usd == 0.0  # Section 0: local models only
    assert cfg.params.n_sellers == 2
    assert len(cfg.conditions) == 8
    assert cfg.metric_window == 50
    assert cfg.env_config.total_periods == 200


def test_pilot_config_uses_the_one_factor_design():
    cfg = load_config(pilot=True)
    assert len(cfg.conditions) == 4  # baseline + one per non-baseline level
    assert cfg.conditions[0] == BASELINE
    assert len(cfg.seeds) == 5


def test_benchmarks_come_from_the_configured_params():
    cfg = load_config()
    assert cfg.benchmarks.as_dict() == compute_benchmarks(cfg.params).as_dict()
    assert cfg.params == PricingParams(n_sellers=2, a=2.0, a0=0.0, c=1.0, mu=0.25)
