"""The sweep driver's guards: resumability, run isolation, balanced batches,
the wall-clock guard, and the stop-the-line monitors.

The run-isolation test is a regression test for a bug found during Stage 1: the
resume key is (condition_id, seed), which is not unique across sweeps, so a
second sweep skipped all of its episodes and reported success while silently
reusing the first sweep's results.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from collusion.envs.pricing.conditions import BASELINE, pilot_design
from collusion.envs.pricing.economics import PricingParams, compute_benchmarks
from collusion.envs.pricing.pricing_env import EnvConfig
from collusion.runner.logging_io import read_episode_summaries, read_turn_log
from collusion.runner.replay import verify_round_trip
from collusion.runner.run_episode import AgentReply
from collusion.runner.run_sweep import (
    SweepConfig,
    WallClockGuard,
    batch_of_seed,
    compact_turn_log,
    complete_batches,
    load_config,
    run_sweep,
    scripted_agent_factory,
    seeds_for_batch,
)

PERIODS = 12


@pytest.fixture
def config(tmp_path, params) -> SweepConfig:
    return SweepConfig(
        raw={},
        run_id="unit",
        params=params,
        env_config=EnvConfig(total_periods=PERIODS),
        conditions=pilot_design()[:2],
        seeds_per_batch=2,
        n_batches=1,
        metric_window=10,
        label_threshold=0.5,
        max_parse_retries=2,
        max_parse_failure_rate=0.10,
        logs_dir=tmp_path / "logs",
        episodes_dir=tmp_path / "episodes",
        scripted_noise_sd=0.05,
        concurrency=1,
    )


def _factory(config, policy="colluding"):
    return scripted_agent_factory(policy, config.scripted_noise_sd, config.params)


def test_sweep_runs_every_cell(config):
    report = run_sweep(config, _factory(config), verbose=False)
    assert report.completed == config.total_episodes == 4
    assert report.failed == []
    assert report.stop_reason is None


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
    assert np.mean(second.scores) < 0.5


def test_no_resume_refuses_to_append_to_existing_data(config):
    """Appending would duplicate turns and corrupt per-turn statistics."""
    run_sweep(config, _factory(config), verbose=False)
    with pytest.raises(FileExistsError, match="already has data"):
        run_sweep(config, _factory(config), resume=False, verbose=False)


def test_turn_log_is_written_per_run(config):
    run_sweep(config, _factory(config), verbose=False)
    rows = list(read_turn_log(config.log_path))
    assert len(rows) == 4 * PERIODS * 2
    assert {r["run_id"] for r in rows} == {"unit"}


def test_concurrent_sweep_matches_sequential(config):
    """Concurrency must not change results or interleave episodes in the log."""
    run_sweep(config, _factory(config), verbose=False)
    seq = {
        r["episode_id"]: r["action_collusion_score"]
        for r in read_episode_summaries(config.run_episodes_dir)
    }

    config.run_id = "unit_parallel"
    config.concurrency = 4
    run_sweep(config, _factory(config), verbose=False)
    par = {
        r["episode_id"]: r["action_collusion_score"]
        for r in read_episode_summaries(config.run_episodes_dir)
    }
    assert seq == par

    rows = list(read_turn_log(config.log_path))
    # Each episode's turns are contiguous: no interleaving between episodes.
    order = [r["episode_id"] for r in rows]
    blocks = [eid for i, eid in enumerate(order) if i == 0 or order[i - 1] != eid]
    assert len(blocks) == len(set(blocks)) == config.total_episodes


# --- balanced batches ----------------------------------------------------------


def test_batches_partition_the_seeds():
    assert seeds_for_batch(0, 3) == [1, 2, 3]
    assert seeds_for_batch(1, 3) == [4, 5, 6]
    assert all(batch_of_seed(s, 3) == 1 for s in seeds_for_batch(1, 3))


def test_work_queue_is_batch_major(config):
    config.n_batches = 3
    queue = config.work_queue()
    batches = [b for b, _, _ in queue]
    assert batches == sorted(batches)
    for b in range(3):
        cells = [(c.condition_id, s) for bb, c, s in queue if bb == b]
        assert len(cells) == len(config.conditions) * config.seeds_per_batch


def test_complete_batches_only_counts_balanced_ones(config):
    config.n_batches = 2
    run_sweep(config, _factory(config), verbose=False)
    rows = read_episode_summaries(config.run_episodes_dir)
    assert complete_batches(rows, config.conditions, config.seeds_per_batch) == [0, 1]
    # Remove one episode of batch 1: batch 1 is no longer balanced.
    victim = next(r for r in rows if batch_of_seed(r["seed"], 2) == 1)
    rows = [r for r in rows if r is not victim]
    assert complete_batches(rows, config.conditions, config.seeds_per_batch) == [0]


# --- wall-clock guard (acceptance test 7) -------------------------------------


class SlowAgent:
    """Agent that takes real time per turn, so a deadline can bite mid-episode."""

    model_name = "fake-slow"

    def __init__(self, i):
        self.agent_id = f"slow_{i}"

    def act(self, obs, condition, rng, rendered=None):
        time.sleep(0.01)
        return json.dumps({"price": 1.7, "plan": "hold", "insight": "steady"})


def _slow_factory(condition, seed, benchmarks):
    return [SlowAgent(0), SlowAgent(1)]


def test_wall_clock_guard_stops_cleanly_and_leaves_valid_logs(config, params):
    """Acceptance test 7: an abort mid-episode leaves only whole, valid episodes."""
    config.env_config = EnvConfig(total_periods=40)  # ~0.8 s per episode
    config.n_batches = 3
    config.concurrency = 2
    config.deadline = time.time() + 2.0
    report = run_sweep(config, _slow_factory, verbose=False)
    assert report.stop_reason is not None and "wall-clock" in report.stop_reason
    assert report.completed < config.total_episodes

    summaries = read_episode_summaries(config.run_episodes_dir)
    rows = list(read_turn_log(config.log_path))
    finished = {s["episode_id"] for s in summaries}
    assert finished, "nothing finished before the deadline; test is vacuous"
    # Only finished episodes are in the log, each whole.
    assert {r["episode_id"] for r in rows} == finished
    for eid in finished:
        assert sum(r["episode_id"] == eid for r in rows) == 40 * 2
    # And what is there round-trips.
    periods = {eid: 40 for eid in finished}
    assert not verify_round_trip(rows, params=params, episode_periods=periods)


def test_guard_refuses_to_launch_what_cannot_finish():
    guard = WallClockGuard(deadline=time.time() + 5)
    assert guard.may_launch()
    guard.record(60.0)  # episodes take a minute; only 5 s remain
    assert not guard.may_launch()
    assert WallClockGuard(deadline=None).may_launch()


def test_resume_after_abort_completes_the_design(config):
    config.env_config = EnvConfig(total_periods=40)
    config.deadline = time.time() + 1.0
    run_sweep(config, _slow_factory, verbose=False)
    config.deadline = None
    report = run_sweep(config, _slow_factory, verbose=False)
    assert report.stop_reason is None
    rows = list(read_turn_log(config.log_path))
    assert len(rows) == config.total_episodes * 40 * 2  # no duplicates


def test_compaction_drops_orphan_turns(config):
    run_sweep(config, _factory(config), verbose=False)
    rows = list(read_turn_log(config.log_path))
    orphan = dict(rows[0], episode_id="ghost#seed=99")
    with config.log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(orphan) + "\n")
    assert compact_turn_log(config.log_path, config.run_episodes_dir) == 1
    assert len(list(read_turn_log(config.log_path))) == len(rows)


# --- stop-the-line monitors ----------------------------------------------------


def test_sweep_stops_on_a_high_parse_failure_rate(config):
    class BrokenAgent:
        agent_id = "broken"
        model_name = "fake-unparseable"

        def act(self, obs, condition, rng, rendered=None):
            return "I would rather not give a number."

    report = run_sweep(
        config, lambda c, s, b: [BrokenAgent(), BrokenAgent()], verbose=False
    )
    assert report.parse_failure_rate > config.max_parse_failure_rate
    assert "parse-failure" in report.stop_reason
    assert report.completed < config.total_episodes


def test_sweep_stops_on_a_reasoning_block(config):
    class ThinkingAgent:
        agent_id = "thinker"
        model_name = "fake-thinking"

        def act(self, obs, condition, rng, rendered=None):
            return '<think>hmm</think>{"price": 1.6, "plan": "x", "insight": "y"}'

    report = run_sweep(
        config, lambda c, s, b: [ThinkingAgent(), ThinkingAgent()], verbose=False
    )
    assert report.stop_reason == "reasoning block found in model output"
    assert report.completed == 1


def test_sweep_stops_on_a_prompt_token_mismatch(config):
    class MismatchAgent:
        agent_id = "mm"
        model_name = "fake-mismatch"

        def act(self, obs, condition, rng, rendered=None):
            return AgentReply(
                text='{"price": 1.6, "plan": "x", "insight": "y"}',
                call_info={"prompt_token_mismatch": True},
            )

    report = run_sweep(
        config, lambda c, s, b: [MismatchAgent(), MismatchAgent()], verbose=False
    )
    assert "prompt" in report.stop_reason


def test_a_failing_episode_is_recorded_not_swallowed(config):
    class ExplodingAgent:
        agent_id = "boom"
        model_name = "fake-exploding"

        def act(self, obs, condition, rng, rendered=None):
            raise RuntimeError("inference backend fell over")

    report = run_sweep(
        config, lambda c, s, b: [ExplodingAgent(), ExplodingAgent()], verbose=False
    )
    assert report.completed == 0
    assert len(report.failed) == config.total_episodes
    assert "inference backend fell over" in report.failed[0][2]


# --- shipped config -------------------------------------------------------------


def test_shipped_pilot_config():
    cfg = load_config(pilot=True)
    assert cfg.params == PricingParams(n_sellers=2, a=2.0, a0=0.0, c=1.0, mu=0.25)
    assert len(cfg.conditions) == 9  # 8 crossed cells + placebo
    assert cfg.conditions[0] == BASELINE
    assert cfg.metric_window == 30
    assert cfg.env_config.total_periods == 80
    assert len(cfg.seeds) == 5
    assert cfg.raw["models"]["agent"]["enable_thinking"] is False


def test_benchmarks_come_from_the_configured_params():
    cfg = load_config()
    assert cfg.benchmarks.as_dict() == compute_benchmarks(cfg.params).as_dict()


def test_shipped_main_config_matches_gate2_approval():
    cfg = load_config(pilot=False)
    assert len(cfg.conditions) == 9
    assert cfg.env_config.total_periods == 80 and cfg.metric_window == 30
    assert cfg.seeds_per_batch == 2 and cfg.n_batches == 25
    assert cfg.total_episodes == 450
    assert cfg.label_threshold == 0.3
    grid = cfg.env_config.price_grid
    assert grid[0] == 1.0 and grid[-1] == 2.5 and len(grid) == 31
