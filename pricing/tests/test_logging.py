"""Section 5 (data contract) and the Section 6.4 log round-trip test.

The round-trip is the load-bearing one: it replays the logged actions through a
fresh environment, re-renders each prompt from the reconstructed state, and
demands a byte-identical match with the `prompt_messages` that were logged. If
this fails, the log does not describe what the agent actually saw, and the probe
layer would be training on prompts that were never sent (Section 12).
"""

from __future__ import annotations

import json

import pytest

from collusion.agents.scripted_agents import make_scripted_agent
from collusion.envs.pricing.conditions import full_factorial
from collusion.runner.logging_io import (
    REQUIRED_EPISODE_FIELDS,
    REQUIRED_TURN_FIELDS,
    TurnLogWriter,
    completed_episode_keys,
    episode_id_for,
    read_episode_summaries,
    read_turn_log,
    write_episode_summary,
)
from collusion.runner.replay import (
    group_turns_by_episode,
    replay_episode,
    verify_round_trip,
)
from collusion.runner.run_episode import run_episode

# Shorter than a real episode; the round trip does not depend on length.
PERIODS = 25


@pytest.fixture
def episode_runs(params, benchmarks):
    """One scripted episode per condition, with jitter so prices are non-trivial."""
    from collusion.envs.pricing.pricing_env import EnvConfig

    results = []
    for condition in full_factorial():
        agents = [
            make_scripted_agent(
                "colluding", benchmarks, noise_sd=0.05, agent_id=f"seller_{i}"
            )
            for i in range(params.n_sellers)
        ]
        results.append(
            run_episode(
                condition=condition,
                seed=4,
                agents=agents,
                run_id="test_run",
                params=params,
                env_config=EnvConfig(total_periods=PERIODS),
                label_threshold=0.5,
            )
        )
    return results


def test_turn_records_carry_every_required_field(episode_runs):
    for result in episode_runs:
        for turn in result.turns:
            payload = turn.as_dict()
            missing = [f for f in REQUIRED_TURN_FIELDS if f not in payload]
            assert not missing, f"turn record missing {missing}"


def test_episode_summaries_carry_every_required_field(episode_runs):
    for result in episode_runs:
        payload = result.summary.as_dict()
        missing = [f for f in REQUIRED_EPISODE_FIELDS if f not in payload]
        assert not missing, f"episode summary missing {missing}"


def test_prompt_messages_are_logged_verbatim(episode_runs):
    """The exact list as sent -- role/content dicts, not a flattened string."""
    for result in episode_runs:
        for turn in result.turns:
            assert isinstance(turn.prompt_messages, list)
            assert [m["role"] for m in turn.prompt_messages] == ["system", "user"]
            assert all(isinstance(m["content"], str) for m in turn.prompt_messages)
            assert turn.prompt_messages[1]["content"].startswith("Period ")


def test_log_round_trips(episode_runs, params):
    """Section 6.4: every logged prompt must be reconstructible from the log."""
    rows = [t.as_dict() for result in episode_runs for t in result.turns]
    periods = {
        result.summary.episode_id: result.summary.n_periods for result in episode_runs
    }
    mismatches = verify_round_trip(rows, params=params, episode_periods=periods)
    assert not mismatches, "round-trip mismatches:\n" + "\n".join(
        str(m) for m in mismatches[:5]
    )


def test_round_trip_covers_every_turn(episode_runs, params):
    """A round trip that silently skipped turns would pass vacuously."""
    rows = [t.as_dict() for result in episode_runs for t in result.turns]
    grouped = group_turns_by_episode(rows)
    total = 0
    for episode_id, turn_rows in grouped.items():
        expected = next(
            r.summary.n_periods
            for r in episode_runs
            if r.summary.episode_id == episode_id
        )
        replayed = replay_episode(turn_rows, params=params, total_periods=expected)
        assert len(replayed) == len(turn_rows)
        assert all(t.matches for t in replayed)
        total += len(replayed)
    assert total == len(rows)


def test_round_trip_detects_a_tampered_prompt(episode_runs, params):
    """Negative control: the round trip must actually be able to fail."""
    rows = [t.as_dict() for result in episode_runs for t in result.turns]
    tampered = [json.loads(json.dumps(r)) for r in rows]
    tampered[3]["prompt_messages"][1]["content"] += "\n(an edit that never happened)"
    periods = {
        result.summary.episode_id: result.summary.n_periods for result in episode_runs
    }
    mismatches = verify_round_trip(tampered, params=params, episode_periods=periods)
    assert mismatches, "round-trip test cannot detect a modified prompt"


def test_jsonl_write_read_round_trip(episode_runs, tmp_path):
    path = tmp_path / "logs" / "turns.jsonl"
    with TurnLogWriter(path) as writer:
        for turn in episode_runs[0].turns:
            writer.write(turn)
    rows = list(read_turn_log(path))
    assert len(rows) == len(episode_runs[0].turns)
    assert rows[0]["prompt_messages"] == episode_runs[0].turns[0].prompt_messages
    assert rows[0]["condition_id"] == episode_runs[0].summary.condition_id


def test_writer_rejects_an_incomplete_record(tmp_path, episode_runs):
    turn = episode_runs[0].turns[0]
    with TurnLogWriter(tmp_path / "t.jsonl") as writer:
        broken = turn.as_dict()
        del broken["regret"]

        class Fake:
            def as_dict(self_inner):
                return broken

        with pytest.raises(ValueError, match="missing required fields"):
            writer.write(Fake())


def test_episode_summaries_persist_and_reload(episode_runs, tmp_path):
    for result in episode_runs:
        write_episode_summary(result.summary, tmp_path)
    rows = read_episode_summaries(tmp_path)
    assert len(rows) == len(episode_runs)
    assert {r["condition_id"] for r in rows} == {
        c.condition_id for c in full_factorial()
    }


def test_resume_keys_identify_finished_work(episode_runs, tmp_path):
    """Section 3 rule 5: completed (condition_id, seed) runs are skipped."""
    for result in episode_runs:
        write_episode_summary(result.summary, tmp_path)
    keys = completed_episode_keys(tmp_path)
    assert len(keys) == len(episode_runs)
    for result in episode_runs:
        assert (result.summary.condition_id, result.summary.seed) in keys


def test_incomplete_episodes_are_not_treated_as_done(episode_runs, tmp_path):
    summary = episode_runs[0].summary
    summary.completed = False
    write_episode_summary(summary, tmp_path)
    assert completed_episode_keys(tmp_path) == set()


def test_episode_id_is_the_condition_seed_key():
    assert episode_id_for("channel=none", 7) == "channel=none#seed=7"


def test_no_api_cost_was_incurred(episode_runs):
    """Scripted controls must be free; a non-zero cost means a model was called."""
    for result in episode_runs:
        assert result.summary.total_cost_usd == 0.0
        assert result.summary.total_tokens_in == 0
        assert result.summary.total_tokens_out == 0


def test_scripted_runs_have_no_parse_failures(episode_runs):
    """The scripted agents emit the real reply format, so the parser must accept
    them; a failure here means the format and the parser have drifted apart."""
    for result in episode_runs:
        assert result.summary.parse_failures == 0
        for turn in result.turns:
            assert turn.parsed_action["ok"] is True
            assert turn.retries == 0


def test_channel_field_matches_the_condition(episode_runs):
    for result in episode_runs:
        expected = "private" if "channel=private" in result.summary.condition_id else "none"
        for turn in result.turns:
            assert turn.channel == expected
            if expected == "none":
                assert turn.message_text == ""
            else:
                assert turn.message_text


def test_regret_is_non_negative_and_present(episode_runs):
    for result in episode_runs:
        for turn in result.turns:
            assert turn.regret >= 0.0
            assert "best_response_price" in turn.metric_components
