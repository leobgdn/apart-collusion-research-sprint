"""Section 3 rule 7: conditions may differ only in the manipulated field.

The test does not eyeball a text diff. It holds the environment state fixed,
varies one factor at a time, and asserts that the only prompt slots whose values
changed are the slots that factor *owns* (conditions.OWNED_SLOTS). A leak -- a
reworded instruction that quietly travels with an unrelated factor -- fails
here rather than silently confounding the sweep.
"""

from __future__ import annotations

import itertools

import pytest

from collusion.envs.pricing import prompts
from collusion.envs.pricing.conditions import (
    FACTORS,
    OWNED_SLOTS,
    Condition,
    full_factorial,
)
from collusion.envs.pricing.pricing_env import EnvConfig, PricingEnvironment
from conftest import fixed_action

# A price path long enough that a 20-period window is full and differs from a
# 3-period window. Identical for every condition, so any prompt difference must
# come from the condition itself.
PRICE_PATH = [(1.50 + 0.01 * t, 1.60 - 0.005 * t) for t in range(25)]
MESSAGES = ("let us both hold at 1.90", "agreed, holding")


def _slots_for(condition: Condition, params, agent_index: int = 0) -> dict[str, str]:
    """Drive the env through the fixed path, then render agent `agent_index`."""
    env = PricingEnvironment(
        condition=condition,
        params=params,
        config=EnvConfig(total_periods=len(PRICE_PATH) + 1),
        seed=7,
    )
    for p0, p1 in PRICE_PATH:
        env.step(
            [
                fixed_action(p0, message=MESSAGES[0]),
                fixed_action(p1, message=MESSAGES[1]),
            ]
        )
    return prompts.build_slots(condition, env.observe(agent_index))


def _unowned_slots() -> set[str]:
    owned = set().union(*OWNED_SLOTS.values())
    sample = _slots_for(full_factorial()[0], _default_params())
    return set(sample) - owned


def _default_params():
    from collusion.envs.pricing.economics import PricingParams

    return PricingParams()


@pytest.mark.parametrize("factor", sorted(FACTORS))
def test_changing_one_factor_changes_only_its_own_slots(factor, params):
    """The core rule-7 check, for every level of every factor."""
    base = full_factorial()[0]
    allowed = OWNED_SLOTS[factor]
    base_slots = _slots_for(base, params)
    for level in FACTORS[factor]:
        if level == getattr(base, factor):
            continue
        other = base.replace(**{factor: level})
        other_slots = _slots_for(other, params)
        assert set(base_slots) == set(other_slots), "slot set must not change"
        changed = {k for k in base_slots if base_slots[k] != other_slots[k]}
        leaked = changed - allowed
        assert not leaked, (
            f"changing {factor} to {level!r} also changed unowned slot(s) "
            f"{sorted(leaked)}; only {sorted(allowed)} may change"
        )


@pytest.mark.parametrize("factor", sorted(FACTORS))
def test_each_factor_actually_changes_something(factor, params):
    """A manipulation that changes no slot is not a manipulation at all.

    Section 12 lists "identical results across conditions (suggests the
    manipulation was not applied)" as a stop-the-line condition; this catches
    the prompt-level version of that before any model is ever called.
    """
    base = full_factorial()[0]
    base_slots = _slots_for(base, params)
    for level in FACTORS[factor]:
        if level == getattr(base, factor):
            continue
        other_slots = _slots_for(base.replace(**{factor: level}), params)
        changed = {k for k in base_slots if base_slots[k] != other_slots[k]}
        assert changed, f"factor {factor}={level!r} changed no prompt slot"


def test_unowned_slots_are_identical_across_every_condition(params):
    """Slots no factor owns must be constant given the same state."""
    unowned = _unowned_slots()
    assert unowned, "expected some condition-invariant slots"
    all_slots = [_slots_for(c, params) for c in full_factorial()]
    for name in sorted(unowned):
        values = {s[name] for s in all_slots}
        assert len(values) == 1, (
            f"slot {name!r} is owned by no factor but took {len(values)} "
            f"different values across conditions: {sorted(values)}"
        )


def test_templates_are_shared_verbatim_by_all_conditions():
    """Conditions fill one template; they never get a template of their own."""
    assert "{horizon_instruction}" in prompts.SYSTEM_TEMPLATE
    assert "{channel_instruction}" in prompts.SYSTEM_TEMPLATE
    assert "{history_table}" in prompts.USER_TEMPLATE
    # Every slot referenced by a template must be produced by build_slots, and
    # vice versa -- otherwise a slot could be silently dropped from a prompt.
    import string

    referenced = {
        name
        for template in (prompts.SYSTEM_TEMPLATE, prompts.USER_TEMPLATE)
        for _, name, _, _ in string.Formatter().parse(template)
        if name
    }
    produced = set(_slots_for(full_factorial()[0], _default_params()))
    assert referenced == produced, (
        f"template/slot mismatch: only in template {sorted(referenced - produced)}, "
        f"only in build_slots {sorted(produced - referenced)}"
    )


def test_no_channel_condition_never_mentions_messaging(params):
    """The negative level of the positive control must be clean."""
    slots = _slots_for(full_factorial()[0].replace(channel="none"), params)
    rendered = prompts.render(
        full_factorial()[0].replace(channel="none"),
        _observation_for("none", params),
    )
    assert slots["message_field"] == ""
    assert slots["inbox_section"] == ""
    assert "MESSAGE:" not in rendered.system
    assert MESSAGES[0] not in rendered.user


def _observation_for(channel: str, params):
    cond = full_factorial()[0].replace(channel=channel)
    env = PricingEnvironment(
        condition=cond,
        params=params,
        config=EnvConfig(total_periods=len(PRICE_PATH) + 1),
        seed=7,
    )
    for p0, p1 in PRICE_PATH:
        env.step(
            [fixed_action(p0, message=MESSAGES[0]), fixed_action(p1, message=MESSAGES[1])]
        )
    return env.observe(0)


def test_private_channel_delivers_the_rivals_message_only(params):
    """With a channel, you see the rival's message and not your own."""
    cond = full_factorial()[0].replace(channel="private")
    rendered = prompts.render(cond, _observation_for("private", params))
    assert MESSAGES[1] in rendered.user  # seller B's message reaches seller A
    assert MESSAGES[0] not in rendered.user  # A does not get its own back
    assert "MESSAGE:" in rendered.system


def test_history_window_truncates_the_visible_table(params):
    """H2's manipulation has to actually shorten what the agent can see."""
    short = _slots_for(full_factorial()[0].replace(history_window="short"), params)
    long = _slots_for(full_factorial()[0].replace(history_window="long"), params)
    short_rows = short["history_table"].count("\n")
    long_rows = long["history_table"].count("\n")
    assert short_rows < long_rows
    # 3 data rows + 1 header vs 20 data rows + 1 header
    assert short_rows == 3
    assert long_rows == 20


def test_all_condition_pairs_differ_only_in_declared_factors(params):
    """Exhaustive cross-check over the whole factorial."""
    cells = full_factorial()
    slots = {c.condition_id: _slots_for(c, params) for c in cells}
    for a, b in itertools.combinations(cells, 2):
        differing = a.differing_factors(b)
        allowed = set().union(*(OWNED_SLOTS[f] for f in differing))
        changed = {
            k
            for k in slots[a.condition_id]
            if slots[a.condition_id][k] != slots[b.condition_id][k]
        }
        assert changed <= allowed, (
            f"{a.condition_id} vs {b.condition_id} differ in factors "
            f"{sorted(differing)} but changed unowned slots "
            f"{sorted(changed - allowed)}"
        )
