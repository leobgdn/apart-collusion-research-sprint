"""Shared test fixtures. Living at the repo root also puts the root on sys.path
so `import collusion` works under pytest's default import mode.
"""

from __future__ import annotations

import pytest

from collusion.agents.parsing import ParsedAction
from collusion.envs.pricing.economics import PricingParams, compute_benchmarks
from collusion.envs.pricing.pricing_env import EnvConfig

# Literature reference values for the default parameters (a=2, a0=0, c=1,
# mu=0.25, n=2), from Calvano, Calzolari, Denicolo & Pastorello (2020),
# "Artificial Intelligence, Algorithmic Pricing, and Collusion", AER 110(10).
# Used as an external check that our solvers are right, not as inputs.
CALVANO_PRICE_NASH = 1.4729
CALVANO_PRICE_MONOPOLY = 1.9249


@pytest.fixture
def params() -> PricingParams:
    return PricingParams()


@pytest.fixture
def benchmarks(params):
    return compute_benchmarks(params)


@pytest.fixture
def short_env_config() -> EnvConfig:
    """Enough periods that the 50-period metric window is fully populated."""
    return EnvConfig(total_periods=60)


def fixed_action(price: float, *, message: str = "") -> ParsedAction:
    """A parsed action with deterministic prose, for replay/diff tests."""
    return ParsedAction(
        price=price,
        plan="hold steady",
        insight="rival is matching",
        message=message,
        ok=True,
        failure_reason=None,
    )


def strip_volatile(record: dict) -> dict:
    """Drop fields that legitimately differ between two identical runs."""
    out = dict(record)
    out.pop("timestamp", None)
    return out
