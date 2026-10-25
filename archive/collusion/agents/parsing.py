"""Parsing of an agent's free-text reply into a structured action.

Deliberately strict about the price (an unreadable or out-of-range price is a
parse failure, counted and reported, never silently defaulted) and lenient about
the prose fields, which are only used by the text judge and the probe layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict

# A price must be readable and economically meaningful. The upper bound is well
# above the joint-monopoly price (~1.92 at the default parameters), so a clamp
# here never binds on plausible play; it only catches runaway output.
MIN_PRICE = 0.0
MAX_PRICE = 4.0

_FIELD_RE = {
    "plan": re.compile(r"^\s*PLAN\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE),
    "insight": re.compile(r"^\s*INSIGHT\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE),
    "message": re.compile(r"^\s*MESSAGE\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE),
}
_PRICE_RE = re.compile(
    r"^\s*PRICE\s*:\s*\$?\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE | re.MULTILINE
)


@dataclass(frozen=True)
class ParsedAction:
    """The structured action taken from one agent reply."""

    price: float | None
    plan: str
    insight: str
    message: str
    ok: bool
    failure_reason: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def parse_response(text: str, *, expect_message: bool) -> ParsedAction:
    """Parse an agent reply.

    `expect_message` is the condition's channel setting. A MESSAGE in a
    no-channel condition is dropped rather than failing the turn, but it is
    reported via `failure_reason` so the sweep can flag a leaking manipulation.
    """
    text = text or ""
    fields = {}
    for name, pattern in _FIELD_RE.items():
        match = pattern.search(text)
        fields[name] = match.group(1).strip() if match else ""

    price_match = _PRICE_RE.search(text)
    if price_match is None:
        return ParsedAction(
            price=None,
            plan=fields["plan"],
            insight=fields["insight"],
            message="",
            ok=False,
            failure_reason="no_price_field",
        )

    price = float(price_match.group(1))
    if not MIN_PRICE <= price <= MAX_PRICE:
        return ParsedAction(
            price=None,
            plan=fields["plan"],
            insight=fields["insight"],
            message="",
            ok=False,
            failure_reason=f"price_out_of_range:{price}",
        )

    reason = None
    message = fields["message"]
    if message and not expect_message:
        # The agent tried to talk in a condition with no channel. Suppress it so
        # the manipulation holds, but surface it.
        message = ""
        reason = "message_in_no_channel_condition"

    return ParsedAction(
        price=price,
        plan=fields["plan"],
        insight=fields["insight"],
        message=message,
        ok=True,
        failure_reason=reason,
    )


def format_response(
    *, price: float, plan: str, insight: str, message: str | None = None
) -> str:
    """Render a reply in the format `parse_response` expects.

    Used by the scripted agents so they travel the same parse path as a model.
    """
    lines = [f"PLAN: {plan}", f"INSIGHT: {insight}"]
    if message is not None:
        lines.append(f"MESSAGE: {message}")
    lines.append(f"PRICE: {price:.4f}")
    return "\n".join(lines)
