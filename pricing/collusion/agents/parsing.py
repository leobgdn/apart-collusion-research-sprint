"""Parsing of an agent's reply into a structured action.

The reply format is a single JSON object (Phase 1 override 4):
`{"price": ..., "plan": ..., "insight": ...}` plus `"message"` only when the
channel is on.

Deliberately strict about the price (an unreadable or out-of-range price is a
parse failure, counted and reported, never silently defaulted) and lenient about
the prose fields, which are only used by the text measures and the probe layer.
Prose over the word cap is truncated, not failed, and the truncation is recorded.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict, field

# A price must be readable and economically meaningful. The upper bound is well
# above the joint-monopoly price (~1.92 at the default parameters), so a clamp
# here never binds on plausible play; it only catches runaway output.
MIN_PRICE = 0.0
MAX_PRICE = 4.0

# Word cap on every prose field (plan, insight, message).
MAX_WORDS = 40

# Qwen3 thinking output. With thinking disabled none should ever appear; any hit
# is a stop-the-line condition (Phase 1 Section 8), so it is detected here.
_REASONING_RE = re.compile(r"<think>|</think>", re.IGNORECASE)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class ParsedAction:
    """The structured action taken from one agent reply."""

    price: float | None
    plan: str
    insight: str
    message: str
    ok: bool
    failure_reason: str | None = None
    truncated_fields: tuple[str, ...] = field(default_factory=tuple)
    has_reasoning_block: bool = False

    def as_dict(self) -> dict:
        out = asdict(self)
        out["truncated_fields"] = list(self.truncated_fields)
        return out


def has_reasoning_block(text: str) -> bool:
    return bool(_REASONING_RE.search(text or ""))


def _truncate(text: str) -> tuple[str, bool]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return text.strip(), False
    return " ".join(words[:MAX_WORDS]), True


def _extract_object(text: str) -> dict | None:
    """The reply's JSON object, tolerating code fences and surrounding prose."""
    stripped = _FENCE_RE.sub("", text.strip())
    candidates = [stripped]
    match = _OBJECT_RE.search(stripped)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _fail(
    reason: str, *, reasoning: bool, plan: str = "", insight: str = ""
) -> ParsedAction:
    return ParsedAction(
        price=None,
        plan=plan,
        insight=insight,
        message="",
        ok=False,
        failure_reason=reason,
        has_reasoning_block=reasoning,
    )


def _read_price(raw) -> float | None:
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return float(str(raw).strip().lstrip("$"))
    except ValueError:
        return None


def parse_response(
    text: str,
    *,
    expect_message: bool,
    price_grid: tuple[float, ...] | None = None,
) -> ParsedAction:
    """Parse an agent reply.

    `expect_message` is the condition's channel setting. A message in a
    no-channel condition is dropped rather than failing the turn, but it is
    reported via `failure_reason` so the sweep can flag a leaking manipulation.
    """
    text = text or ""
    reasoning = has_reasoning_block(text)
    obj = _extract_object(text)
    if obj is None:
        return _fail("no_json_object", reasoning=reasoning)

    fields = {str(k).strip().lower(): v for k, v in obj.items()}
    truncated: list[str] = []
    prose: dict[str, str] = {}
    for name in ("plan", "insight", "message"):
        value = fields.get(name)
        prose[name], cut = _truncate("" if value is None else str(value))
        if cut:
            truncated.append(name)

    price = _read_price(fields.get("price"))
    if price is None:
        reason = "no_price_field" if "price" not in fields else "unreadable_price"
        return _fail(reason, reasoning=reasoning, plan=prose["plan"], insight=prose["insight"])
    if not MIN_PRICE <= price <= MAX_PRICE:
        return _fail(
            f"price_out_of_range:{price}",
            reasoning=reasoning,
            plan=prose["plan"],
            insight=prose["insight"],
        )

    if price_grid is not None and not any(abs(price - g) < 1e-6 for g in price_grid):
        return _fail(
            f"price_off_grid:{price}",
            reasoning=reasoning,
            plan=prose["plan"],
            insight=prose["insight"],
        )

    reason = None
    message = prose["message"]
    if message and not expect_message:
        # The agent tried to talk in a condition with no channel. Suppress it so
        # the manipulation holds, but surface it.
        message = ""
        reason = "message_in_no_channel_condition"
        if "message" in truncated:
            truncated.remove("message")

    return ParsedAction(
        price=price,
        plan=prose["plan"],
        insight=prose["insight"],
        message=message,
        ok=True,
        failure_reason=reason,
        truncated_fields=tuple(truncated),
        has_reasoning_block=reasoning,
    )


def format_response(
    *, price: float, plan: str, insight: str, message: str | None = None
) -> str:
    """Render a reply in the format `parse_response` expects.

    Used by the scripted agents so they travel the same parse path as a model.
    """
    payload: dict = {"price": round(price, 4), "plan": plan, "insight": insight}
    if message is not None:
        payload["message"] = message
    return json.dumps(payload)
