"""LLM agent: Qwen3-8B behind a local OpenAI-compatible (vLLM) server.

No external APIs (Phase 1 override 1). Every request goes to the local server.

Per call we log what activation extraction will later need to reproduce the
model's input exactly:

* `rendered_prompt` -- the full prompt string after the chat template, rendered
  locally with the model's own tokenizer and the same `enable_thinking=False`
  flag sent to the server;
* `call_info.server_prompt_tokens` -- the prompt token count the server reports,
  and `local_prompt_tokens` -- the count of our locally rendered string. A
  mismatch means the logged prompt is not what the server saw (Phase 1 Section
  8: stop the line), and is flagged per turn as `prompt_token_mismatch`.

Thinking is disabled through `chat_template_kwargs: {"enable_thinking": false}`
in the request body (vLLM forwards it to the chat template). Whether that works
is verified on real outputs by `serve/check_server.py`, and every reply is
scanned for reasoning blocks by the parser.

Seeds: each request's sampling seed is derived from (episode seed, period,
agent index, attempt) so a retry draws a fresh sample while staying reproducible
in principle. Batched serving can still introduce nondeterminism (batch
composition changes floating-point reduction order), so identical seeds do not
guarantee identical text; this is recorded in DECISIONS.md.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import requests

from collusion.envs.pricing.conditions import Condition
from collusion.envs.pricing.prompts import RenderedPrompt
from collusion.envs.pricing.state import Observation
from collusion.runner.run_episode import AgentReply

DEFAULT_MODEL = "Qwen/Qwen3-8B"
SHARED_DIR = Path.home() / "collusion_shared"
ENDPOINT_FILE = SHARED_DIR / "endpoint.txt"
DEFAULT_ENDPOINT = "http://localhost:8000/v1"


def resolve_endpoint(explicit: str | None = None) -> str:
    """Explicit arg, then $COLLUSION_ENDPOINT, then the shared file, then default."""
    if explicit:
        return explicit.rstrip("/")
    env = os.environ.get("COLLUSION_ENDPOINT")
    if env:
        return env.rstrip("/")
    if ENDPOINT_FILE.exists():
        text = ENDPOINT_FILE.read_text(encoding="utf-8").strip()
        if text:
            return text.rstrip("/")
    return DEFAULT_ENDPOINT


@dataclass(frozen=True)
class SamplingSettings:
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 150
    enable_thinking: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
        }


class ServerStats:
    """Thread-safe running totals across all requests (throughput measurement)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with getattr(self, "_lock", threading.Lock()):
            self.requests = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.http_retries = 0
            self.latencies: list[float] = []
            self.started = time.time()

    def record(self, prompt_tokens: int, completion_tokens: int, latency: float = 0.0) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.latencies.append(latency)

    def record_retry(self) -> None:
        with self._lock:
            self.http_retries += 1

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            elapsed = max(time.time() - self.started, 1e-9)
            return {
                "elapsed_s": elapsed,
                "requests": self.requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "prefill_tok_per_s": self.prompt_tokens / elapsed,
                "decode_tok_per_s": self.completion_tokens / elapsed,
                "requests_per_s": self.requests / elapsed,
                "http_retries": self.http_retries,
                "mean_latency_s": float(np.mean(self.latencies)) if self.latencies else 0.0,
                "p90_latency_s": float(np.percentile(self.latencies, 90)) if self.latencies else 0.0,
            }


STATS = ServerStats()


class ChatTemplate:
    """Local rendering of the model's chat template (tokenizer only, no weights)."""

    _cache: dict[str, "ChatTemplate"] = {}
    _cache_lock = threading.Lock()

    def __init__(self, model: str):
        from transformers import AutoTokenizer  # heavy import, only when needed

        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(model)

    @classmethod
    def for_model(cls, model: str) -> "ChatTemplate":
        with cls._cache_lock:
            if model not in cls._cache:
                cls._cache[model] = cls(model)
            return cls._cache[model]

    def render(self, messages: list[dict[str, str]], *, enable_thinking: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def count(self, rendered: str) -> int:
        return len(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])


class ServerError(RuntimeError):
    """The server failed after all HTTP retries."""


class OpenAICompatClient:
    """Minimal, thread-safe chat-completions client for the local vLLM server."""

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        timeout_s: float = 120.0,
        max_http_retries: int = 5,
        stats: ServerStats = STATS,
    ):
        self.endpoint = resolve_endpoint(endpoint)
        self.model = model
        self.timeout_s = timeout_s
        self.max_http_retries = max_http_retries
        self.stats = stats
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
        return self._local.session

    def chat(
        self,
        messages: list[dict[str, str]],
        sampling: SamplingSettings,
        *,
        seed: int | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": sampling.temperature,
            "top_p": sampling.top_p,
            "max_tokens": sampling.max_tokens,
            "chat_template_kwargs": {"enable_thinking": sampling.enable_thinking},
        }
        if seed is not None:
            body["seed"] = int(seed)
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_http_retries + 1):
            started = time.time()
            try:
                resp = self._session().post(
                    f"{self.endpoint}/chat/completions",
                    json=body,
                    timeout=self.timeout_s,
                )
                if resp.status_code >= 500 or resp.status_code == 429:
                    raise ServerError(f"HTTP {resp.status_code}: {resp.text[:300]}")
                if resp.status_code != 200:
                    # A 4xx other than 429 is our bug, not a transient fault.
                    raise ValueError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                payload = resp.json()
                latency = time.time() - started
                choice = payload["choices"][0]
                usage = payload.get("usage") or {}
                self.stats.record(
                    int(usage.get("prompt_tokens", 0)),
                    int(usage.get("completion_tokens", 0)),
                    latency,
                )
                return {
                    "text": choice["message"].get("content") or "",
                    "reasoning_content": choice["message"].get("reasoning_content"),
                    "finish_reason": choice.get("finish_reason"),
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "latency_s": latency,
                    "http_attempts": attempt + 1,
                }
            except (requests.RequestException, ServerError) as exc:
                last_exc = exc
                self.stats.record_retry()
                if attempt == self.max_http_retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
        raise ServerError(
            f"request failed after {self.max_http_retries + 1} attempts: {last_exc!r}"
        )


def request_seed(episode_seed: int, period: int, agent_index: int, attempt: int) -> int:
    """Per-request sampling seed derived from (episode seed, turn, agent, attempt)."""
    state = np.random.SeedSequence([episode_seed, period, agent_index, attempt])
    return int(state.generate_state(1)[0] & 0x7FFFFFFF)


@dataclass
class LLMAgent:
    """One seller played by the local model."""

    client: OpenAICompatClient
    episode_seed: int
    agent_index: int
    sampling: SamplingSettings = field(default_factory=SamplingSettings)
    template: ChatTemplate | None = None
    agent_id: str = "seller"
    model_name: str = DEFAULT_MODEL

    def __post_init__(self) -> None:
        self._period = -1
        self._attempt = 0

    def act(
        self,
        obs: Observation,
        condition: Condition,
        rng: np.random.Generator,
        rendered: RenderedPrompt | None = None,
    ) -> AgentReply:
        if rendered is None:
            raise ValueError("LLMAgent needs the rendered prompt")
        if obs.period != self._period:
            self._period, self._attempt = obs.period, 0
        else:
            self._attempt += 1
        seed = request_seed(self.episode_seed, obs.period, self.agent_index, self._attempt)

        rendered_prompt = ""
        local_tokens = None
        if self.template is not None:
            rendered_prompt = self.template.render(
                rendered.messages, enable_thinking=self.sampling.enable_thinking
            )
            local_tokens = self.template.count(rendered_prompt)

        result = self.client.chat(rendered.messages, self.sampling, seed=seed)
        text, reasoning_text = result["text"], None
        if self.sampling.enable_thinking:
            # Thinking deliberately on (Gate 2 diagnostic): the reasoning is
            # logged separately and the action is parsed from what follows it.
            # A reply cut off mid-thought has no action and fails to parse.
            head, sep, tail = text.partition("</think>")
            if sep:
                reasoning_text = head.replace("<think>", "").strip()
                text = tail.strip()
            else:
                reasoning_text, text = text, ""
        call_info = {
            "request_seed": seed,
            "attempt": self._attempt,
            "endpoint": self.client.endpoint,
            "server_prompt_tokens": result["prompt_tokens"],
            "local_prompt_tokens": local_tokens,
            "prompt_token_mismatch": (
                local_tokens is not None and local_tokens != result["prompt_tokens"]
            ),
            "finish_reason": result["finish_reason"],
            "latency_s": round(result["latency_s"], 4),
            "http_attempts": result["http_attempts"],
            "reasoning_content": result["reasoning_content"],
            "reasoning_text": reasoning_text,
        }
        return AgentReply(
            text=text,
            tokens_in=result["prompt_tokens"],
            tokens_out=result["completion_tokens"],
            cost_usd=0.0,
            model_settings={**self.sampling.as_dict(), "seed": seed},
            model=self.model_name,
            rendered_prompt=rendered_prompt,
            call_info=call_info,
        )
