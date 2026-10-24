"""The LLM agent against a stub OpenAI-compatible server (no GPU, no model).

Checks what is sent (thinking off via chat_template_kwargs, sampling params, a
per-request seed that changes on retry) and what is logged (rendered prompt,
server vs local token counts, mismatch flag), end to end through run_sweep.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from collusion.agents.llm_agent import (
    ChatTemplate,
    LLMAgent,
    OpenAICompatClient,
    SamplingSettings,
    request_seed,
)
from collusion.envs.pricing.conditions import BASELINE
from collusion.envs.pricing.pricing_env import EnvConfig
from collusion.runner.logging_io import read_turn_log
from collusion.runner.replay import verify_round_trip
from collusion.runner.run_sweep import SweepConfig, run_sweep

MODEL = "Qwen/Qwen3-8B"


@pytest.fixture(scope="module")
def template():
    try:
        return ChatTemplate.for_model(MODEL)
    except Exception as exc:  # noqa: BLE001 - no cached tokenizer / no network
        pytest.skip(f"tokenizer unavailable: {exc!r}")


class _Stub:
    """Records requests; replies with a fixed price and a chosen token count."""

    def __init__(self, template, *, token_offset=0, first_reply_bad=False):
        self.template = template
        self.token_offset = token_offset
        self.first_reply_bad = first_reply_bad
        self.requests: list[dict] = []
        self.lock = threading.Lock()


def _serve(stub: _Stub):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with stub.lock:
                stub.requests.append(body)
                n = len(stub.requests)
            rendered = stub.template.render(
                body["messages"],
                enable_thinking=body["chat_template_kwargs"]["enable_thinking"],
            )
            tokens = stub.template.count(rendered) + stub.token_offset
            content = (
                "no json here"
                if stub.first_reply_bad and n == 1
                else json.dumps({"price": 1.7, "plan": "hold", "insight": "steady"})
            )
            payload = {
                "choices": [
                    {"message": {"content": content}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": tokens, "completion_tokens": 12},
            }
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def _config(tmp_path, params, run_id="llm_stub"):
    return SweepConfig(
        raw={},
        run_id=run_id,
        params=params,
        env_config=EnvConfig(total_periods=6),
        conditions=[BASELINE, BASELINE.replace(channel="private")],
        seeds_per_batch=1,
        n_batches=1,
        metric_window=5,
        label_threshold=None,
        max_parse_retries=2,
        max_parse_failure_rate=0.5,
        logs_dir=tmp_path / "logs",
        episodes_dir=tmp_path / "episodes",
        concurrency=2,
    )


def _factory(endpoint, template):
    client = OpenAICompatClient(endpoint, model=MODEL)
    sampling = SamplingSettings()

    def build(condition, seed, benchmarks):
        return [
            LLMAgent(client=client, episode_seed=seed, agent_index=i, sampling=sampling,
                     template=template, agent_id=f"seller_{i}", model_name=MODEL)
            for i in range(2)
        ]

    return build


def test_requests_and_logs_through_a_sweep(tmp_path, params, template):
    stub = _Stub(template)
    server, endpoint = _serve(stub)
    try:
        config = _config(tmp_path, params)
        report = run_sweep(config, _factory(endpoint, template), verbose=False,
                           parallel_agents=True)
    finally:
        server.shutdown()
    assert report.completed == 2 and report.stop_reason is None

    body = stub.requests[0]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["temperature"] == 0.7 and body["top_p"] == 0.9
    assert body["max_tokens"] == 150 and isinstance(body["seed"], int)

    rows = list(read_turn_log(config.log_path))
    assert len(rows) == 2 * 6 * 2
    for row in rows:
        info = row["call_info"]
        assert info["server_prompt_tokens"] == info["local_prompt_tokens"]
        assert info["prompt_token_mismatch"] is False
        assert row["rendered_prompt"].startswith("<|im_start|>system\n")
        assert row["rendered_prompt"].endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
        assert row["prompt_messages"][1]["content"] in row["rendered_prompt"]
        assert row["tokens_out"] == 12
    assert not verify_round_trip(rows, params=params,
                                 episode_periods={r["episode_id"]: 6 for r in rows})


def test_a_token_mismatch_stops_the_sweep(tmp_path, params, template):
    stub = _Stub(template, token_offset=3)
    server, endpoint = _serve(stub)
    try:
        report = run_sweep(_config(tmp_path, params, "mm"), _factory(endpoint, template),
                           verbose=False, parallel_agents=True)
    finally:
        server.shutdown()
    assert report.stop_reason and "prompt" in report.stop_reason


def test_retry_uses_a_fresh_seed(tmp_path, params, template):
    stub = _Stub(template, first_reply_bad=True)
    server, endpoint = _serve(stub)
    try:
        client = OpenAICompatClient(endpoint, model=MODEL)
        agent = LLMAgent(client=client, episode_seed=5, agent_index=0, template=template)
        from collusion.runner.run_episode import run_episode

        other = LLMAgent(client=client, episode_seed=5, agent_index=1, template=template)
        res = run_episode(condition=BASELINE, seed=5, agents=[agent, other],
                          env_config=EnvConfig(total_periods=1))
    finally:
        server.shutdown()
    seeds = [r["seed"] for r in stub.requests]
    assert len(seeds) == 3  # one failed + retry for seller 0, one for seller 1
    assert seeds[0] == request_seed(5, 1, 0, 0) and seeds[1] == request_seed(5, 1, 0, 1)
    assert res.turns[0].retries == 1 and res.summary.parse_failures == 0
