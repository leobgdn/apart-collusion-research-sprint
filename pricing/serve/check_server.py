"""Check the shared model server, verify thinking is off, publish the endpoint.

    python -m serve.check_server [--endpoint http://localhost:8000/v1]

1. Finds an OpenAI-compatible endpoint serving Qwen/Qwen3-8B (reuses one that is
   already up, per Track A addendum Section 1).
2. Verifies the thinking-off mechanism on real outputs: real pilot prompts sent
   with `chat_template_kwargs: {"enable_thinking": false}` must produce no
   reasoning block, AND a control request with thinking left on must produce
   one -- otherwise the check could pass vacuously (e.g. if the server ignored
   the kwarg and the model happened not to think).
3. Verifies the server's prompt token count equals the token count of our
   locally rendered chat template, so the logged `rendered_prompt` is what the
   server actually saw.
4. Writes the endpoint URL to ~/collusion_shared/endpoint.txt and the results
   to reports/serving_check.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import requests

from collusion.agents.llm_agent import (
    DEFAULT_MODEL,
    ENDPOINT_FILE,
    ChatTemplate,
    OpenAICompatClient,
    SamplingSettings,
    resolve_endpoint,
)
from collusion.agents.parsing import has_reasoning_block, parse_response
from collusion.agents.scripted_agents import make_scripted_agent
from collusion.envs.pricing.conditions import pilot_design
from collusion.envs.pricing.economics import PricingParams, compute_benchmarks
from collusion.envs.pricing.pricing_env import EnvConfig
from collusion.runner.run_episode import run_episode

ROOT = Path(__file__).resolve().parents[1]


def sample_prompts(n_per_cell: int = 2) -> list[tuple[str, list[dict]]]:
    """Real pilot prompts: drive a short scripted episode per cell, take turns."""
    params = PricingParams()
    benchmarks = compute_benchmarks(params)
    out = []
    for condition in pilot_design():
        agents = [
            make_scripted_agent("colluding", benchmarks, noise_sd=0.05, agent_id=f"s{i}")
            for i in range(2)
        ]
        res = run_episode(
            condition=condition, seed=1, agents=agents, env_config=EnvConfig(total_periods=12)
        )
        for turn in (res.turns[0], res.turns[-1])[:n_per_cell]:
            out.append((condition.condition_id, turn.prompt_messages))
    return out


def gpu_memory() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
        ).stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"unavailable: {exc!r}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--endpoint", default=None)
    args = ap.parse_args(argv)
    endpoint = resolve_endpoint(args.endpoint)

    try:
        models = requests.get(f"{endpoint}/models", timeout=10).json()
    except Exception as exc:  # noqa: BLE001
        print(f"No server at {endpoint}: {exc!r}", file=sys.stderr)
        return 1
    served = [m["id"] for m in models.get("data", [])]
    print(f"server at {endpoint} serves {served}")
    if DEFAULT_MODEL not in served:
        print(f"{DEFAULT_MODEL} is not served there", file=sys.stderr)
        return 1

    client = OpenAICompatClient(endpoint, model=DEFAULT_MODEL)
    template = ChatTemplate.for_model(DEFAULT_MODEL)
    off = SamplingSettings(temperature=0.7, top_p=0.9, max_tokens=150, enable_thinking=False)
    on = SamplingSettings(temperature=0.7, top_p=0.9, max_tokens=400, enable_thinking=True)

    results = {"endpoint": endpoint, "model": DEFAULT_MODEL, "gpu_memory": gpu_memory()}
    rows = []
    for k, (cid, messages) in enumerate(sample_prompts()):
        r = client.chat(messages, off, seed=1000 + k)
        rendered = template.render(messages, enable_thinking=False)
        local = template.count(rendered)
        parsed = parse_response(r["text"], expect_message="channel=private" in cid)
        rows.append(
            {
                "condition_id": cid,
                "reasoning_block": has_reasoning_block(r["text"])
                or bool(r["reasoning_content"]),
                "server_prompt_tokens": r["prompt_tokens"],
                "local_prompt_tokens": local,
                "token_match": r["prompt_tokens"] == local,
                "parsed_ok": parsed.ok,
                "finish_reason": r["finish_reason"],
                "completion_tokens": r["completion_tokens"],
                "text": r["text"],
            }
        )
    control = client.chat(sample_prompts(1)[0][1], on, seed=7)
    control_thinks = has_reasoning_block(control["text"]) or bool(control["reasoning_content"])

    results["thinking_off_samples"] = rows
    results["thinking_on_control_shows_reasoning"] = control_thinks
    results["thinking_on_control_text"] = control["text"][:600]
    n_reason = sum(r["reasoning_block"] for r in rows)
    n_mismatch = sum(not r["token_match"] for r in rows)
    n_parse = sum(r["parsed_ok"] for r in rows)
    results["summary"] = {
        "samples": len(rows),
        "reasoning_blocks_with_thinking_off": n_reason,
        "token_mismatches": n_mismatch,
        "parsed_ok": n_parse,
    }

    print(json.dumps(results["summary"], indent=2))
    print(f"control (thinking on) shows reasoning: {control_thinks}")
    for r in rows[:3]:
        print(f"  [{r['condition_id']}] {r['text'][:200]!r}")
    print(f"GPU memory: {results['gpu_memory']}")

    out = ROOT / "reports" / "serving_check.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    ok = n_reason == 0 and n_mismatch == 0 and control_thinks
    if ok:
        ENDPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
        ENDPOINT_FILE.write_text(endpoint + "\n", encoding="utf-8")
        print(f"PASS. Endpoint written to {ENDPOINT_FILE}")
    else:
        print("FAIL: see reports/serving_check.json", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
