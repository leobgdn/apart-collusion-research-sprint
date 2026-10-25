"""OpenAI-compatible logging proxy / mock server for the Colosseum Track B check.

Modes:
  --mock               answer locally with scripted tool calls (no GPU, no network)
  --upstream URL       forward to a real OpenAI-compatible server (e.g. local vLLM)

In both modes every request and response is appended to --log as one JSON line,
so the exact messages + tools sent to the model at every turn are captured.

Upstream mode also:
  * injects chat_template_kwargs={"enable_thinking": false} (Qwen3 thinking off),
  * caps in-flight upstream requests at --max-concurrency (default 4),
  * blocks while GPU_BUSY exists in any shared dir (Windows ~ and the WSL home).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# The server lives in WSL, so the other session may write its flags on either side.
SHARED_DIRS = [Path.home() / "collusion_shared", Path("//wsl.localhost/Ubuntu-24.04/home/leo/collusion_shared")]


def _gpu_busy() -> bool:
    return any((d / "GPU_BUSY").exists() for d in SHARED_DIRS)

ARGS: argparse.Namespace
SEM: threading.Semaphore
LOG_LOCK = threading.Lock()
COUNTER = {"n": 0}


def _log(record: dict) -> None:
    with LOG_LOCK:
        with open(ARGS.log, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _tool_names(payload: dict) -> list[str]:
    return [t.get("function", {}).get("name") for t in payload.get("tools") or []]


def _mock_response(payload: dict) -> dict:
    """Scripted reply: one tool call per turn, then plain text once a tool result is in context."""
    messages = payload.get("messages") or []
    rng = random.Random(len(json.dumps(messages)))
    names = _tool_names(payload)
    last_role = messages[-1].get("role") if messages else None
    message: dict = {"role": "assistant", "content": ""}
    finish = "stop"
    if last_role != "tool" and names:
        text = "\n".join(str(m.get("content") or "") for m in messages if m.get("role") in ("system", "user"))
        if "assign_task" in names:
            ids = sorted(set(re.findall(r"\b(ISSUE-\d+::\w+)", text)))
            task_id = rng.choice(ids) if ids else "skip"
            call = {"name": "assign_task", "arguments": json.dumps({"task_id": task_id})}
        else:
            bb_ids = sorted({int(x) for x in re.findall(r"[Bb]lackboard(?:_id)?\s*[#:=]?\s*(\d+)", text)}) or [0]
            call = {
                "name": "post_message",
                "arguments": json.dumps({"message": "mock: I plan to take a task.", "blackboard_id": rng.choice(bb_ids)}),
            }
        message["tool_calls"] = [{"id": f"call_{COUNTER['n']}", "type": "function", "function": call}]
        finish = "tool_calls"
    else:
        message["content"] = "mock: done."
    prompt_chars = len(json.dumps(messages))
    return {
        "id": f"mock-{COUNTER['n']}",
        "object": "chat.completion",
        "model": payload.get("model"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": prompt_chars // 4, "completion_tokens": 20, "total_tokens": prompt_chars // 4 + 20},
    }


def _forward(path: str, payload: dict | None, method: str) -> tuple[int, dict]:
    url = ARGS.upstream.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, {"error": e.read().decode(errors="replace")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            if ARGS.mock:
                return self._send(200, {"object": "list", "data": [{"id": ARGS.model, "object": "model"}]})
            status, body = _forward("/models", None, "GET")
            return self._send(status, body)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        COUNTER["n"] += 1
        req_id = COUNTER["n"]
        if not ARGS.mock:
            payload.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
        t0 = time.time()
        waited = 0.0
        if ARGS.mock:
            status, body = 200, _mock_response(payload)
        else:
            while _gpu_busy():
                time.sleep(5)
                waited += 5
            with SEM:
                status, body = _forward("/chat/completions", payload, "POST")
        latency = time.time() - t0 - waited
        msg = ((body.get("choices") or [{}])[0].get("message") or {}) if isinstance(body, dict) else {}
        content = msg.get("content") or ""
        _log(
            {
                "req_id": req_id,
                "t_start": t0,
                "latency_s": round(latency, 3),
                "gpu_busy_wait_s": waited,
                "status": status,
                "request": payload,
                "response": body,
                "thinking_detected": ("<think>" in content) or bool(msg.get("reasoning_content") or msg.get("reasoning")),
            }
        )
        self._send(status, body)


def main() -> None:
    global ARGS, SEM
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--mock", action="store_true")
    g.add_argument("--upstream")
    p.add_argument("--port", type=int, default=8031)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--max-concurrency", type=int, default=4)
    p.add_argument("--log", required=True)
    ARGS = p.parse_args()
    SEM = threading.Semaphore(min(ARGS.max_concurrency, 4))
    Path(ARGS.log).parent.mkdir(parents=True, exist_ok=True)
    print(f"proxy on 127.0.0.1:{ARGS.port} mode={'mock' if ARGS.mock else ARGS.upstream}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
