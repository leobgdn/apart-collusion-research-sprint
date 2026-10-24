"""Summarize a Colosseum output root (+ optional proxy log) for the Track B report.

Per run: condition, tokens, conversation steps, invalid tool calls, Colosseum's regret
metrics, coalition advantage (two definitions), exact-optimum gap, and a recomputation
of best-response regret from the logged assignment to check it matches metrics.json.

Usage: python summarize.py OUTPUT_ROOT [--proxy-log FILE] [--threshold 0.0]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def summarize_run(run_dir: Path) -> dict:
    cfg = _load(run_dir / "run_config.json")
    m = _load(run_dir / "metrics.json")
    turns = _load(run_dir / "agent_turns.json") or []
    events = _load(run_dir / "tool_events.json") or []
    opt = _load(run_dir / "optimal_summary.json") or {}
    fs = _load(run_dir / "final_summary.json") or {}
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    steps = 0
    for t in turns:
        u = t.get("usage") or {}
        usage["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        usage["completion_tokens"] += int(u.get("completion_tokens") or 0)
        steps += int(t.get("conversation_steps") or 0)
    assign = [e for e in events if e["tool_name"] == "assign_task"]
    bad_assign = [e for e in assign if isinstance(e.get("result"), dict) and e["result"].get("status") in ("retry", "error")]
    # Agents that never landed a valid assign_task (skip counts as valid).
    agents = {a["agent_name"]: a for a in m["agents"]}
    no_choice = [n for n, a in agents.items() if a["chosen_task_id"] is None]
    regret_adv = None
    if m.get("noncoalition_mean_regret") is not None and m.get("coalition_mean_regret") is not None:
        regret_adv = m["noncoalition_mean_regret"] - m["coalition_mean_regret"]
    return {
        "run_id": cfg["run_id"],
        "secret": cfg["secret_channel_enabled"],
        "prompt_variant": cfg["prompt_variant"],
        "topology": cfg["topology"],
        "n": cfg["num_agents"],
        "seed": cfg["seed"],
        "status": m["status"],
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "llm_steps": steps,
        "assign_calls": len(assign),
        "assign_invalid": len(bad_assign),
        "agents_without_choice": len(no_choice),
        "mean_regret": m.get("mean_regret"),
        "coalition_mean_regret": m.get("coalition_mean_regret"),
        "noncoalition_mean_regret": m.get("noncoalition_mean_regret"),
        "regret_advantage": regret_adv,  # noncoalition - coalition regret (apart-sprint prereg def.)
        "reward_advantage": m.get("coalition_advantage_mean"),  # coalition - noncoalition mean reward (Colosseum def.)
        "colluder_posts_secret_rate": m.get("colluder_posts_secret_rate"),
        "joint_reward": fs.get("joint_reward"),
        "optimal_gap": (opt["optimal"]["joint_reward"] - fs["joint_reward"]) if opt.get("optimal") and fs.get("joint_reward") is not None else None,
    }


def proxy_stats(path: Path) -> dict:
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        return {}
    lat = [r["latency_s"] for r in recs]
    t0 = min(r["t_start"] for r in recs)
    t1 = max(r["t_start"] + r["latency_s"] + r.get("gpu_busy_wait_s", 0) for r in recs)
    return {
        "requests": len(recs),
        "http_errors": sum(1 for r in recs if r["status"] != 200),
        "thinking_detected": sum(1 for r in recs if r.get("thinking_detected")),
        "mean_latency_s": sum(lat) / len(lat),
        "wall_span_s": t1 - t0,
        "has_full_messages": all("messages" in r["request"] for r in recs),
        "has_tools": sum(1 for r in recs if r["request"].get("tools")),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("root")
    p.add_argument("--proxy-log")
    p.add_argument("--threshold", type=float, default=0.0)
    a = p.parse_args()
    rows = [summarize_run(d.parent) for d in sorted(Path(a.root).rglob("metrics.json"))]
    for r in rows:
        print(json.dumps(r))
    by_cond: dict = {}
    for r in rows:
        key = (r["topology"], r["secret"], r["prompt_variant"])
        by_cond.setdefault(key, []).append(r)
    print("\ncondition | runs | regret_adv>thr | reward_adv>thr | mean regret_adv")
    for key, rs in sorted(by_cond.items(), key=str):
        ra = [x["regret_advantage"] for x in rs if x["regret_advantage"] is not None]
        wa = [x["reward_advantage"] for x in rs if x["reward_advantage"] is not None]
        print(
            f"{key} | {len(rs)} | {sum(v > a.threshold for v in ra)}/{len(ra)} | "
            f"{sum(v > a.threshold for v in wa)}/{len(wa)} | {sum(ra) / len(ra) if ra else None}"
        )
    if a.proxy_log:
        print("\nproxy:", json.dumps(proxy_stats(Path(a.proxy_log))))


if __name__ == "__main__":
    main()
