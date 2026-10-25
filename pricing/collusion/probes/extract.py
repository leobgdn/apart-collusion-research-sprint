"""Residual-stream activations for every logged turn (SPEC Section 10).

Run INSIDE WSL with the vLLM venv (it has CUDA torch + transformers), after the
vLLM server is stopped so the GPU is free:

    source ~/vllm-env/bin/activate
    cd /mnt/c/Users/leobo/OneDrive/Documents/GitHub/apart-collusion-research-sprint/pricing
    python -m collusion.probes.extract --run-id main_v1 [--batch-size 16] [--limit N]

For each turn the logged `rendered_prompt` (the exact chat-templated string the
server saw; token counts were checked against the server for every call) is fed
through Qwen3-8B with a single forward pass. The residual stream (hidden state
output of the layer) at the final prompt token is saved for layers at 30-80% of
depth, as float16.

Outputs in data/activations/<run_id>/:
  layer_<L>.npy   float16, shape (n_turns, d_model), row i <-> index row i
  index.jsonl     one line per row: episode_id, condition_id, period,
                  agent_index, turn, n_prompt_tokens
  meta.json       model, revision, layers, dtype, sanity-check results

Sanity checks (written to meta.json; extraction stops if they fail):
  (a) teacher-forced mean log-probability of the logged response under the
      model, on a sample of turns. A very low value means a template or
      tokenisation mismatch.
  (b) determinism: the first batch is extracted twice and must match exactly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL = "Qwen/Qwen3-8B"
REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
LAYER_FRACTIONS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# Mean per-token log-prob below this on the logged replies means the prompt the
# model sees here is not the one it generated from.
MIN_PLAUSIBLE_LOGPROB = -1.5


def load_turns(run_id: str, limit: int | None):
    from collusion.runner.logging_io import read_episode_summaries, read_turn_log

    done = {
        s["episode_id"]
        for s in read_episode_summaries(ROOT / "data" / "episodes" / run_id)
        if s.get("completed")
    }
    rows = [r for r in read_turn_log(ROOT / "data" / "logs" / f"{run_id}.jsonl")
            if r["episode_id"] in done]
    rows.sort(key=lambda r: (r["episode_id"], r["period"], r["agent_index"]))
    missing = [r for r in rows if not r.get("rendered_prompt")]
    if missing:
        raise SystemExit(f"{len(missing)} turns have no rendered_prompt; cannot extract")
    return rows[:limit] if limit else rows


def main(argv: list[str] | None = None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="main_v1")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--logprob-sample", type=int, default=64)
    args = ap.parse_args(argv)

    rows = load_turns(args.run_id, args.limit)
    out_dir = ROOT / "data" / "activations" / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa",
    ).eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({max(1, round(f * n_layers)) for f in LAYER_FRACTIONS})
    d_model = model.config.hidden_size
    print(f"{len(rows)} turns, layers {layers} of {n_layers}, d_model {d_model}")

    @torch.no_grad()
    def batch_acts(texts):
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        # Left padding: positions must count real tokens only, or every padded
        # prompt is read at shifted rotary positions.
        pos = (enc["attention_mask"].cumsum(1) - 1).clamp(min=0)
        out = model(**enc, position_ids=pos, output_hidden_states=True, use_cache=False)
        # hidden_states[0] is the embeddings; [L] is the output of layer L.
        # Left padding puts every prompt's final token at position -1.
        return {L: out.hidden_states[L][:, -1, :].float().cpu().numpy() for L in layers}, \
            enc["attention_mask"].sum(1).tolist()

    # (b) determinism on the first batch
    first = [r["rendered_prompt"] for r in rows[: args.batch_size]]
    a1, _ = batch_acts(first)
    a2, _ = batch_acts(first)
    deterministic = all(np.array_equal(a1[L], a2[L]) for L in layers)
    max_diff = max(float(np.abs(a1[L] - a2[L]).max()) for L in layers)
    # (c) padding invariance: each prompt alone vs inside the padded batch.
    pad_rel = 0.0
    for k in range(min(4, len(first))):
        solo, _ = batch_acts([first[k]])
        for L in layers:
            rel = np.linalg.norm(solo[L][0] - a1[L][k]) / (np.linalg.norm(solo[L][0]) + 1e-9)
            pad_rel = max(pad_rel, float(rel))

    # (a) teacher-forced log-prob of logged replies on a sample
    rng = np.random.default_rng(0)
    sample = rng.choice(len(rows), size=min(args.logprob_sample, len(rows)), replace=False)
    lps = []
    with torch.no_grad():
        for i in sample:
            r = rows[int(i)]
            p_ids = tok(r["rendered_prompt"], add_special_tokens=False)["input_ids"]
            c_ids = tok(r["response_text"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
            ids = torch.tensor([p_ids + c_ids], device="cuda")
            logits = model(ids, use_cache=False).logits[0, len(p_ids) - 1:-1].float()
            lp = torch.log_softmax(logits, -1).gather(1, ids[0, len(p_ids):, None]).squeeze(1)
            lps.append(float(lp.mean()))
    mean_lp = float(np.mean(lps))
    meta = {
        "model": MODEL, "revision": REVISION, "layers": layers, "n_layers": n_layers,
        "layer_fractions": LAYER_FRACTIONS, "dtype": "float16", "position": "final prompt token",
        "n_turns": len(rows), "deterministic_first_batch": deterministic,
        "determinism_max_abs_diff": max_diff,
        "padding_max_relative_diff": pad_rel,
        "teacher_forced_mean_logprob": mean_lp, "logprob_sample": len(lps),
        "logprob_min_plausible": MIN_PLAUSIBLE_LOGPROB,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    if mean_lp < MIN_PLAUSIBLE_LOGPROB:
        raise SystemExit("teacher-forced log-prob implausibly low: template/tokenisation mismatch")
    if pad_rel > 0.05:
        raise SystemExit(f"padded-batch activations differ from solo by {pad_rel:.3f} (relative)")

    mm = {L: np.lib.format.open_memmap(out_dir / f"layer_{L}.npy", mode="w+",
                                       dtype=np.float16, shape=(len(rows), d_model))
          for L in layers}
    # Sort by length inside the pass to cut padding, writing rows back in index order.
    order = np.argsort([len(r["rendered_prompt"]) for r in rows])
    n_tok = [0] * len(rows)
    for start in range(0, len(rows), args.batch_size):
        idx = order[start:start + args.batch_size]
        acts, lens = batch_acts([rows[i]["rendered_prompt"] for i in idx])
        for L in layers:
            mm[L][idx] = acts[L].astype(np.float16)
        for i, n in zip(idx, lens):
            n_tok[i] = int(n)
        if (start // args.batch_size) % 200 == 0:
            print(f"  {start + len(idx)}/{len(rows)}")
    for L in layers:
        mm[L].flush()
    with (out_dir / "index.jsonl").open("w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({
                "row": i, "episode_id": r["episode_id"], "condition_id": r["condition_id"],
                "period": r["period"], "agent_index": r["agent_index"], "turn": r["turn"],
                "n_prompt_tokens": n_tok[i],
                "server_prompt_tokens": r["call_info"].get("server_prompt_tokens"),
            }) + "\n")
    bad = sum(1 for i, r in enumerate(rows)
              if r["call_info"].get("server_prompt_tokens") not in (None, n_tok[i]))
    meta["token_count_mismatches_vs_server"] = bad
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done; token-count mismatches vs server: {bad}")
    return 0 if bad == 0 and math.isfinite(mean_lp) else 1


if __name__ == "__main__":
    raise SystemExit(main())
