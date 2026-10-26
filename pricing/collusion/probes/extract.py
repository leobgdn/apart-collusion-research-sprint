"""Residual-stream activations for every logged turn.

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


# Only these fields are kept per turn; the full records (about 7 KB each) made
# the first full run hold several GB in memory.
_KEEP = ("episode_id", "condition_id", "period", "agent_index", "turn", "rendered_prompt")


def load_turns(run_id: str, limit: int | None, subset: str = "all"):
    from collusion.runner.logging_io import read_episode_summaries, read_turn_log

    done = {
        s["episode_id"]
        for s in read_episode_summaries(ROOT / "data" / "episodes" / run_id)
        if s.get("completed")
    }
    rows = []
    for r in read_turn_log(ROOT / "data" / "logs" / f"{run_id}.jsonl"):
        if r["episode_id"] not in done:
            continue
        if subset == "window" and not r.get("in_metric_window"):
            continue
        if subset == "rest" and r.get("in_metric_window"):
            continue
        slim = {k: r.get(k) for k in _KEEP}
        slim["server_prompt_tokens"] = (r.get("call_info") or {}).get("server_prompt_tokens")
        slim["response_text"] = r.get("response_text") or ""
        rows.append(slim)
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
    ap.add_argument("--subset", choices=["all", "window", "rest"], default="all",
                    help="window = scored last-30-round turns (extract first); rest = the others")
    args = ap.parse_args(argv)

    rows = load_turns(args.run_id, args.limit, args.subset)
    name = args.run_id if args.subset == "all" else f"{args.run_id}_{args.subset}"
    out_dir = ROOT / "data" / "activations" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    # No device_map (that needs `accelerate`): load, then move to the GPU.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa",
    ).to("cuda").eval()
    # Keep the caching allocator inside the card: past ~24 GB the WSL driver
    # silently spills to system RAM instead of failing (4x slower).
    torch.cuda.set_per_process_memory_fraction(0.85)
    n_layers = model.config.num_hidden_layers
    layers = sorted({max(1, round(f * n_layers)) for f in LAYER_FRACTIONS})
    d_model = model.config.hidden_size
    print(f"{len(rows)} turns, layers {layers} of {n_layers}, d_model {d_model}")

    # Hooks keep only the final-token vector of the saved layers. Asking for all
    # hidden states plus full-vocabulary logits filled the 24 GB card, and on
    # Windows/WSL the driver then spills to system RAM and runs ~20x slower.
    # Hooked output of decoder layer L-1 == hidden_states[L] (output of layer L).
    captured = {}

    def hook(L):
        def fn(_module, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Left padding puts every prompt's final token at position -1.
            captured[L] = h[:, -1, :].float().cpu().numpy()
        return fn

    for L in layers:
        model.model.layers[L - 1].register_forward_hook(hook(L))

    @torch.no_grad()
    def batch_acts(texts):
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        # Left padding: positions must count real tokens only, or every padded
        # prompt is read at shifted rotary positions.
        pos = (enc["attention_mask"].cumsum(1) - 1).clamp(min=0)
        captured.clear()
        model.model(**enc, position_ids=pos, use_cache=False)  # backbone only: no logits
        return {L: captured[L] for L in layers}, enc["attention_mask"].sum(1).tolist()

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
        "n_turns": len(rows), "subset": args.subset, "deterministic_first_batch": deterministic,
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

    # Resume: reuse existing arrays of the right shape. A row is done when it is
    # non-zero in every layer (fresh arrays are all zeros; a real residual-stream
    # vector never is).
    shape = (len(rows), d_model)
    paths = {L: out_dir / f"layer_{L}.npy" for L in layers}
    resume = all(p.exists() for p in paths.values()) and all(
        np.load(p, mmap_mode="r").shape == shape for p in paths.values())
    mm = {L: (np.load(paths[L], mmap_mode="r+") if resume else
              np.lib.format.open_memmap(paths[L], mode="w+", dtype=np.float16, shape=shape))
          for L in layers}
    # The tail check catches a row cut off mid-write when a run is killed.
    done = np.ones(len(rows), dtype=bool)
    for L in layers:
        a = np.asarray(mm[L])
        done &= (a != 0).any(axis=1) & (a[:, -16:] != 0).any(axis=1)
    print(f"resume={resume}: {int(done.sum())}/{len(rows)} rows already extracted")

    # Rows salvaged from an interrupted run are re-extracted on a sample and must
    # agree within the padding tolerance (batch composition differs, so exact
    # equality is not expected).
    salvage_rel = None
    if done.any():
        prev = np.flatnonzero(done)
        chk = rng.choice(prev, size=min(64, len(prev)), replace=False)
        salvage_rel = 0.0
        for start in range(0, len(chk), args.batch_size):
            idx = chk[start:start + args.batch_size]
            acts, _ = batch_acts([rows[i]["rendered_prompt"] for i in idx])
            for L in layers:
                old = np.asarray(mm[L][idx], dtype=np.float32)
                rel = np.linalg.norm(acts[L] - old, axis=1) / (np.linalg.norm(old, axis=1) + 1e-9)
                salvage_rel = max(salvage_rel, float(rel.max()))
        meta["salvaged_rows"] = int(done.sum())
        meta["salvage_check_rows"] = int(len(chk))
        meta["salvage_max_relative_diff"] = salvage_rel
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"salvage check on {len(chk)} rows: max relative diff {salvage_rel:.4f}")
        if salvage_rel > 0.05:
            raise SystemExit("salvaged rows disagree with re-extraction; delete the arrays and rerun")

    # Sort by length inside the pass to cut padding, writing rows back in index order.
    todo = np.flatnonzero(~done)
    order = todo[np.argsort([len(rows[i]["rendered_prompt"]) for i in todo], kind="stable")]
    for b, start in enumerate(range(0, len(order), args.batch_size)):
        idx = order[start:start + args.batch_size]
        acts, _ = batch_acts([rows[i]["rendered_prompt"] for i in idx])
        for L in layers:
            mm[L][idx] = acts[L].astype(np.float16)
        if b % 50 == 0:
            for L in layers:
                mm[L].flush()
            torch.cuda.empty_cache()
            print(f"  {int(done.sum()) + start + len(idx)}/{len(rows)}", flush=True)
    for L in layers:
        mm[L].flush()

    # Token counts from the tokenizer (equal to the attention-mask sums above).
    n_tok = []
    for start in range(0, len(rows), 512):
        enc = tok([r["rendered_prompt"] for r in rows[start:start + 512]], add_special_tokens=False)
        n_tok.extend(len(x) for x in enc["input_ids"])
    with (out_dir / "index.jsonl").open("w") as fh:
        for i, r in enumerate(rows):
            fh.write(json.dumps({
                "row": i, "episode_id": r["episode_id"], "condition_id": r["condition_id"],
                "period": r["period"], "agent_index": r["agent_index"], "turn": r["turn"],
                "n_prompt_tokens": n_tok[i],
                "server_prompt_tokens": r["server_prompt_tokens"],
            }) + "\n")
    bad = sum(1 for i, r in enumerate(rows)
              if r["server_prompt_tokens"] not in (None, n_tok[i]))
    zero = int(sum(int((~(np.asarray(mm[L]) != 0).any(axis=1)).sum()) for L in layers))
    meta["token_count_mismatches_vs_server"] = bad
    meta["all_zero_rows_after_extraction"] = zero
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done; token-count mismatches vs server: {bad}; all-zero rows: {zero}")
    return 0 if bad == 0 and zero == 0 and math.isfinite(mean_lp) else 1


if __name__ == "__main__":
    raise SystemExit(main())
