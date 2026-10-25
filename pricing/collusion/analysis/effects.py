"""Confirmatory effects (prereg "Planned analyses" 1-3), from the saved logs only.

    python -m collusion.analysis.effects --run-id main_v1

* Unit of analysis: the episode. Only complete balanced batches; below-cost and
  over-priced episodes excluded (counts reported per cell).
* H1-H3: main effect of each factor on Delta over the 8 crossed cells, as the
  difference in means between levels (in the balanced design this equals the
  OLS main-effect coefficient x 1). p-value: 10,000 permutations of that
  factor's labels within strata of the other two factors. 95% CI: bootstrap of
  episodes within cell. Holm correction across H1-H3.
* OLS with the three main effects (HC3 SEs) as a cross-check; logistic
  regression on the binary label at 0.3 (primary) and 0.15 / 0.5 (sensitivity).
* Placebo vs baseline: difference in Delta, bootstrap CI, permutation p.

Writes results/effects.md and results/effects.json.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from collusion.envs.pricing.conditions import BASELINE, pilot_design
from collusion.runner.logging_io import read_episode_summaries
from collusion.runner.run_sweep import complete_batches

ROOT = Path(__file__).resolve().parents[2]
N_PERM = 10_000
N_BOOT = 10_000

# Hypothesis: factor, level expected HIGHER, level expected LOWER.
HYPOTHESES = {
    "H1": ("channel", "private", "none"),
    "H2": ("history_window", "long", "short"),
    "H3": ("horizon", "long_run", "myopic"),
}


def load(run_id: str, seeds_per_batch: int) -> tuple[pd.DataFrame, dict]:
    rows = [s for s in read_episode_summaries(ROOT / "data" / "episodes" / run_id)
            if s.get("completed")]
    cells = pilot_design()
    batches = complete_batches(rows, cells, seeds_per_batch)
    keep_seeds = {s for b in batches for s in range(b * seeds_per_batch + 1,
                                                     (b + 1) * seeds_per_batch + 1)}
    df = pd.DataFrame([{
        "episode_id": s["episode_id"], "condition_id": s["condition_id"], "seed": s["seed"],
        **s["factors"], "delta": s["action_collusion_score"], "delta_p": s["price_index"],
        "category": s["category"], "pathological": s["is_pathological"],
        "joint_reward": s["joint_reward"],
        "consumer_surplus": s["action_metrics"].get("avg_consumer_surplus"),
    } for s in rows])
    info = {"episodes_completed": len(df), "complete_batches": batches,
            "n_complete_batches": len(batches)}
    df = df[df.seed.isin(keep_seeds)]
    info["episodes_in_complete_batches"] = len(df)
    info["pathological_by_cell"] = {
        k: int(v) for k, v in df[df.pathological].groupby("condition_id").size().items()}
    df = df[~df.pathological].copy()
    info["episodes_analysed"] = len(df)
    return df, info


def _diff(df, factor, hi, lo):
    return df.loc[df[factor] == hi, "delta"].mean() - df.loc[df[factor] == lo, "delta"].mean()


def main_effect(df: pd.DataFrame, factor: str, hi: str, lo: str, rng) -> dict:
    others = [f for f in ("channel", "history_window", "horizon") if f != factor]
    obs = _diff(df, factor, hi, lo)
    strata = df.groupby(others).indices
    labels = df[factor].to_numpy()
    y = df["delta"].to_numpy()
    count = 0
    for _ in range(N_PERM):
        perm = labels.copy()
        for idx in strata.values():
            perm[idx] = rng.permutation(perm[idx])
        d = y[perm == hi].mean() - y[perm == lo].mean()
        count += abs(d) >= abs(obs) - 1e-12
    p = (count + 1) / (N_PERM + 1)
    cells = df.groupby("condition_id").indices
    boots = []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(v, len(v)) for v in cells.values()])
        boots.append(_diff(df.iloc[idx], factor, hi, lo))
    pooled_sd = df.groupby("condition_id")["delta"].std(ddof=1).pow(2).mean() ** 0.5
    return {"factor": factor, "contrast": f"{hi} - {lo}", "estimate": float(obs),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "p_perm": float(p), "cohens_d": float(obs / pooled_sd) if pooled_sd else None,
            "n_hi": int((df[factor] == hi).sum()), "n_lo": int((df[factor] == lo).sum())}


def holm(ps: dict[str, float]) -> dict[str, float]:
    order = sorted(ps, key=ps.get)
    out, running = {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * ps[k]))
        out[k] = running
    return out


def main(argv: list[str] | None = None) -> int:
    import statsmodels.formula.api as smf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="main_v1")
    ap.add_argument("--seeds-per-batch", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--sensitivity", type=float, nargs="*", default=[0.15, 0.5])
    args = ap.parse_args(argv)
    rng = np.random.default_rng(20261024)

    df, info = load(args.run_id, args.seeds_per_batch)
    crossed = df[df.framing == "standard"]
    res = {"run_id": args.run_id, "data": info, "hypotheses": {}}
    for h, (f, hi, lo) in HYPOTHESES.items():
        res["hypotheses"][h] = main_effect(crossed, f, hi, lo, rng)
    adj = holm({h: r["p_perm"] for h, r in res["hypotheses"].items()})
    for h in res["hypotheses"]:
        res["hypotheses"][h]["p_holm"] = adj[h]

    ols = smf.ols("delta ~ C(channel, Treatment('none')) + C(history_window, Treatment('short'))"
                  " + C(horizon, Treatment('myopic'))", data=crossed).fit(cov_type="HC3")
    res["ols"] = {k: {"coef": float(ols.params[k]), "se_hc3": float(ols.bse[k]),
                      "p": float(ols.pvalues[k])} for k in ols.params.index}
    res["logit"] = {}
    for t in [args.threshold, *args.sensitivity]:
        d = crossed.assign(label=(crossed.delta >= t).astype(int))
        entry = {"positive_share": float(d.label.mean())}
        try:
            fit = smf.logit("label ~ C(channel, Treatment('none')) + C(history_window, "
                            "Treatment('short')) + C(horizon, Treatment('myopic'))",
                            data=d).fit(disp=0, warn_convergence=False)
            entry["converged"] = bool(fit.mle_retvals.get("converged", False))
            if not entry["converged"]:
                entry["error"] = "did not converge (likely separation); coefficients not interpretable"
            entry.update({k: {"coef": float(fit.params[k]), "p": float(fit.pvalues[k])}
                          for k in fit.params.index})
        except Exception as exc:  # noqa: BLE001 -- e.g. perfect separation; reported
            entry["error"] = repr(exc)
        res["logit"][str(t)] = entry

    base = df[df.condition_id == BASELINE.condition_id]
    plc = df[df.condition_id == BASELINE.replace(framing="placebo").condition_id]
    if len(base) and len(plc):
        both = pd.concat([base, plc])
        obs = plc.delta.mean() - base.delta.mean()
        lab = both.framing.to_numpy(); y = both.delta.to_numpy()
        perm_d = [y[p == "placebo"].mean() - y[p == "standard"].mean()
                  for p in (rng.permutation(lab) for _ in range(N_PERM))]
        boots = [rng.choice(plc.delta.to_numpy(), len(plc)).mean()
                 - rng.choice(base.delta.to_numpy(), len(base)).mean() for _ in range(N_BOOT)]
        res["placebo"] = {"placebo_minus_baseline": float(obs),
                          "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                          "p_perm": float((np.sum(np.abs(perm_d) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1)),
                          "n": [len(plc), len(base)]}

    res["cells"] = {
        cid: {"n": int(len(g)), "delta_mean": float(g.delta.mean()), "delta_sd": float(g.delta.std(ddof=1)),
              "share_ge_threshold": float((g.delta >= args.threshold).mean()),
              "joint_reward": float(g.joint_reward.mean()), "consumer_surplus": float(g.consumer_surplus.mean()),
              "categories": dict(Counter(g.category))}
        for cid, g in df.groupby("condition_id")}

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "effects.json").write_text(json.dumps(res, indent=2))
    md = [f"# Confirmatory effects: `{args.run_id}`", "",
          f"Complete balanced batches: {info['n_complete_batches']}; episodes analysed: "
          f"{info['episodes_analysed']} (pathological excluded: {sum(info['pathological_by_cell'].values())}).", "",
          "| hypothesis | contrast | estimate (Delta) | 95% bootstrap CI | Cohen's d | p (perm) | p (Holm) |",
          "|---|---|---|---|---|---|---|"]
    for h, r in res["hypotheses"].items():
        md.append(f"| {h} | {r['factor']}: {r['contrast']} | {r['estimate']:+.3f} | "
                  f"[{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] | {r['cohens_d']:+.2f} | "
                  f"{r['p_perm']:.4f} | {r['p_holm']:.4f} |")
    if "placebo" in res:
        p = res["placebo"]
        md += ["", f"Placebo - baseline: {p['placebo_minus_baseline']:+.3f}, CI "
               f"[{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}], p = {p['p_perm']:.4f} (n = {p['n'][0]} vs {p['n'][1]})."]
    md += ["", "Binary label (logistic robustness):", ""]
    for t, e in res["logit"].items():
        md.append(f"- threshold {t}: positive share {e['positive_share']:.0%}"
                  + (f"; {e['error']}" if "error" in e else ""))
    md += ["", "| cell | n | Delta mean | sd | share >= threshold | joint reward | categories |",
           "|---|---|---|---|---|---|---|"]
    for cid, c in sorted(res["cells"].items()):
        md.append(f"| {cid} | {c['n']} | {c['delta_mean']:+.3f} | {c['delta_sd']:.3f} | "
                  f"{c['share_ge_threshold']:.0%} | {c['joint_reward']:.4f} | {c['categories']} |")
    (out / "effects.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
