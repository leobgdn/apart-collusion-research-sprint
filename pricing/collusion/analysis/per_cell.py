"""Descriptive per-cell table and score distributions (paper Appendices D and E).

    python -m collusion.analysis.per_cell --run-id main_v1

* Per cell: episodes analysed (excluded), Delta mean/sd/median, median and IQR
  of the per-episode mean price, episodes with both sellers at exactly 1.55 and
  zero dispersion, joint reward, consumer surplus, parse failures (all
  episodes, including excluded ones), and mean judge and keyword scores.
* Delta histogram by channel level.
* Judge score counts per measure (message; plan/insight with and without the
  channel).

Purely descriptive, no tests. Writes results/per_cell.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from collusion.analysis.stage4 import short
from collusion.runner.logging_io import read_episode_summaries

ROOT = Path(__file__).resolve().parents[2]
BINS = [-10, -1.5, -1.0, -0.5, -0.15, 0.15, 0.3, 0.6, 1.0, 10]
LABELS = ["< -1.5", "-1.5 to -1.0", "-1.0 to -0.5", "-0.5 to -0.15",
          "-0.15 to 0.15 (near Nash)", "0.15 to 0.3", "0.3 to 0.6", "0.6 to 1.0", "> 1.0"]


def load(run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = [s for s in read_episode_summaries(ROOT / "data" / "episodes" / run_id)
                 if s.get("completed")]
    judge = pd.DataFrame([json.loads(line) for line in
                          open(ROOT / "data" / "judge" / f"{run_id}.jsonl", encoding="utf-8")])
    scores = {(r.episode_id, r.kind): r.score for r in judge.itertuples()}
    rows = []
    for s in summaries:
        am, kw = s["action_metrics"], s["text_baseline"]
        rows.append(dict(
            cell=s["condition_id"], path=s["is_pathological"], delta=s["action_collusion_score"],
            price=am["avg_price"], disp=am["price_dispersion"], jr=s["joint_reward"],
            cs=am["avg_consumer_surplus"], pf=s["parse_failures"],
            kwm=kw.get("message_score"), kwp=kw.get("plan_insight_score"),
            jm=scores.get((s["episode_id"], "message")),
            jp=scores.get((s["episode_id"], "plan_insight"))))
    return pd.DataFrame(rows), judge


def _mean(x: pd.Series) -> str:
    return "-" if x.isna().all() else f"{x.mean():.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-id", default="main_v1")
    args = ap.parse_args(argv)
    d, judge = load(args.run_id)

    md = [f"# Per-cell results: `{args.run_id}`", "",
          "Prices are per-episode averages over the scored window; median and IQR are across "
          "episodes. Parse failures count turns, including those of excluded episodes.", "",
          "| cell | n (excl.) | Delta mean (sd) | Delta median | mean price, median [IQR] "
          "| both at 1.55, no dispersion | joint reward | consumer surplus | parse failures (turns) "
          "| judge plan/insight mean | judge message mean | keyword plan/insight | keyword message |",
          "|" + "---|" * 13]
    for cell, g in sorted(d.groupby("cell"), key=lambda x: (x[0].startswith("channel=private"), x[0])):
        a = g[~g.path]
        lock = ((a.price.round(3) == 1.55) & (a.disp < 1e-9)).sum()
        md.append(
            f"| {short(cell)} | {len(a)} ({int(g.path.sum())}) | {a.delta.mean():+.2f} "
            f"({a.delta.std():.2f}) | {a.delta.median():+.2f} | {a.price.median():.2f} "
            f"[{a.price.quantile(.25):.2f}, {a.price.quantile(.75):.2f}] | {lock}/{len(a)} "
            f"| {a.jr.mean():.3f} | {a.cs.mean():.3f} | {int(g.pf.sum())} | {_mean(a.jp)} "
            f"| {_mean(a.jm)} | {_mean(a.kwp)} | {_mean(a.kwm)} |")

    a = d[~d.path]
    md += ["", "## Delta distribution by channel", ""]
    for ch in ("none", "private"):
        sub = a[a.cell.str.startswith(f"channel={ch}")]
        h = pd.cut(sub.delta, BINS, labels=LABELS, right=False).value_counts().reindex(LABELS)
        md.append(f"- channel={ch} (n = {len(sub)}): " + "; ".join(f"{k}: {v}" for k, v in h.items()))

    md += ["", "## Judge score counts", "",
           f"{len(judge)} records, {int(judge.score.isna().sum())} unparseable.", ""]
    judge["channel"] = judge.condition_id.str.extract(r"channel=(\w+)")[0]
    counts = judge.groupby(["kind", "channel"]).score.value_counts().sort_index()
    for (kind, ch), g in counts.groupby(level=[0, 1]):
        md.append(f"- {kind}, channel={ch} (n = {int(g.sum())}): "
                  + "; ".join(f"{s:g}: {n}" for (_, _, s), n in g.items()))

    (ROOT / "results" / "per_cell.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
