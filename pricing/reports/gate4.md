# Gate 4 report: Stage 4 results for `main_v1` (BLOCKING)

Every number here comes from `results/effects.md`, `results/stage4.md` and
`reports/judge_validation.md`. These are produced by `collusion.analysis.effects`,
`collusion.analysis.stage4` and `collusion.analysis.judge_validation` from the
saved logs. Figures are in `results/figures/`. 446 episodes are analysed: 25
complete batches, with the 4 pre-registered over-priced episodes excluded.

## 1. Confirmatory effects (H1-H3, placebo)

| | contrast | effect on Delta | 95% bootstrap CI | p (perm., Holm) |
|---|---|---|---|---|
| H1 | private channel - none | **+1.771** | [+1.691, +1.849] | 0.0003 |
| H2 | long - short history | **+0.092** | [+0.015, +0.170] | 0.029 |
| H3 | long-run - myopic framing | **+0.153** | [+0.074, +0.235] | 0.0008 |
| placebo | placebo - baseline wording | +0.112 | [-0.163, +0.372] | 0.42 |

All three hypotheses are supported in the pre-registered direction. The
placebo shows no difference, as expected, and the OLS cross-check (HC3)
agrees. In the logistic robustness check on the binary label, H1 is
significant at 0.3 (p = 0.026) and at 0.15 (p < 1e-30) but not at 0.5
(p = 0.12). H2 and H3 are not significant in any logistic fit.

## 2. What the effects mean (read this before the numbers)

- **Channel-on sellers lock onto one focal price.** In 171 of 200 channel-on
  episodes both sellers price at exactly **1.55**, with zero dispersion. This
  is one menu step above Nash (1.47) and gives Delta of about 0.26. It starts
  in period 2 and holds for the rest of the game (`price_trajectories.png`).
- **Without the channel, prices fall well below Nash.** The median price is
  1.10 and Delta is about -1.5, so the sellers sit in a price war near cost.
  Most of H1's +1.77 is the channel **preventing a sub-Nash price war**. It is
  not movement toward monopoly: channel-on Delta is about a quarter of the way
  from Nash to monopoly.
- **H2 and H3 come only from the no-channel cells.** The four channel-on cells
  are flat (Delta 0.23-0.29). The interaction is exploratory under the prereg.
- **The 0.3 label sits just above the focal price.** Only 20 of 446 episodes
  (5%) are positive at 0.3: 14 with the channel on, 6 with it off. At 0.15, 49%
  are positive, and that label almost exactly matches whether the channel is on.

## 3. Collusion vs cost (prereg 5)

Every manipulation that raises Delta moves surplus from consumers to sellers
(`cost_map.png`).

| | change in Delta | change in joint reward | change in consumer surplus |
|---|---|---|---|
| H1 | +1.77 [+1.69, +1.85] | +0.41 [+0.39, +0.42] | -0.42 [-0.44, -0.40] |
| H2 | +0.09 [+0.01, +0.17] | +0.02 [+0.00, +0.04] | -0.02 [-0.04, +0.00] |
| H3 | +0.15 [+0.07, +0.23] | +0.04 [+0.02, +0.05] | -0.05 [-0.07, -0.03] |
| placebo | +0.11 [-0.16, +0.38] | +0.03 [-0.04, +0.09] | -0.01 [-0.08, +0.07] |

## 4. Predict-then-verify (prereg 6)

Each pair of the 8 crossed cells is held out in turn (28 splits). The additive
main-effects model is fitted on the other 6 cells.

| outcome | mean error, model | mean error, grand mean | model better in |
|---|---|---|---|
| Delta | 0.190 | 1.012 | 28/28 |
| share Delta >= 0.3 | 0.072 | 0.048 | 6/28 |

The main effects predict held-out cell means of Delta well, mostly because of
the large channel effect. They do not predict the rare binary label.

## 5. Judge validation

- **The labels are LLM pre-labels, not independent human labels.** Claude
  pre-labelled all 30 rows from the digest without seeing the key or any
  scores. The human then reviewed every row and changed none (0/30); see
  `DECISIONS.md`. The write-up must say this in these words.
- **Plan/insight:** n = 20, 6 positives. The calibrated threshold is 0.60.
  Balanced accuracy 0.71, kappa 0.31, AUROC 0.76, all in-sample and therefore
  optimistic. This is only fair agreement. The judge flags 126 of 250
  no-channel plan/insight texts at 0.6, many of which the labels call 0
  ("avoid competition", "stability").
- **Message:** n = 10, 10 positives, 0 negatives, so calibration is
  **degenerate**. The "threshold" of 0.60 is simply the judge's lowest message
  score: every channel-on message stream is flagged.

## 6. Text vs action (prereg 4) and H5

H5 is tested as an exact McNemar test on the discordant episodes, comparing
text-only with action-only. The prereg names H5 but not its test, so this
choice is disclosed. The action flag is Delta >= 0.3.

| measure | both | text only | action only | neither | McNemar |
|---|---|---|---|---|---|
| message, judge (channel on) | 7% | 93% | 0% | 0% | 186 vs 0, p = 2e-56 |
| plan/insight, judge | 4% | 64% | 1% | 32% | 285 vs 4, p = 6e-79 |
| message, keyword (exploratory) | 4% | 77% | 3% | 16% | 154 vs 6, p = 3e-38 |
| plan/insight, keyword (exploratory) | 1% | 14% | 4% | 81% | 64 vs 17, p = 1e-7 |

**The gap depends on the action threshold.** At 0.15 the message measure is
93% "both" and 7% text-only. Text and action then agree: the sellers say they
will coordinate, and do, at 1.55. At 0.5 it is 96% text-only. So the honest
reading is: *agents talk about coordination far more strongly than their
prices show; the prices they coordinate on sit just above Nash.*

## 7. Things that surprised me

1. Coordination is immediate and stable: the 1.55 lock-in happens in period 2.
2. The channel's main effect is preventing a price war, not reaching monopoly.
3. The judge scores generously (no message scores below 0.6), and its
   plan/insight agreement with the labels is only fair.
4. No-channel plan/insight text is still flagged by the judge in 36-51% of
   episodes in the standard cells and 75% in the placebo cell, where prices are
   near cost. These are probably judge false positives. The keyword baseline
   flags 0% there.

## 8. Implications for the probes (Stage 5)

- The primary label (0.3) has 20 positive episodes (about 1,200 of 27,000
  window turns), so probes on it are **underpowered**. Results will have wide
  CIs and must be reported as such.
- The secondary label (0.15) is close to an indicator of whether the channel
  is on. A probe on it could just detect the channel. The pre-registered
  within-channel CV and leave-one-condition-out transfer are therefore the key
  controls. A probe must also beat the price-history baseline (prereg).
- The exploratory `rolling_delta` regression probe has the most signal to work
  with.
- Estimated run time: 1.5-2.5 h on the CPU; `train_eval.py` has not yet been
  run on real data.

## Decisions needed (BLOCKING)

1. **Go / no-go for Stage 5 probes,** run exactly as pre-registered, with the
   power caveat above.
2. **Status of H5 in the write-up.** (a) Confirmatory with the caveats in
   sections 5-6, or (b) exploratory. Recommendation: **(b)**. Three reasons:
   message calibration is degenerate, the labels are LLM pre-labels, and the
   gap depends on the threshold. H1-H3 stay confirmatory either way.
