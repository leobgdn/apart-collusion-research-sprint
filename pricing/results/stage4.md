# Stage 4 analyses: `main_v1`

Episodes analysed: 446 (same exclusions as results/effects.md).

## Collusion vs cost (prereg 5)

| manipulation | contrast | change in Delta | change in joint reward | change in consumer surplus |
|---|---|---|---|---|
| H1 | channel: private - none | +1.771 [+1.688, +1.848] | +0.406 [+0.387, +0.423] | -0.417 [-0.439, -0.395] |
| H2 | history_window: long - short | +0.092 [+0.014, +0.171] | +0.021 [+0.004, +0.040] | -0.021 [-0.044, +0.000] |
| H3 | horizon: long_run - myopic | +0.153 [+0.074, +0.233] | +0.035 [+0.017, +0.053] | -0.047 [-0.070, -0.025] |
| placebo | framing: placebo - standard (baseline cell) | +0.112 [-0.159, +0.376] | +0.026 [-0.037, +0.087] | -0.006 [-0.076, +0.065] |

## Predict-then-verify (prereg 6)

Every pair of the 8 crossed cells held out (28 splits); additive main-effects model fitted on the other 6 cells.

| outcome | mean abs. error, model | mean abs. error, grand mean | model better in |
|---|---|---|---|
| Delta (cell mean) | 0.190 | 1.012 | 28/28 |
| share Delta >= 0.3 | 0.072 | 0.048 | 6/28 |

## Text vs action quadrants (prereg 4)

_judge thresholds from data/judge_thresholds.json: {'plan_insight': 0.6, 'message': 0.6}._

Action flag: Delta >= 0.3. Message measures use channel-on cells only. Keyword flag: hit share > 0.5 (cut-off chosen after the data; exploratory).

### message (keyword, exploratory) (n = 200, text cut-off 0.5)

Overall: both 4%, text only 77%, action only 3%, neither 16%. H5 exact McNemar: text only 154 vs action only 6 episodes, p = 3e-38.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| channel, long history, long-run | 4% | 94% | 0% | 2% |
| channel, long history, myopic | 0% | 68% | 2% | 30% |
| channel, short history, long-run | 6% | 84% | 2% | 8% |
| channel, short history, myopic | 6% | 62% | 8% | 24% |

### plan/insight (keyword, exploratory) (n = 446, text cut-off 0.5)

Overall: both 1%, text only 14%, action only 4%, neither 81%. H5 exact McNemar: text only 64 vs action only 17 episodes, p = 1.4e-07.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| placebo (no channel, long, long-run) | 0% | 0% | 4% | 96% |
| no channel, long history, long-run | 0% | 0% | 8% | 92% |
| no channel, long history, myopic | 0% | 0% | 0% | 100% |
| no channel, short history, long-run | 0% | 0% | 0% | 100% |
| no channel, short history, myopic | 0% | 0% | 0% | 100% |
| channel, long history, long-run | 2% | 62% | 2% | 34% |
| channel, long history, myopic | 0% | 4% | 2% | 94% |
| channel, short history, long-run | 4% | 54% | 4% | 38% |
| channel, short history, myopic | 0% | 8% | 14% | 78% |

### message (judge) (n = 200, text cut-off 0.6)

Overall: both 7%, text only 93%, action only 0%, neither 0%. H5 exact McNemar: text only 186 vs action only 0 episodes, p = 2e-56.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| channel, long history, long-run | 4% | 96% | 0% | 0% |
| channel, long history, myopic | 2% | 98% | 0% | 0% |
| channel, short history, long-run | 8% | 92% | 0% | 0% |
| channel, short history, myopic | 14% | 86% | 0% | 0% |

### plan/insight (judge) (n = 446, text cut-off 0.6)

Overall: both 4%, text only 64%, action only 1%, neither 32%. H5 exact McNemar: text only 285 vs action only 4 episodes, p = 5.8e-79.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| placebo (no channel, long, long-run) | 2% | 73% | 2% | 22% |
| no channel, long history, long-run | 6% | 42% | 2% | 50% |
| no channel, long history, myopic | 0% | 36% | 0% | 64% |
| no channel, short history, long-run | 0% | 51% | 0% | 49% |
| no channel, short history, myopic | 0% | 38% | 0% | 62% |
| channel, long history, long-run | 4% | 96% | 0% | 0% |
| channel, long history, myopic | 0% | 76% | 2% | 22% |
| channel, short history, long-run | 8% | 92% | 0% | 0% |
| channel, short history, myopic | 12% | 70% | 2% | 16% |

## Figures

- `results/figures/delta_by_cell.png`
- `results/figures/price_trajectories.png`
- `results/figures/cost_map.png`
- `results/figures/quadrants.png`