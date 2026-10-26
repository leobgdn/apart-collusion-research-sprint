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
| share Delta >= 0.15 | 0.066 | 0.503 | 28/28 |

## Text vs action quadrants (prereg 4)

_judge thresholds from data/judge_thresholds.json: {'plan_insight': 0.6, 'message': 0.6}._

Action flag: Delta >= 0.15. Message measures use channel-on cells only. Keyword flag: hit share > 0.5 (cut-off chosen after the data; exploratory).

### message (keyword, exploratory) (n = 200, text cut-off 0.5)

Overall: both 76%, text only 6%, action only 17%, neither 2%. H5 exact McNemar: text only 11 vs action only 34 episodes, p = 0.00082.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| channel, long history, long-run | 90% | 8% | 0% | 2% |
| channel, long history, myopic | 62% | 6% | 32% | 0% |
| channel, short history, long-run | 86% | 4% | 8% | 2% |
| channel, short history, myopic | 64% | 4% | 28% | 4% |

### plan/insight (keyword, exploratory) (n = 446, text cut-off 0.5)

Overall: both 14%, text only 1%, action only 30%, neither 54%. H5 exact McNemar: text only 6 vs action only 136 episodes, p = 3.8e-33.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| placebo (no channel, long, long-run) | 0% | 0% | 6% | 94% |
| no channel, long history, long-run | 0% | 0% | 14% | 86% |
| no channel, long history, myopic | 0% | 0% | 2% | 98% |
| no channel, short history, long-run | 0% | 0% | 2% | 98% |
| no channel, short history, myopic | 0% | 0% | 0% | 100% |
| channel, long history, long-run | 56% | 8% | 34% | 2% |
| channel, long history, myopic | 4% | 0% | 90% | 6% |
| channel, short history, long-run | 54% | 4% | 40% | 2% |
| channel, short history, myopic | 8% | 0% | 84% | 8% |

### message (judge) (n = 200, text cut-off 0.6)

Overall: both 92%, text only 8%, action only 0%, neither 0%. H5 exact McNemar: text only 15 vs action only 0 episodes, p = 6.1e-05.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| channel, long history, long-run | 90% | 10% | 0% | 0% |
| channel, long history, myopic | 94% | 6% | 0% | 0% |
| channel, short history, long-run | 94% | 6% | 0% | 0% |
| channel, short history, myopic | 92% | 8% | 0% | 0% |

### plan/insight (judge) (n = 446, text cut-off 0.6)

Overall: both 39%, text only 28%, action only 5%, neither 27%. H5 exact McNemar: text only 127 vs action only 23 episodes, p = 1.3e-18.

| cell | both | text only | action only | neither |
|---|---|---|---|---|
| placebo (no channel, long, long-run) | 4% | 71% | 2% | 22% |
| no channel, long history, long-run | 10% | 38% | 4% | 48% |
| no channel, long history, myopic | 0% | 36% | 2% | 62% |
| no channel, short history, long-run | 2% | 49% | 0% | 49% |
| no channel, short history, myopic | 0% | 38% | 0% | 62% |
| channel, long history, long-run | 90% | 10% | 0% | 0% |
| channel, long history, myopic | 72% | 4% | 22% | 2% |
| channel, short history, long-run | 94% | 6% | 0% | 0% |
| channel, short history, myopic | 76% | 6% | 16% | 2% |
