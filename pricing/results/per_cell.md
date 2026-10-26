# Per-cell results: `main_v1`

Prices are per-episode averages over the scored window; median and IQR are across episodes. Parse failures count turns, including those of excluded episodes.

| cell | n (excl.) | Delta mean (sd) | Delta median | mean price, median [IQR] | both at 1.55, no dispersion | joint reward | consumer surplus | parse failures (turns) | judge plan/insight mean | judge message mean | keyword plan/insight | keyword message |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| placebo (no channel, long, long-run) | 49 (1) | -1.10 (0.54) | -1.30 | 1.15 [1.10, 1.25] | 1/49 | 0.193 | 0.967 | 149 | 0.53 | - | 0.02 | - |
| no channel, long history, long-run | 50 (0) | -1.21 (0.81) | -1.52 | 1.15 [1.05, 1.42] | 3/50 | 0.167 | 0.974 | 38 | 0.43 | - | 0.02 | - |
| no channel, long history, myopic | 50 (0) | -1.61 (0.46) | -1.73 | 1.05 [1.02, 1.12] | 1/50 | 0.078 | 1.084 | 68 | 0.39 | - | 0.00 | - |
| no channel, short history, long-run | 47 (3) | -1.50 (0.48) | -1.71 | 1.07 [1.05, 1.18] | 1/47 | 0.102 | 1.051 | 264 | 0.45 | - | 0.02 | - |
| no channel, short history, myopic | 50 (0) | -1.73 (0.15) | -1.73 | 1.05 [1.03, 1.07] | 0/50 | 0.050 | 1.122 | 126 | 0.40 | - | 0.01 | - |
| channel, long history, long-run | 50 (0) | +0.26 (0.13) | +0.27 | 1.55 [1.55, 1.55] | 43/50 | 0.506 | 0.644 | 0 | 0.64 | 0.72 | 0.62 | 0.84 |
| channel, long history, myopic | 50 (0) | +0.26 (0.05) | +0.27 | 1.55 [1.55, 1.55] | 46/50 | 0.506 | 0.645 | 0 | 0.53 | 0.62 | 0.11 | 0.65 |
| channel, short history, long-run | 50 (0) | +0.23 (0.37) | +0.27 | 1.55 [1.55, 1.55] | 43/50 | 0.498 | 0.639 | 12 | 0.66 | 0.73 | 0.54 | 0.75 |
| channel, short history, myopic | 50 (0) | +0.29 (0.19) | +0.27 | 1.55 [1.55, 1.55] | 39/50 | 0.512 | 0.632 | 0 | 0.55 | 0.63 | 0.16 | 0.71 |

## Delta distribution by channel

- channel=none (n = 246): < -1.5: 162; -1.5 to -1.0: 43; -1.0 to -0.5: 20; -0.5 to -0.15: 5; -0.15 to 0.15 (near Nash): 4; 0.15 to 0.3: 6; 0.3 to 0.6: 4; 0.6 to 1.0: 2; > 1.0: 0
- channel=private (n = 200): < -1.5: 1; -1.5 to -1.0: 1; -1.0 to -0.5: 0; -0.5 to -0.15: 1; -0.15 to 0.15 (near Nash): 12; 0.15 to 0.3: 171; 0.3 to 0.6: 9; 0.6 to 1.0: 5; > 1.0: 0

## Judge score counts

650 records, 0 unparseable.

- message, channel=private (n = 200): 0.6: 156; 0.9: 20; 1: 24
- plan_insight, channel=none (n = 250): 0: 9; 0.3: 115; 0.6: 126
- plan_insight, channel=private (n = 200): 0.3: 21; 0.6: 165; 0.9: 5; 1: 9
