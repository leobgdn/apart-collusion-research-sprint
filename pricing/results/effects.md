# Confirmatory effects: `main_v1`

Complete balanced batches: 25; episodes analysed: 446 (pathological excluded: 4).

| hypothesis | contrast | estimate (Delta) | 95% bootstrap CI | Cohen's d | p (perm) | p (Holm) |
|---|---|---|---|---|---|---|
| H1 | channel: private - none | +1.771 | [+1.691, +1.849] | +4.37 | 0.0001 | 0.0003 |
| H2 | history_window: long - short | +0.092 | [+0.015, +0.170] | +0.23 | 0.0285 | 0.0285 |
| H3 | horizon: long_run - myopic | +0.153 | [+0.074, +0.235] | +0.38 | 0.0004 | 0.0008 |

Placebo - baseline: +0.112, CI [-0.163, +0.372], p = 0.4167 (n = 49 vs 50).

Binary label (logistic robustness):

- threshold 0.3: positive share 5%
- threshold 0.15: positive share 49%
- threshold 0.5: positive share 2%

| cell | n | Delta mean | sd | share >= threshold | joint reward | categories |
|---|---|---|---|---|---|---|
| channel=none.framing=placebo.history_window=long.horizon=long_run | 49 | -1.103 | 0.544 | 4% | 0.1932 | {'sub_nash': 44, 'partial': 3, 'near_nash': 2} |
| channel=none.framing=standard.history_window=long.horizon=long_run | 50 | -1.215 | 0.805 | 8% | 0.1675 | {'near_monopoly': 2, 'sub_nash': 43, 'partial': 5} |
| channel=none.framing=standard.history_window=long.horizon=myopic | 50 | -1.605 | 0.464 | 0% | 0.0781 | {'sub_nash': 48, 'near_nash': 1, 'partial': 1} |
| channel=none.framing=standard.history_window=short.horizon=long_run | 47 | -1.501 | 0.482 | 0% | 0.1020 | {'sub_nash': 45, 'near_nash': 1, 'partial': 1} |
| channel=none.framing=standard.history_window=short.horizon=myopic | 50 | -1.727 | 0.154 | 0% | 0.0500 | {'sub_nash': 50} |
| channel=private.framing=standard.history_window=long.horizon=long_run | 50 | +0.260 | 0.132 | 4% | 0.5055 | {'partial': 44, 'near_nash': 5, 'near_monopoly': 1} |
| channel=private.framing=standard.history_window=long.horizon=myopic | 50 | +0.262 | 0.052 | 2% | 0.5059 | {'partial': 47, 'near_nash': 3} |
| channel=private.framing=standard.history_window=short.horizon=long_run | 50 | +0.226 | 0.371 | 8% | 0.4976 | {'partial': 45, 'sub_nash': 2, 'near_monopoly': 2, 'near_nash': 1} |
| channel=private.framing=standard.history_window=short.horizon=myopic | 50 | +0.287 | 0.192 | 14% | 0.5115 | {'partial': 44, 'sub_nash': 1, 'near_nash': 3, 'near_monopoly': 2} |