# Probes: `main_v1` (window turns)

26760 turns, 446 non-pathological episodes, layers [11, 14, 18, 22, 25, 29].

## Label Delta >= 0.3 (positive share 4%, n+ = 20)

| layer | LR mean | LR max | DoM mean | DoM max | LR mean TPR@5%FPR |
|---|---|---|---|---|---|
| 11 | 0.878 [0.80, 0.94] | 0.877 [0.80, 0.94] | 0.619 [0.46, 0.77] | 0.632 [0.48, 0.77] | 0.55 |
| 14 | 0.836 [0.74, 0.92] | 0.819 [0.76, 0.88] | 0.630 [0.48, 0.77] | 0.628 [0.47, 0.77] | 0.50 |
| 18 | 0.917 [0.83, 0.99] | 0.884 [0.80, 0.94] | 0.630 [0.47, 0.77] | 0.631 [0.47, 0.77] | 0.85 |
| 22 | 0.853 [0.74, 0.94] | 0.851 [0.75, 0.94] | 0.668 [0.54, 0.80] | 0.677 [0.54, 0.79] | 0.70 |
| 25 | 0.915 [0.85, 0.96] | 0.909 [0.84, 0.96] | 0.681 [0.55, 0.80] | 0.687 [0.56, 0.80] | 0.65 |
| 29 | 0.909 [0.84, 0.97] | 0.907 [0.84, 0.96] | 0.682 [0.56, 0.79] | 0.685 [0.57, 0.79] | 0.65 |

Shuffled-label control (layer 18): AUROC 0.5529342723004695

Baselines:

- tfidf: AUROC 0.857 [0.7622938918618853, 0.9383143581719094]
- price_history: AUROC 0.996 [0.9897077004966294, 0.999692862316439]
- judge_max: AUROC 0.564 [0.4835085436378005, 0.6342432819216356]
- judge_plan_insight: AUROC 0.572 [0.4732955625758705, 0.6650997652582158]

Probe (layer 18, LR mean) minus baseline, AUROC, paired bootstrap:

- vs tfidf: +0.060 [-0.017, +0.151]
- vs price_history: -0.079 [-0.169, -0.009]
- vs judge_max: +0.354 [+0.242, +0.457]

Leave-one-condition-out transfer (LR mean, best layer):

- channel=none.framing=placebo.history_window=long.horizon=long_run: 1.0 (n+ = 2/49)
- channel=none.framing=standard.history_window=long.horizon=long_run: 0.842 (n+ = 4/50)
- channel=none.framing=standard.history_window=long.horizon=myopic: single-class cell, AUROC undefined (n+ = 0/50)
- channel=none.framing=standard.history_window=short.horizon=long_run: single-class cell, AUROC undefined (n+ = 0/47)
- channel=none.framing=standard.history_window=short.horizon=myopic: single-class cell, AUROC undefined (n+ = 0/50)
- channel=private.framing=standard.history_window=long.horizon=long_run: 0.885 (n+ = 2/50)
- channel=private.framing=standard.history_window=long.horizon=myopic: 1.0 (n+ = 1/50)
- channel=private.framing=standard.history_window=short.horizon=long_run: 0.995 (n+ = 4/50)
- channel=private.framing=standard.history_window=short.horizon=myopic: 0.85 (n+ = 7/50)

Within-channel CV: {"private": {"auroc": 0.9650537634408602, "ci95": [0.9111752611234476, 0.9975949975949976], "tpr_at_5fpr": 0.9285714285714286, "n_episodes": 200, "n_positive": 14}, "none": {"auroc": 0.9125, "ci95": [0.8152381365747917, 0.9945693254089718], "tpr_at_5fpr": 0.5, "n_episodes": 246, "n_positive": 6}}

## Label Delta >= 0.15 (positive share 44%, n+ = 197)

| layer | LR mean | LR max | DoM mean | DoM max | LR mean TPR@5%FPR |
|---|---|---|---|---|---|
| 11 | 0.970 [0.95, 0.98] | 0.959 [0.94, 0.98] | 0.931 [0.90, 0.96] | 0.932 [0.90, 0.96] | 0.95 |
| 14 | 0.964 [0.95, 0.98] | 0.957 [0.93, 0.98] | 0.930 [0.90, 0.96] | 0.929 [0.90, 0.96] | 0.95 |
| 18 | 0.968 [0.95, 0.98] | 0.966 [0.95, 0.98] | 0.927 [0.90, 0.95] | 0.927 [0.90, 0.95] | 0.96 |
| 22 | 0.968 [0.95, 0.98] | 0.965 [0.95, 0.98] | 0.929 [0.90, 0.96] | 0.928 [0.90, 0.95] | 0.96 |
| 25 | 0.977 [0.96, 0.99] | 0.962 [0.94, 0.98] | 0.934 [0.91, 0.96] | 0.938 [0.91, 0.96] | 0.97 |
| 29 | 0.976 [0.96, 0.99] | 0.965 [0.95, 0.98] | 0.945 [0.92, 0.97] | 0.948 [0.92, 0.97] | 0.96 |

Shuffled-label control (layer 25): AUROC 0.40017939779422257

Baselines:

- tfidf: AUROC 0.972 [0.9563494245789058, 0.9860116609644102]
- price_history: AUROC 1.0 [0.9986658138137764, 1.0]
- judge_max: AUROC 0.777 [0.7450407178071791, 0.8090586318987054]
- judge_plan_insight: AUROC 0.707 [0.6687658940735428, 0.7436698445773606]

Probe (layer 25, LR mean) minus baseline, AUROC, paired bootstrap:

- vs tfidf: +0.005 [-0.009, +0.018]
- vs price_history: -0.023 [-0.038, -0.011]
- vs judge_max: +0.199 [+0.168, +0.231]

Leave-one-condition-out transfer (LR mean, best layer):

- channel=none.framing=placebo.history_window=long.horizon=long_run: 0.761 (n+ = 3/49)
- channel=none.framing=standard.history_window=long.horizon=long_run: 0.957 (n+ = 7/50)
- channel=none.framing=standard.history_window=long.horizon=myopic: 1.0 (n+ = 1/50)
- channel=none.framing=standard.history_window=short.horizon=long_run: 1.0 (n+ = 1/47)
- channel=none.framing=standard.history_window=short.horizon=myopic: single-class cell, AUROC undefined (n+ = 0/50)
- channel=private.framing=standard.history_window=long.horizon=long_run: 0.933 (n+ = 45/50)
- channel=private.framing=standard.history_window=long.horizon=myopic: 1.0 (n+ = 47/50)
- channel=private.framing=standard.history_window=short.horizon=long_run: 1.0 (n+ = 47/50)
- channel=private.framing=standard.history_window=short.horizon=myopic: 0.647 (n+ = 46/50)

Within-channel CV: {"private": {"auroc": 0.836036036036036, "ci95": [0.685483870967742, 0.9432424549445827], "tpr_at_5fpr": 1.0, "n_episodes": 200, "n_positive": 185}, "none": {"auroc": 0.905982905982906, "ci95": [0.8350255807117202, 0.9684913117789489], "tpr_at_5fpr": 0.6666666666666666, "n_episodes": 246, "n_positive": 12}}

## Exploratory: rolling_delta ridge (episode-grouped CV R^2)

- layer 11: 0.859
- layer 14: 0.853
- layer 18: 0.861
- layer 22: 0.857
- layer 25: 0.868
- layer 29: 0.861