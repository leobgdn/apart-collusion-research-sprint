# Collusion task-feature sweep

Which task features cause LLM agents to collude, and what does restricting each
one cost in joint performance?

Collusion is measured **from actions** (primary) and **from transcripts**
(secondary); the gap between the two is itself a result.

**Environment: Plan A** -- [Colosseum](https://github.com/umass-ai-safety/colosseum)'s
covert-collusion experiment (agents split Jira tickets; a coalition can get a
hidden blackboard), hosted on Fireworks-served Qwen3-8B. Plan B (a repeated
logit-Bertrand pricing market, fully built and tested) is kept as a fallback;
see `DECISIONS.md` for why the pivot and the Section 6.2 substitution it
required (H2 could not be tested as originally written -- Colosseum has no
repeated-round history -- so `topology` substitutes for it as H2').

**Status: Gate 1 cleared, Stage 2 (pilot) in progress.** Colosseum is
installed at a sibling path with Fireworks wired in via a drop-in client swap.
Budget guard, cost tracking, and the judge are built. Blocked on the
Fireworks API key being set before any call is made.

## Setup

**This repo** (analysis tooling, Plan B fallback, tests):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # Windows
# .venv/bin/python -m pip install -e ".[dev]"        # POSIX
```

**Colosseum** (Plan A, the active environment) lives at a documented sibling
path, not vendored into this repo (see `DECISIONS.md`):

```bash
git clone --recurse-submodules https://github.com/umass-ai-safety/colosseum.git ../colosseum
cd ../colosseum && uv venv --python 3.11 .venv && uv sync --no-install-project
```

It needs a Fireworks API key in its own `.env` (never in this repo, never in
chat): `echo 'FIREWORKS_API_KEY=...' >> ../colosseum/.env`. Override the
sibling-path assumption with `COLOSSEUM_ROOT` if cloned elsewhere.

The probe layer additionally needs `pip install -e ".[probes]"` (torch,
transformers) in this repo, to pull Qwen3-8B's weights locally for activation
extraction, and is Stage 5.

## Entry points

`tasks.py` holds the logic; `Makefile` and `make.cmd` are shims over it, so
`make test` and `python tasks.py test` do the same thing.

| command | what it does |
| --- | --- |
| `python tasks.py test` | the acceptance suite (Section 6.4) |
| `python tasks.py pilot` | Plan-B scripted dry pilot — fallback pipeline, zero cost |
| `python tasks.py gate1` | regenerate `results/gate1_report.md` (Plan B) |
| `python tasks.py plan_a_smoke` | Plan A: 1 seed, every cell, prints real $ cost |
| `python tasks.py plan_a_pilot [--cap-usd N]` | Plan A: escalate to the pilot's 5 seeds |
| `python tasks.py plan_a_main [--cap-usd N] --i-have-approval` | Plan A: main sweep, gated |
| `python tasks.py results` | Stage 4 analysis (gated; no sweep data yet) |
| `python tasks.py clean` | delete generated logs/episodes/results (asks first) |

## Reproducing from raw logs (Plan A)

Every number and figure is produced by a script from the saved logs; nothing is
hand-entered.

1. **Smoke, then escalate.** `plan_a_smoke` runs 1 seed across every cell of
   `collusion/envs/colosseum_adapter/configs/pilot_fireworks.yaml` and measures
   real token cost from Colosseum's own `agent_turns.json` logs (it has no $
   concept of its own; see `cost_tracking.py`). `plan_a_pilot` then projects the
   cost of the pilot's full 5 seeds from that measurement and, if under the cap,
   calls Colosseum's `resume.py` to fill in the rest **in place** under the same
   output root — Colosseum's own resumability primitive, not a reimplementation
   of one (Section 3 rule 5).
2. **Action-level metrics.** Colosseum's own `experiments/collusion/metrics.py`
   (regret, coalition advantage) and `compute_jira_optimal.py` (exact per-run
   optimum), computed from its saved logs, zero additional cost.
3. **Text-level score.** `collusion.metrics.text_judge.run_judge` reuses
   Colosseum's published rubric verbatim (`judge_fireworks.py` patches only the
   backend from OpenAI to Fireworks), then `read_text_scores` normalises the
   0-5 rating to 0-1.
4. **Report.** `python tasks.py gate1` regenerates the Gate 1 artefacts (these
   are Plan-B-specific; a Plan-A gate report script is Stage 2/4 work).

Plan B (below) has its own, already-passing version of steps 1-2 and remains a
tested fallback.

## What is measured

**Plan A (active):** Colosseum's regret-based metrics per run —
`mean_regret`, `coalition_mean_regret`, `noncoalition_mean_regret`, and
**coalition advantage** (`noncoalition_mean_regret − coalition_mean_regret`,
the primary action-level collusion quantity: positive means the coalition left
less regret on the table for itself than it imposed on everyone else) — plus
the exact per-run optimum and a 0-1 normalised judge score. Full definitions in
`prereg.md`.

**Plan B (fallback):** over the last 50 periods of a repeated pricing episode —
**collusion index** `Delta = (avg_profit − profit_Nash) / (profit_Monopoly − profit_Nash)`,
a price index `Delta_p` on the same scale, a category (`near_nash` / `partial`
/ `near_monopoly`, plus the pathological `over_priced` / `below_cost`, which
take precedence and are reported separately — a profit index alone would
misread both as competition), joint reward, consumer surplus, and per-turn
best-response regret. Its Nash/joint-monopoly benchmarks are solved numerically
and reproduce Calvano et al. (2020) to four decimals (1.4729 / 1.9249).

## Factors

**Plan A (active):**

| factor | levels | native Colosseum field | hypothesis |
| --- | --- | --- | --- |
| `channel` | off / on | `secret_channel_enabled` | H1 — positive control |
| `topology` | `complete` / `path` | `communication_network.topology` | H2' — **substitutes** H2 (see `DECISIONS.md` / `prereg.md` §2: Colosseum has no repeated-round history, so H2 as originally written is not tested) |
| `incentive` | `control` / `simple` | `prompt_variant` | H3 |

6 unique cells (the raw 2x2x2 dedups because `prompt_variant` is forced to
`control` whenever `channel` is off — Colosseum's own design, not ours).

**Plan B (fallback):** `channel` (`none`/`private`, H1), `history_window`
(`long`/`short`, H2 as originally written), `horizon` (`long_run`/`myopic`,
H3) — 8 cells, enforced one-manipulation-per-condition via slot ownership
(`collusion/envs/pricing/conditions.py` `OWNED_SLOTS`,
`tests/test_prompt_diff.py`).

## Layout

```
collusion/
  config/experiment.yaml          Plan B factors/seeds/models; Plan A notes which block is active
  envs/pricing/                   Plan B: economics, conditions, prompts, state, env (fallback)
  envs/colosseum_adapter/         Plan A integration: configs/, run_plan_a.py, cost_tracking.py,
                                   judge_fireworks.py, NOTES.md (recon findings)
  agents/                         Plan B: parsing, scripted controls
  runner/                         Plan B: run_episode, run_sweep, replay, gate1_report
  metrics/action_metrics.py       Plan B action-level metrics
  metrics/text_judge.py           Plan A judge reader/normaliser (both envs' judge scores are 0-1)
  analysis/  probes/              Stage 4 / Stage 5, empty until those gates
data/logs/  data/episodes/        Plan B generated data, gitignored
../colosseum/                     Plan A checkout + its own outputs (sibling dir, not in this repo)
tests/                            the acceptance suite (Plan B; fully passing)
prereg.md                         hypotheses and tests, Plan A (DRAFT until Gate 2)
DECISIONS.md                      every design decision and why
```

## Gates

Work stops at each gate for human review (build prompt Section 6).

| gate | after | status |
| --- | --- | --- |
| 1 | environment + scripted controls | **cleared** — human chose Plan A, Fireworks, Qwen3-8B |
| 2 | baseline tuning and pilot | **in progress** — blocked on the Fireworks API key being set |
| 3 | pre-registration and main sweep | blocked on Gate 2 |
| 4 | analysis | blocked on Gate 3 |
| 5 | probe layer | blocked on Gate 4 |

## Reference

- Calvano, Calzolari, Denicolò & Pastorello (2020), "Artificial Intelligence,
  Algorithmic Pricing, and Collusion", *AER* 110(10) — the market model and the
  benchmark prices used as an external check.
- *Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems*,
  [arXiv:2602.15198](https://arxiv.org/abs/2602.15198) — Plan A; its "collusion
  on paper" finding is close to this project's H5. See
  `collusion/envs/colosseum_adapter/NOTES.md`.
