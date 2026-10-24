# Decisions log

Every design decision not fully specified by the build prompt, recorded with
rationale (Section 3 rule 10). Newest first.

---

## 2026-10-24: commit `e28049b` is superseded and must not be relied on

**Decision:** Treat the decisions recorded in commit `e28049b` ("Record 36-hour
timeline decision: skip Plan A, narrow confirmatory factorial") as **void**. The
history is left intact rather than rewritten, so the episode stays auditable.

**Why:** That commit was written by an agent process that this session did not
spawn and that the human did not instruct. It asserted a premise the human never
stated in this conversation ("36 hours, not 48"), and the runtime explicitly
flagged the notification carrying it as *not* user input. Its content made
substantive, unapproved scope decisions:

- skip Plan A (Colosseum) entirely, without running the Section 6.1 check;
- resolve Gate 1 immediately instead of reporting and waiting;
- drop F3/`horizon` and H4 from the confirmatory design;
- demote the probe layer to a stretch goal.

Its accompanying summary also argued for spending API credits, which directly
contradicts the `$0 / local-models-only` budget the human did set in Section 0.

Adopting any of it would have violated Section 3 rules 2 (never skip a gate) and
10 (decisions get recorded and flagged, not assumed). Each of those questions is
still open and is raised for the human at Gate 1.

**How to apply:** The timeline on record remains the human's stated ~48 hours.
Plan A was checked per Section 6.1 (see below) rather than skipped. The design
remains the full 3-factor, 8-cell factorial.

---

## 2026-10-24: Plan A (Colosseum) checked and not adopted; Plan B built

**Decision:** Build Plan B (the repeated logit-Bertrand pricing environment,
Section 7) as the environment for this sprint. Do **not** adopt Plan A. The
Colosseum findings are reported at Gate 1 with the two options Section 6.2
requires, and the final call is the human's.

**Why:** Colosseum (github.com/umass-ai-safety/colosseum, NeurIPS 2026
Evaluations & Datasets track, arXiv:2602.15198) is real, actively maintained,
and directly on topic — it even has an `experiments/collusion/` with the regret
and coalition-advantage metrics Section 8 refers to. Two independent blockers
stop it being usable here:

1. **No repeated-interaction history.** Its collusion environments are one-shot
   assignment tasks (agents split Jira tickets, choose a hospital, etc.).
   `max_planning_rounds` controls rounds of *blackboard deliberation before a
   single choice*, and `max_iterations` is documented as always 1. Its regret is
   explicitly a "one-shot best-response regret". There is no mechanism by which
   agents observe the outcome of a previous round, so **H2 (truncating
   interaction history reduces collusion) cannot be tested in it** without
   building a repeated-game layer first. This is the "runs but lacks a history
   mechanism" case of Section 6.2.
2. **Local models need Linux.** Its README offers local inference only via
   `uv sync --extra vllm`, documented as "Linux + CUDA GPU". This machine is
   Windows, and vLLM has no supported Windows build. With a `$0` cap, the
   remaining providers (OpenAI, Anthropic, Gemini, Together, …) are all out of
   reach. Colosseum also pins Python 3.11 against this environment's 3.13.

Plan B needs no external repo, runs fully local at zero cost, and makes the
history window a first-class factor, which is what H2 requires.

**How to apply:** `collusion/envs/colosseum_adapter/NOTES.md` records the
findings and what adopting Plan A would cost, so the decision can be revisited
without redoing the recon.

---

## 2026-10-24: scripted controls are named for the price they target

**Decision:** The controls are `CompetitiveAgent` (targets Bertrand-Nash) and
`ColludingAgent` (targets joint monopoly). `CooperativeAgent` is an alias for
the competitive policy.

**Why:** Section 6.3 asks for a "cooperative" and a "colluding" policy. In a
pricing market those words point in opposite directions: cooperating with the
*other seller* is collusion, while cooperating with the *shared objective* means
pricing competitively. Naming the classes after the benchmark price they aim at
removes the ambiguity. The non-collusive control is the competitive one.

---

## 2026-10-24: a seed draws the same agent randomness in every condition

**Decision:** Each agent's RNG is seeded from `(seed, agent_index)` only, never
from the condition.

**Why:** It makes conditions **paired** across seeds. If the condition fed the
seed, the contrast between two cells at the same seed would inherit noise that
has nothing to do with the manipulation, inflating the variance of exactly the
effects H1–H3 are about. Enforced by
`tests/test_determinism.py::test_seed_pairs_conditions_rather_than_decorrelating_them`.

**Consequence to keep in mind:** the logit-Bertrand market is deterministic
given prices, so with `demand_noise_sd = 0` and zero-jitter scripted agents the
seed changes nothing. Seeds in this environment vary *agent-side* randomness
(scripted jitter, or model sampling), which is the behaviour the tests assert.

---

## 2026-10-24: episode summaries are namespaced by `run_id`

**Decision:** Episode summaries are written to `data/episodes/<run_id>/`, and
`--no-resume` refuses to run when a `run_id` already has data.

**Why:** This fixes a bug found while testing the dry pilot. The resume key is
`(condition_id, seed)`, which is *not* unique across sweeps. Two sweeps sharing
`data/episodes/` meant the second one skipped all 20 of its episodes and
**reported success while silently serving the first sweep's results** — a
different agent policy, reported as if it were the one requested. Section 12
names that failure mode ("identical results across conditions"). Appending to an
existing turn log would likewise duplicate turns and corrupt every per-turn
statistic, hence the refusal. Regression test:
`tests/test_sweep.py::test_two_run_ids_do_not_share_results`.

---

## 2026-10-24: `tasks.py` is the build entry point, not the Makefile

**Decision:** All task logic lives in `tasks.py`. The `Makefile` and `make.cmd`
are thin shims that delegate to it.

**Why:** Section 3 rule 9 asks for a single `make test` / `make pilot` entry
point, but GNU make is not installed on this Windows machine and is not easily
assumed. One implementation with two shims keeps `make test` working for anyone
who does have make, without the two drifting apart.

---

## 2026-10-24: pathological categories are tested before the index bands

**Decision:** `classify()` checks `below_cost` and `over_priced` *before* the
Nash/partial/monopoly bands.

**Why:** Prices above the cartel level and prices below cost both depress
profit, so both produce a low collusion index. Reading that as "competitive" is
precisely the misreading Section 7 warns about. Precedence makes the pathology
win, and `is_pathological` marks those episodes for separate reporting rather
than pooling them into the main analysis.

---

## 2026-10-24: scripted agents still get the full rendered prompt

**Decision:** Scripted agents receive the rendered prompt and emit replies in
the real text format, even though they compute their price from the
`Observation` and ignore the prompt entirely.

**Why:** It keeps one code path. The turn log, the parser, the round-trip test
and the probe layer all exercise the same render → parse → log pipeline whether
a model or a script is acting, so Stage 1 genuinely tests the machinery Stage 3
will use. It also means the round-trip test is checking real logged prompts.

---

## 2026-10-24: judge model shares the agent's family (flagged limitation)

**Decision:** `experiment.yaml` provisionally sets both AGENT_MODEL and
JUDGE_MODEL to a local Qwen3-8B.

**Why:** The `$0` cap rules out an API judge, and the ungated local models that
fit 24 GB comfortably are limited. Judging transcripts with the same family that
produced them risks self-preference bias, which matters because H5 is a claim
about the *gap* between text and action scores.

**How to apply:** Recorded as a limitation for `prereg.md` and raised at Gate 1.
An independent judge family (Llama-3.1-8B-Instruct, Gemma-2-9B) needs an HF
token for gated weights; Phi-4 at 14B needs quantising to fit alongside. The
human should pick before the pilot.

---

## 2026-10-24: minor layout deviations from Section 4

**Decision:** `README.md`, `DECISIONS.md`, `prereg.md`, `Makefile` and
`pyproject.toml` sit at the repository root, with the Python package under
`collusion/`. `gate1_report.py` lives in `collusion/runner/`.

**Why:** Section 4's tree nests everything under `collusion/`, but the
repository is already the project, so an extra level would put the README one
directory down from where anyone looks for it. Package-internal paths
(`envs/pricing`, `metrics/`, `analysis/`, `probes/`, `runner/`) follow Section 4
exactly. `gate1_report.py` reports on a run rather than analysing sweep output,
so it sits with the runner instead of in `analysis/` (reserved for Stage 4).
