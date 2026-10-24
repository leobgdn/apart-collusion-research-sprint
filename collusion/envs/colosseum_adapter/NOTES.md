# Plan A recon: Colosseum

Findings from the Section 6.1 check, recorded so the Plan A/B decision can be
revisited without repeating the investigation. No adapter code has been written.

## What it is

- Repo: <https://github.com/umass-ai-safety/colosseum>
- Paper: *Colosseum: Auditing Collusion in Cooperative Multi-Agent Systems*,
  arXiv:2602.15198, NeurIPS 2026 Evaluations & Datasets track.
- Built on the Terrarium agent framework (`terrarium-agents==0.2.1`); the paper's
  numbers come from tag `paper-v0.1.1`.
- Has a dedicated `experiments/collusion/` with exactly the metrics Section 8
  refers to for Plan A: one-shot best-response regret (`metrics.py`,
  `mean_regret` / `coalition_mean_regret` / `noncoalition_mean_regret`), an
  availability-aware sequential variant (`compute_sequential_regret.py`), an
  exact per-run optimum (`compute_jira_optimal.py`), and a secret-blackboard LLM
  judge (`judge_blackboards.py`).
- Its headline finding is close to this project's H5: it reports "collusion on
  paper", where agents plan to collude in text but then pick non-collusive
  actions. That is strong independent motivation for the text-vs-action gap, and
  a reason to keep this repo in the related-work section regardless of which
  environment we run.

## Blocker 1 — no repeated-interaction history

The collusion environments are **one-shot choice tasks**: agents each claim at
most one Jira ticket (or hospital, meeting slot, grid allocation), and
overlapping claims are penalised.

- `simulation.max_iterations: 1`, documented as "the runner always uses 1".
- `simulation.max_planning_rounds: 2` is "rounds in which agents take turns
  posting on blackboards **before choosing a ticket**" — deliberation depth
  within a single decision, not repeated play.
- The regret in `metrics.json` is explicitly "a one-shot best-response regret".

So agents never observe the outcome of a previous round. **H2 (truncating
interaction history decreases collusion) is not testable** without first
building a repeated-game layer on top. This matches the build prompt's warning
that the authors' experiments omit long-horizon effects like memory.

Per Section 6.2 the two options are:

- **(a) Implement history.** Wrap the choice environment in a repeated loop and
  feed back previous rounds' claims and payoffs. Real work: the task has no
  natural repeated form (tickets are consumed), so it needs re-generating the
  problem each round and deciding what carries over. Days, not hours.
- **(b) Replace the history factor.** Drop F2 and use a factor Colosseum already
  supports: `topology` (path / star / complete / Erdos-Renyi / Watts-Strogatz /
  Barabasi-Albert), `num_agents`, or `colluder_count`. This keeps H1 (its
  `secret_channel_enabled` is the positive control, already validated in the
  paper) and H3-analogues via `prompt_variant`, but abandons H2 as written.

## Blocker 2 — local models are Linux-only

README: "For local models via vLLM (**Linux** + CUDA GPU), use
`uv sync --no-install-project --extra vllm`." The `llm.provider` enum is
`openai | anthropic | gemini | together | fireworks | grok | foundry | vllm`.

This machine is Windows 11, and vLLM has no supported Windows build. With the
Section 0 cap of `$0`, every remaining provider is an API that costs money. The
options would be WSL2 + CUDA passthrough + a vLLM install, or API credits — the
first is setup risk inside a weekend, the second breaks the stated budget.

Minor: it pins `uv venv --python 3.11`; this environment has Python 3.13.

## Verdict

Plan B gives H1, H2 and H3 natively, at zero cost, on this machine. Plan A would
cost either the H2 hypothesis or a repeated-game implementation, *and* needs a
Linux/vLLM setup or API budget before a single episode runs.

Recommendation: stay on Plan B. Raised for the human at Gate 1.

A local clone used for this recon is at (scratch, not committed):
`…/scratchpad/colosseum`.
