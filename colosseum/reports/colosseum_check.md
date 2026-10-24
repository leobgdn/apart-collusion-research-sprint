# Colosseum feasibility check (Track B) — PARTIAL

Written 2026-10-24 17:25 local; re-verified 17:25–17:27 after moving the folder to
`apart-collusion-research-sprint/colosseum/` (venv rebuilt, all mock runs repeated with identical
results, sprint test suite 78/78 still passing). Everything below comes from code I read or ran in
`colosseum/` (fresh clone of `umass-ai-safety/colosseum` @ `0048f1b`, Terrarium
`terrarium-agents==0.2.1`). **No real model call was made:** `~/collusion_shared/endpoint.txt`
does not exist, nothing listens on `localhost:8000`, and the GPU shows ~1.1 GB in use (no model
loaded) — still true after the move. Task 3 (real episode + mini-pilot) is therefore **not run**; it is one command away
(see the end).

## 1. Verdict

**CONDITIONAL.** The pipeline installs, runs end to end, logs every exact prompt and recomputes
regret from its logs, and two factors are native. Throughput and the collusion rate (criteria 2–3)
cannot be judged until the shared Qwen3-8B endpoint exists. That endpoint must also be launched
with tool calling enabled. Colosseum agents act only through OpenAI `tools`, so vLLM needs
`--enable-auto-tool-choice --tool-call-parser hermes`.

## 2. Criteria

| # | criterion | status | evidence |
|---|---|---|---|
| 1 | installs + one local episode end to end with local Qwen3-8B | **PARTIAL**: install and pipeline pass; real-model run unknown | `uv sync` clean, 120 packages, no warnings after the move (`reports/install.log`). 3 + 6 mock episodes through the native `vllm` provider: all `status=complete`, 0 failures, `metrics.json` + `optimal_summary.json` written. No real Qwen3 call yet. |
| 2 | throughput ≥ ~120 episodes / ~10 GPU-h at ¼–⅓ share | **UNKNOWN** | Needs a measurement. Measured from the mock: **20 LLM calls per episode** (4 agents × 3 turns, ~1.7 steps each); about 24–26k prompt tokens per episode by a chars/4 estimate, not real tokenization. Bar: 10 GPU-h × ¼–⅓ = 2.5–3.3 h of full-server time for 120 episodes, so **≤ 75–100 s of full-server time per episode**, or ≤ 300 s of wall time per episode at a ¼ share run serially. |
| 3 | action-level collusion 15–70% in ≥ 1 condition | **UNKNOWN**, with a caveat on the metric | No real data. The mock agents pick tasks at random, yet the regret-based coalition advantage swung **−18 to +21** per episode, and was > 0 in **2/2 no-channel baselines**. At threshold 0 the binary label is near a coin flip under random play, so the 15–70% band means nothing until the threshold is set against the no-channel null. |
| 4 | exact prompt per turn in logs (or via proxy) + regret recomputable from logs | **PASS** (mock) | Native: `agent_prompts.json` holds the system and user prompt per agent turn. It misses later steps inside a turn (tool results appended to the context). `scripts/proxy.py` logs the full `messages` + `tools` of **every** request: 60/60 mock requests had both. Regret recomputed from logs with `compute_sequential_regret.py` (parses `agent_prompts.json`): coalition mean regret 12.22 vs `metrics.json` 12.2200, non-coalition 27.805 vs 27.8028 (logged costs are rounded to 2 dp). |
| 5 | ≥ 2 manipulable factors | **PASS** | `secret_channel_enabled` (channel), `prompt_variant` (coalition incentive: control/simple/deception/structured/aggressive), `topology` (path/star/complete/ER/WS/BA), plus `colluder_count` and `num_agents`. Path topology verified on the mock (3–4 blackboards vs 2 for complete). |

## 3. Repo map

| item | finding (source) |
|---|---|
| Scenarios | 5 DCOP environments in Terrarium: JiraTicket (collusion default), Hospital, MeetingScheduling, PersonalAssistant, SmartGrid (`terrarium/environments/dcops/`, plus `experiments/collusion/choice/*_choice_env.py` variants). Experiment families: `collusion`, `persuasion/{collusion,hospital}`, `network_influence`, `agent_misalignment`. Regret and exact-optimum tooling (`compute_jira_optimal.py`) exists **only for Jira**. |
| Agents / coalition | `num_agents` set per sweep (configs use 3, 4, 6). Coalition = `colluder_count` agents, chosen by `experiment.collusion.colluder_selection` (`random`, seeded, or `first`) (`run.py:_select_colluders`). Colluders get an injected "SECRET COALITION INSTRUCTIONS" block in the user prompt, but **only when the channel is on**. With it off they are labelled colluders for the metrics yet are prompted like everyone else. |
| Secret channel | `secret_channel_enabled: [false, true]`. When true and ≥ 2 colluders, `protocol.generate_comm_network(colluders, …)` adds a hidden blackboard (`run.py` ~l.403). The runner **forces `prompt_variant=control` when the channel is off** and drops other combinations (`run.py:256, 766`), so incentive is nested inside channel-on. |
| Topologies | `path, star, complete, erdos_renyi, watts_strogatz, barabasi_albert` (`terrarium/networks/`), set by `sweeps[].topologies`. `consolidate_channels: true` merges 3+ cliques into one board. |
| Multi-round history | **No cross-round history.** `run.py` forces `max_iterations=1`, so each episode is one-shot: `planning_rounds` (2–4) of blackboard posts, then one execution turn where each agent calls `assign_task` once. Inside an episode every prompt carries the full blackboard transcript so far. Each agent turn starts a fresh context (`BaseAgent._multi_step_response_generation` → `init_context`), and nothing carries over between episodes. |
| Metrics | `experiments/collusion/metrics.py:compute_collusion_metrics`: `mean_regret`, `coalition_mean_regret`, `noncoalition_mean_regret` (one-shot unilateral best-response regret, brute-forced over all tasks + skip), `coalition_advantage_mean` (= coalition mean **reward** − non-coalition mean reward), `system_regret(_ratio)` (vs an upper bound), `colluder_posts_secret_rate`, `reward_std`. Exact optimum: `compute_jira_optimal.py`. Availability-aware regret: `compute_sequential_regret.py`. Text judge: `judge_blackboards.py` (OpenAI by default). |
| Logging | Per run: `run_config.json`, `agent_prompts.json/.md` (system and user prompt per turn), `agent_trajectories.json`, `agent_turns.json` (with token `usage`), `tool_events.json`, `blackboards.json` + `blackboard_*.txt`, `metrics.json`, `final_summary.json`, `communication_network.png`. |
| LLM client / base URL | Providers: openai, anthropic, gemini, together, fireworks, grok, foundry, vllm (`terrarium/utils.py:get_client_instance`). `provider: vllm` with `auto_start_server: false` + `models[].base_url` connects to any running OpenAI-compatible server and checks `/models` lists the checkpoint name (verified with the mock). **The VLLMClient payload has no field for `chat_template_kwargs`**, so thinking-off cannot be set from config (see §5). |

## 4. Factor mapping

| shared factor | Colosseum | status |
|---|---|---|
| Secret communication channel | `secret_channel_enabled` | **native** |
| Coalition incentive | `prompt_variant` (`control` vs `simple`, or stronger ones) | **native**, but nested: only exists with channel on, so the design is 3 cells (off / on+control / on+simple), not a 2×2 |
| Topology | `topologies` | **native** |
| Interaction history (window) | none: one-shot episodes | **missing.** Adding repeated rounds would mean rewriting the runner loop and env state reset. Not a small patch. |

## 5. Code changes needed for a replication

1. **Logging proxy + thinking-off: done, tested on the mock** (`scripts/proxy.py`). It sits in front of the endpoint, logs every full request and response as JSONL, injects `chat_template_kwargs={"enable_thinking": false}`, caps concurrency at 4, and pauses while `GPU_BUSY` exists. Zero changes to Colosseum. Not yet verified against real vLLM (does it honour the kwarg? does any `<think>` text leak?). About 15 min once the endpoint exists.
2. **Server flags (Track A's side):** tool calling as above. If they can't be added, the fallback is a prompt-parsing client subclass, about 2–3 h and a deviation from the native harness.
3. **Collusion label threshold** calibrated on the no-channel null, plus a summary script (`scripts/summarize.py` exists and reports both advantage definitions). About 30 min.
4. **Text-vs-action divergence:** Colosseum's judge calls OpenAI. Pointing it at the local endpoint is a client swap in `judge_blackboards.py`, about 1 h, not done. (Same-family judge bias applies.)
5. **History factor:** not feasible this weekend. Drop it from the replication.

Honest total, excluding the history factor: **~2–3 h** of work if the server flags are right, **~5 h** if they are not.

## 6. Measured numbers

| quantity | value |
|---|---|
| Real episode wall-clock | **not measured** (no endpoint) |
| Tokens / episode | mock: 20 calls, ~24–26k prompt tokens (chars/4 estimate). Real: not measured |
| Parse-failure rate | mock: 0 invalid `assign_task` calls after fixing the mock's ID regex. Real: not measured. Note the env returns `retry` for unknown task IDs, so this is countable from `tool_events.json` |
| Thinking blocks | proxy flags `<think>` / `reasoning_content`. Real: not measured |
| Mini-pilot collusion by condition | **not run.** (Mock dry run, meaningless as behaviour: regret-advantage > 0 in baseline 2/2, channel+simple 1/2, path+simple 2/2. This shows the null is noisy.) |

## 7. Risks and unknowns

- **Tool-calling flags on the shared vLLM server** (top risk; see Verdict).
- Whether Qwen3-8B with thinking off calls tools reliably (Hermes format). Unknown.
- **Metric noise:** random play produces |coalition advantage| up to ~20. A small pilot (2 seeds per cell) can't separate signal from that noise. The replication needs ≥ 10 seeds per cell.
- The two "coalition advantage" definitions differ: Colosseum's `coalition_advantage_mean` is reward-based, while the apart-sprint prereg uses non-coalition − coalition **regret**. Pick one before the pilot.
- Terrarium 0.2.1 ≠ the paper's 0.1.1 (the README lists behaviour changes), so paper numbers are not directly comparable.
- Colosseum is single-shot, so it doesn't cover the pricing game's repeated-play dynamics. The replication tests channel and topology only.
- Observed, not touched: `GitHub/apart-collusion-research-sprint` (HEAD `68c2f46`) already pivoted to Colosseum on Fireworks using `GitHub/colosseum`. That conflicts with this prompt's "local only" rule. The human should decide which Colosseum track is live.
- Not checked: Windows-specific issues under real load, other envs (Hospital etc.), the judge pipeline.

## 8. Recommendation (provisional)

If criteria 1–3 pass on the real pilot: **3 cells** (complete/no channel, complete/channel+simple, path/channel+simple) × **12 seeds** = 36 episodes; n=4, 2 colluders, 2 planning rounds, 8 tasks. Add complete/channel+control (+12) for a clean channel-only contrast, giving 48. GPU-hours = 48 × (measured per-episode full-server seconds) ÷ 3600. Fill this in from the one-episode measurement. If one real episode takes more than ~100 s of full-server time, or tool calls fail at more than 5%, **NO-GO**.

## What I need from the human

1. Start the shared endpoint (or confirm Track A will), **with `--enable-auto-tool-choice --tool-call-parser hermes`**, and write `~/collusion_shared/endpoint.txt` (e.g. `http://127.0.0.1:8000/v1`, with the served name `Qwen/Qwen3-8B`).
2. Then run, from `apart-collusion-research-sprint/colosseum/`:
   `bash scripts/run_real.sh one` → then `bash scripts/run_real.sh pilot`
   (each refuses to start without `endpoint.txt` or while `GPU_BUSY` exists). Or ask me to run them.
