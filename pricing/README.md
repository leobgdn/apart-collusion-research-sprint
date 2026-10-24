# Track A: repeated pricing game (`pricing/`)

Which task features make LLM agents collude, and at what cost to the task can
each be restricted? Track A plays a repeated logit-Bertrand duopoly with two
Qwen3-8B sellers served locally (vLLM in WSL2, RTX 3090), and measures
collusion from prices and profits (primary) and from what the agents write
(secondary).

Specs, in order of precedence: `TRACKA_ADDENDUM.md` > `PHASE1.md` > `SPEC.md`.
Every design decision is in `DECISIONS.md`; the analysis plan is `prereg.md`
(draft until Gate 2).

## Layout

```
collusion/
  envs/pricing/      environment, economics, conditions (factors), prompts
  agents/            scripted controls, JSON reply parser, LLM agent (vLLM client)
  metrics/           action-level index; crude keyword text baseline
  runner/            run_episode, run_sweep (batches, wall-clock guard), replay,
                     throughput, gate1_report
  analysis/          pilot_report (Gate 2 numbers from the logs)
  config/experiment.yaml
serve/               WSL setup, vLLM start script, weight verification, server check
tests/               acceptance tests (python tasks.py test)
reports/             gate1.md, gate2.md and their generated data
data/logs/           turn logs (JSONL, one line per agent turn)
data/episodes/       episode summaries, one JSON per episode, per run_id
```

## Reproduce

Windows side (this folder):

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install numpy scipy pandas statsmodels matplotlib scikit-learn pyyaml pytest requests transformers tokenizers jinja2 huggingface_hub
python tasks.py test        # acceptance suite
python tasks.py gate1       # reports/gate1.md
```

Model server (WSL2 Ubuntu; one-time `wsl --install -d Ubuntu-24.04` as admin):

```
bash /mnt/c/Users/<you>/OneDrive/Documents/GitHub/apart-collusion-research-sprint/pricing/serve/setup_wsl.sh
bash /mnt/c/Users/<you>/OneDrive/Documents/GitHub/apart-collusion-research-sprint/pricing/serve/start_vllm.sh
```

Then, back on Windows:

```
python tasks.py check_server   # thinking-off verified, endpoint -> ~/collusion_shared/endpoint.txt
python tasks.py throughput     # <=10 min dedicated window, holds ~/collusion_shared/GPU_BUSY
python tasks.py pilot          # 9 cells x 5 seeds x 80 rounds, 90-minute cap
python tasks.py gate2_data -- --run-id pilot_v1 --hard-stop 2026-10-25T12:00
```

The main sweep is gated on Gate 2 approval
(`python -m collusion.runner.run_sweep --llm --main --i-have-gate2-approval ...`).

## Data contract

Turn log (one JSON line per agent turn): SPEC Section 5 fields, plus
`rendered_prompt` (the chat-templated string), `call_info` (request seed, server
and local prompt token counts, latency, finish reason), and the Track A probe
fields `rolling_delta`, `next_rolling_delta`, `in_metric_window`,
`price_window`. Episode summary: SPEC Section 5 fields plus action metrics,
benchmarks, `batch`, and the keyword `text_baseline` (`text_collusion_score`
stays null until the LLM judge runs).
