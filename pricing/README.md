# Reproducing the pricing-game experiment

Overview and results: [`../README.md`](../README.md). Analysis plan:
[`prereg.md`](prereg.md). All parameters: `collusion/config/experiment.yaml`.

## Setup

Windows side (this folder, Python 3.11+):

```
py -3.13 -m venv .venv
.venv\Scripts\python -m pip install numpy scipy pandas statsmodels matplotlib scikit-learn pyyaml pytest requests transformers tokenizers jinja2 huggingface_hub
python tasks.py test           # test suite; needs no GPU or model
python tasks.py dry_pilot      # scripted agents through the full pipeline
```

Model server (WSL2 Ubuntu 24.04, one GPU with 24 GB):

```
bash serve/setup_wsl.sh        # vLLM venv and Qwen3-8B weights (~16 GB)
bash serve/start_vllm.sh       # OpenAI-compatible server on localhost:8000
python tasks.py check_server   # back on Windows: checks thinking is off
```

## Pipeline

```
# 1. main sweep: 9 cells x 50 seeds x 80 rounds, resumable (about 2.7 GPU-hours)
python -m collusion.runner.run_sweep --llm --main --i-have-gate2-approval --run-id main_v1

# 2. LLM judge over plans and messages
python -m collusion.metrics.llm_judge --run-id main_v1

# 3. activations (inside WSL; the script stops the vLLM server first)
bash serve/run_extraction.sh main_v1

# 4. effects, judge validation, cost/quadrant tables and figures, probes
python tasks.py results
```

Step 4 writes `results/effects.md`, `stage4.md`, `probes.md` (each with a
`.json`), `judge_validation.md` and `results/figures/`. It runs on CPU.

## Data

Generated data is not in git: `data/logs/` (turn logs), `data/episodes/`,
`data/judge/`, `data/activations/`.

- **Turn log**, one JSON line per seller per round: condition and seed, the full
  prompt and raw response, parsed price, plan, insight and message, reward,
  token counts, and the per-request sampling seed and server info.
- **Episode summary**, one JSON per episode: Delta, label, category, joint
  reward, Nash and monopoly benchmarks, parse failures.
