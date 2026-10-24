#!/usr/bin/env bash
# Start the shared vLLM server for Qwen3-8B inside WSL2. Settings are recorded in
# DECISIONS.md; change them here only, and re-record.
#
#   bash /mnt/c/Users/leobo/OneDrive/Documents/GitHub/apart-collusion-research-sprint/pricing/serve/start_vllm.sh
#
# Reachable from Windows at http://localhost:8000/v1 (WSL2 localhost forwarding).
set -euo pipefail

ENV_DIR="${VLLM_ENV:-$HOME/vllm-env}"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.88}"
MAX_SEQS="${MAX_SEQS:-64}"

# FlashInfer's sampler JIT-compiles a kernel at warm-up and needs nvcc, which
# WSL does not have. The PyTorch sampler applies the same temperature/top_p.
export VLLM_USE_FLASHINFER_SAMPLER=0

# --generation-config vllm: do NOT silently apply the model's generation_config
#   defaults (Qwen3 ships top_k=20 etc.); only the sampling parameters we send
#   are used, and all of those are logged per turn.
# --max-num-seqs 64: 32 concurrent episodes x 2 sellers whose requests are sent
#   in parallel each period.
# --enable-auto-tool-choice --tool-call-parser hermes: for the Track B Colosseum
#   check, whose agents act only via OpenAI `tools`. Requests without `tools`
#   (the pricing game) are unaffected.
vllm serve "$MODEL" \
  --served-model-name "$MODEL" \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --max-num-seqs "$MAX_SEQS" \
  --generation-config vllm \
  --seed 0 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --host 0.0.0.0 \
  --port "$PORT" \
  2>&1 | tee "$HOME/vllm_server.log"
