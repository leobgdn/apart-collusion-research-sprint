#!/usr/bin/env bash
# Overnight activation extraction (Stage 5 input), run inside WSL:
#
#   bash /mnt/c/Users/leobo/OneDrive/Documents/GitHub/apart-collusion-research-sprint/pricing/serve/run_extraction.sh main_v1
#
# 1. holds ~/collusion_shared/GPU_BUSY (the Colosseum session waits on it)
# 2. stops the vLLM server and waits until the GPU is free
# 3. extracts the scored-window turns first, then the remaining turns
# 4. releases GPU_BUSY, whatever happened (restart vLLM separately)
set -uo pipefail

RUN_ID="${1:-main_v1}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PRICING="$(dirname "$HERE")"
SHARED=/mnt/c/Users/leobo/collusion_shared
LOG="$HOME/extraction_${RUN_ID}.log"

release() {
  rm -f "$SHARED/GPU_BUSY"
  echo "== GPU_BUSY released $(date). Restart vLLM with serve/start_vllm.sh."
}
trap release EXIT

exec > >(tee -a "$LOG") 2>&1
echo "== extraction for $RUN_ID starting $(date)"
mkdir -p "$SHARED" && touch "$SHARED/GPU_BUSY"

echo "== stopping vLLM"
pkill -f "vllm serve" || true
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  [ "$used" -lt 3000 ] && break
  sleep 2
done
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

# shellcheck disable=SC1091
source "$HOME/vllm-env/bin/activate"
cd "$PRICING"
export PYTHONPATH="$PRICING" HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
# Variable-length batches fragment the CUDA cache; expandable segments avoid it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for subset in window rest; do
  if grep -q token_count_mismatches_vs_server "data/activations/${RUN_ID}_${subset}/meta.json" 2>/dev/null; then
    echo "== subset $subset already complete, skipping"; continue
  fi
  echo "== subset $subset $(date)"
  python -u -m collusion.probes.extract --run-id "$RUN_ID" --subset "$subset" --batch-size 16 \
    || { echo "== extraction FAILED on subset $subset"; exit 1; }
done
echo "== extraction complete $(date)"
