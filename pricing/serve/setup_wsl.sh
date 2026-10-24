#!/usr/bin/env bash
# One-time setup inside WSL2 Ubuntu: vLLM + Qwen3-8B (bf16) weights.
#
#   bash /mnt/c/Users/leobo/OneDrive/Documents/GitHub/apart-collusion-research-sprint/pricing/serve/setup_wsl.sh
#
# The CUDA driver comes from the Windows NVIDIA driver (WSL passes it through);
# no CUDA toolkit is installed in Linux. Weights go to the Linux filesystem
# (~/.cache/huggingface), which loads far faster than /mnt/c.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${VLLM_ENV:-$HOME/vllm-env}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"

echo "== GPU visible from WSL?"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv

echo "== base packages"
sudo apt-get update -qq
sudo apt-get install -y -qq curl ca-certificates build-essential python3-dev

if ! command -v uv >/dev/null 2>&1; then
  echo "== installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "== venv at $ENV_DIR (python 3.12)"
uv venv "$ENV_DIR" --python 3.12 --seed
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "== installing vllm (pulls a CUDA build of torch)"
uv pip install vllm huggingface_hub

python - <<'PY'
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("vllm", vllm.__version__)
assert torch.cuda.is_available(), "CUDA not visible to torch inside WSL"
print("device:", torch.cuda.get_device_name(0))
PY

echo "== downloading $MODEL (bf16 safetensors, ~16 GB)"
HF_XET_HIGH_PERFORMANCE=1 hf download "$MODEL"

echo "== verifying weight files against the Hub's sha256"
python "$HERE/verify_weights.py" "$MODEL"

echo "setup complete. Start the server with: bash $HERE/start_vllm.sh"
