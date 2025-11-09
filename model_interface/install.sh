#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VLLM_TORCH_BACKEND="${VLLM_TORCH_BACKEND:-cu124}"

echo "=== Installing/Upgrading pip and uv ==="
python -m pip install --upgrade pip
python -m pip install --upgrade uv

if ! command -v uv >/dev/null 2>&1; then
    echo "uv command is not available even after installation. Aborting." >&2
    exit 1
fi

echo "=== Installing vLLM (torch backend: $VLLM_TORCH_BACKEND) ==="
uv pip install vllm --torch-backend="$VLLM_TORCH_BACKEND"


# Install the model provider dependencies
echo "=== Installing model provider dependencies ==="
pip install openai
pip install google-genai

echo "=== Done. ==="
