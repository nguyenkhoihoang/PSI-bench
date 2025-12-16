#!/usr/bin/env bash
set -euo pipefail

# Run emotional comparison for all supported datasets (esc, hope, annomi).
# Usage: ./run_emotional_compare_all.sh
# Optional env overrides:
#   PSI=eeyore
#   DATA_DIR=/work/hdd/bfjp/data/synthetic/test/eeyore/hosted_vllm_openai_gpt-oss-120b/
#   MAX_SESSIONS=100     # optional limit for speed

MAX_SESSIONS=4
SHOW_FLAG=""
MAX_SESSIONS_ARG=""
if [[ -n "${MAX_SESSIONS:-}" ]]; then
  MAX_SESSIONS_ARG="--max-sessions ${MAX_SESSIONS}"
fi

PSI="${PSI:-eeyore}"
DATA_DIR="${DATA_DIR:-/work/hdd/bfjp/data/synthetic/test/eeyore/hosted_vllm_openai_gpt-oss-120b/}"

datasets=(esc hope annomi)

for ds in "${datasets[@]}"; do
  echo "=== Running dataset: ${ds} (psi=${PSI}) ALL sessions ==="
  python -m psibench.eval.emotional_compare --dataset "${ds}" --psi "${PSI}" --data-dir "${DATA_DIR}" --all-sessions ${SHOW_FLAG} ${MAX_SESSIONS_ARG}
  echo
done
