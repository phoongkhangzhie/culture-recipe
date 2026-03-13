#!/bin/bash
# Usage: ./scripts/eval_blend.sh <model-path>
#   e.g. ./scripts/eval_blend.sh /nlp/scr/phoongkz/models/meta-llama-Llama-3.1-8B-Instruct
#
# Optional env var overrides (defaults shown):
#   MODEL_TYPE="instruct"  # "instruct" or "base"
#   COUNTRIES=""           # space-separated country codes, e.g. "US KR CN" (default: all)
#   MAX_EXAMPLES=""        # cap examples for debugging (default: all)
#   NUM_GPUS=1             # tensor parallel GPUs
#   BATCH_SIZE=512         # vLLM generation batch size
#   MAX_NEW_TOKENS=50      # tokens to generate per example

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <model-path>"
    exit 1
fi

MODEL="$1"
MODEL_TYPE="${MODEL_TYPE:-instruct}"
COUNTRIES="${COUNTRIES:-Indonesia}"
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-512}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-50}"

echo "[$(date)] Model:  ${MODEL}"
echo "[$(date)] GPUs:   ${NUM_GPUS}"

CMD=(
    python evaluation/blend/eval.py
    --model                "${MODEL}"
    --model-type           "${MODEL_TYPE}"
    --tensor-parallel-size "${NUM_GPUS}"
    --batch-size           "${BATCH_SIZE}"
    --max-new-tokens       "${MAX_NEW_TOKENS}"
)

if [ -n "${COUNTRIES:-}" ]; then
    # shellcheck disable=SC2086
    CMD+=(--countries ${COUNTRIES})
fi

if [ -n "${MAX_EXAMPLES:-}" ]; then
    CMD+=(--max-examples "${MAX_EXAMPLES}")
fi

"${CMD[@]}"
