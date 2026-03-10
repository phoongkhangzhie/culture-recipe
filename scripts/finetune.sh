#!/bin/bash
# Usage: ./finetune.sh <train-file> <output-dir>
#   e.g. ./finetune.sh train.jsonl ./checkpoints/lora
#
# Optional env var overrides (defaults shown):
#   MODEL="Qwen/Qwen2.5-7B-Instruct"
#   VAL_FILE=""             # leave empty to skip validation
#   LORA="true"             # "true" = LoRA, "false" = full fine-tuning
#   LORA_R=16
#   LORA_ALPHA=32
#   EPOCHS=3
#   BATCH_SIZE=2
#   GRAD_ACCUM=8
#   LR="2e-4"
#   MAX_SEQ_LEN=4096
#   MERGE_AFTER="false"     # "true" = merge LoRA adapter after training

set -e

if [ $# -lt 2 ]; then
    echo "Usage: $0 <train-file> <output-dir>"
    exit 1
fi

TRAIN_FILE="$1"
OUTPUT_DIR="$2"

MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
VAL_FILE="${VAL_FILE:-}"
LORA="${LORA:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-2e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"

mkdir -p logs

echo "[$(date)] Model:      ${MODEL}"
echo "[$(date)] Train file: ${TRAIN_FILE}"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] LoRA:       ${LORA}"

# ---------------------------------------------------------------------------
# 1. Build the training command
# ---------------------------------------------------------------------------
CMD=(
    python main.py finetune train
    --model         "${MODEL}"
    --train-file    "${TRAIN_FILE}"
    --output-dir    "${OUTPUT_DIR}"
    --epochs        "${EPOCHS}"
    --per-device-batch-size   "${BATCH_SIZE}"
    --grad-accumulation-steps "${GRAD_ACCUM}"
    --learning-rate "${LR}"
    --max-seq-length "${MAX_SEQ_LEN}"
    --bf16
)

if [ -n "${VAL_FILE}" ]; then
    CMD+=(--val-file "${VAL_FILE}")
fi

if [ "${LORA}" = "true" ]; then
    CMD+=(--lora --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}")
fi

# ---------------------------------------------------------------------------
# 2. Run training
# ---------------------------------------------------------------------------
echo "[$(date)] Starting training…"
"${CMD[@]}"
echo "[$(date)] Training complete. Checkpoint saved to ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 3. Optional: merge LoRA adapter immediately after training
# ---------------------------------------------------------------------------
if [ "${LORA}" = "true" ] && [ "${MERGE_AFTER:-false}" = "true" ]; then
    MERGED_DIR="${OUTPUT_DIR}-merged"
    echo "[$(date)] Merging LoRA adapter → ${MERGED_DIR}"
    python main.py finetune merge-lora \
        --base-model  "${MODEL}" \
        --lora-dir    "${OUTPUT_DIR}" \
        --output-dir  "${MERGED_DIR}"
    echo "[$(date)] Merge complete."
fi

echo "[$(date)] Done."
