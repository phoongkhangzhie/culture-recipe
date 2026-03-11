#!/bin/bash
# Usage: ./scripts/finetune.sh <train-file>
#   e.g. ./scripts/finetune.sh prepared_output/japanese_english-train.jsonl
#
# Output dir is derived automatically from the train file name:
#   prepared_output/japanese_english-train.jsonl → checkpoints/japanese_english/
#
# Optional env var overrides (defaults shown):
#   MODEL="meta-llama/Llama-3.1-8B"
#   VAL_FILE=""             # leave empty to skip validation
#   LORA="true"             # "true" = LoRA, "false" = full fine-tuning
#   LORA_R=8
#   LORA_ALPHA=16
#   LORA_DROPOUT=0.1
#   EPOCHS=4
#   BATCH_SIZE=1
#   GRAD_ACCUM=1
#   LR="2e-4"
#   MAX_SEQ_LEN=4096
#   OPTIM="adamw_torch"
#   MERGE_AFTER="false"     # "true" = merge LoRA adapter after training

set -e

if [ $# -lt 1 ]; then
    echo "Usage: $0 <train-file>"
    exit 1
fi

TRAIN_FILE="$1"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
VAL_FILE="${VAL_FILE:-}"
LORA="${LORA:-true}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_DROPOUT="${LORA_DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
LR="${LR:-2e-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
OPTIM="${OPTIM:-adamw_torch}"
MERGE_AFTER="${MERGE_AFTER:-false}"

mkdir -p logs

echo "[$(date)] Model:      ${MODEL}"
echo "[$(date)] Train file: ${TRAIN_FILE}"
echo "[$(date)] LoRA:       ${LORA} (r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT})"
echo "[$(date)] Epochs:     ${EPOCHS}  LR: ${LR}  Batch: ${BATCH_SIZE}  Accum: ${GRAD_ACCUM}"

# ---------------------------------------------------------------------------
# 1. Build the training command
# ---------------------------------------------------------------------------
CMD=(
    python main.py finetune train
    --model         "${MODEL}"
    --train-file    "${TRAIN_FILE}"
    --epochs        "${EPOCHS}"
    --per-device-batch-size   "${BATCH_SIZE}"
    --grad-accumulation-steps "${GRAD_ACCUM}"
    --learning-rate "${LR}"
    --max-seq-length "${MAX_SEQ_LEN}"
    --optim         "${OPTIM}"
    --bf16
)

if [ -n "${VAL_FILE}" ]; then
    CMD+=(--val-file "${VAL_FILE}")
fi

if [ "${LORA}" = "true" ]; then
    CMD+=(--lora --lora-r "${LORA_R}" --lora-alpha "${LORA_ALPHA}" --lora-dropout "${LORA_DROPOUT}")
fi

# ---------------------------------------------------------------------------
# 2. Run training (output dir is derived from train file name)
# ---------------------------------------------------------------------------
echo "[$(date)] Starting training…"
"${CMD[@]}"

# Derive the output dir the same way train.py does, for the merge step
STEM=$(basename "${TRAIN_FILE}" .jsonl)
STEM="${STEM%-train}"
OUTPUT_DIR="checkpoints/${STEM}"
echo "[$(date)] Training complete. Checkpoint saved to ${OUTPUT_DIR}"

# ---------------------------------------------------------------------------
# 3. Optional: merge LoRA adapter immediately after training
# ---------------------------------------------------------------------------
if [ "${LORA}" = "true" ] && [ "${MERGE_AFTER}" = "true" ]; then
    MERGED_DIR="${OUTPUT_DIR}-merged"
    echo "[$(date)] Merging LoRA adapter → ${MERGED_DIR}"
    python main.py finetune merge-lora \
        --base-model  "${MODEL}" \
        --lora-dir    "${OUTPUT_DIR}" \
        --output-dir  "${MERGED_DIR}"
    echo "[$(date)] Merge complete."
fi

echo "[$(date)] Done."
