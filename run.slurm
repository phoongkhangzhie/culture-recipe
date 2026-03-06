#!/bin/bash
#SBATCH --job-name=culture-recipe
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/slurm-%j.log
#SBATCH --error=logs/slurm-%j.log

set -e

VLLM_PORT=8000
VLLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
VLLM_LOG="logs/vllm-${SLURM_JOB_ID}.log"
WAIT_TIMEOUT=1800   # seconds before giving up (30 min covers large model loads)
WAIT_INTERVAL=10    # poll interval in seconds
READY_MARKER="Application startup complete."

mkdir -p logs

# ---------------------------------------------------------------------------
# 1. Start vLLM server in background, capturing its output separately
# ---------------------------------------------------------------------------
echo "[$(date)] Starting vLLM (model: ${VLLM_MODEL}, port: ${VLLM_PORT})"
echo "[$(date)] vLLM log: ${VLLM_LOG}"

vllm serve "${VLLM_MODEL}" \
    --port "${VLLM_PORT}" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    > "${VLLM_LOG}" 2>&1 &

VLLM_PID=$!

# ---------------------------------------------------------------------------
# 2. Wait for Uvicorn's "Application startup complete." in the vLLM log.
#    This fires only after all startup handlers (including model loading)
#    have finished — more reliable than polling /health.
# ---------------------------------------------------------------------------
echo "[$(date)] Waiting for vLLM to finish loading model..."

elapsed=0
while true; do
    # Check if the ready marker has appeared in the log
    if grep -q "${READY_MARKER}" "${VLLM_LOG}" 2>/dev/null; then
        break
    fi

    # Abort early if the vLLM process itself has died
    if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM process (PID ${VLLM_PID}) exited unexpectedly."
        echo "[$(date)] Last 20 lines of vLLM log:"
        tail -20 "${VLLM_LOG}" || true
        exit 1
    fi

    if [ "${elapsed}" -ge "${WAIT_TIMEOUT}" ]; then
        echo "[$(date)] ERROR: vLLM did not become ready within ${WAIT_TIMEOUT}s."
        kill "${VLLM_PID}" 2>/dev/null
        exit 1
    fi

    sleep "${WAIT_INTERVAL}"
    elapsed=$((elapsed + WAIT_INTERVAL))
done

echo "[$(date)] vLLM is ready (waited ${elapsed}s)."

# ---------------------------------------------------------------------------
# 3. Run the pipeline  — pass any extra args via sbatch --export or hardcode
# ---------------------------------------------------------------------------
python main.py \
    --culture "${CULTURE:-Japanese}" \
    --dimension "${DIMENSION:-guest_hospitality}" \
    --output-dir "${OUTPUT_DIR:-./output}" \
    --verbose

# ---------------------------------------------------------------------------
# 4. Shut down vLLM cleanly
# ---------------------------------------------------------------------------
echo "[$(date)] Pipeline done. Stopping vLLM (PID ${VLLM_PID})."
kill "${VLLM_PID}"
wait "${VLLM_PID}" 2>/dev/null || true
echo "[$(date)] Done."
