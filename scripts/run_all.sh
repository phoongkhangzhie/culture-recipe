#!/bin/bash
# Usage: ./run.sh <culture> <output-dir>
#   e.g. ./run.sh Japanese ./output/japanese

set -e

# if [ $# -lt 2 ]; then
#     echo "Usage: $0 <culture> <output-dir>"
#     exit 1
# fi

TENSOR_PARALLEL="$1"

VLLM_PORT=8000
VLLM_MODEL="/nlp/scr/phoongkz/models/Qwen-Qwen3-30B-A3B-Instruct-2507"
VLLM_LOG="logs/vllm-$$.log"
WAIT_TIMEOUT=2700   # seconds before giving up (45 min covers large model loads)
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
    --tensor-parallel "${TENSOR_PARALLEL}" \
    --dtype bfloat16 \
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
    if grep -q "${READY_MARKER}" "${VLLM_LOG}" 2>/dev/null; then
        break
    fi

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
# 3. Run the pipeline
# ---------------------------------------------------------------------------
python main.py generate \
    --culture "Japanese" --all-dimensions --output-dir "./output/japanese_english" --trace --verbose

python main.py generate \
    --culture "Japanese" --all-dimensions --output-dir "./output/japanese_english_insider" --implicit-culture --trace --verbose

python main.py generate \
    --culture "Indonesian" --all-dimensions --output-dir "./output/indonesian_english" --trace --verbose

python main.py generate \
    --culture "Indonesian" --all-dimensions --output-dir "./output/indonesian_english_insider" --implicit-culture --trace --verbose

python main.py generate \
    --culture "Korean" --all-dimensions --output-dir "./output/korean_english" --trace --verbose

python main.py generate \
    --culture "Korean" --all-dimensions --output-dir "./output/korean_english_insider" --implicit-culture --trace --verbose

python main.py generate \
    --culture "Malaysian" --all-dimensions --output-dir "./output/malaysian_english" --trace --verbose

python main.py generate \
    --culture "Malaysian" --all-dimensions --output-dir "./output/malaysian_english_insider" --implicit-culture --trace --verbose

python main.py generate \
    --culture "Nigerian" --all-dimensions --output-dir "./output/nigerian_english" --trace --verbose

python main.py generate \
    --culture "Nigerian" --all-dimensions --output-dir "./output/nigerian_english_insider" --implicit-culture --trace --verbose

python main.py generate \
    --culture "Thai" --all-dimensions --output-dir "./output/thai_english" --trace --verbose

python main.py generate \
    --culture "Thai" --all-dimensions --output-dir "./output/thai_english_insider" --implicit-culture --trace --verbose

python main.py generate \
    --culture "Vietnamese" --all-dimensions --output-dir "./output/vietnamese_english" --trace --verbose

python main.py generate \
    --culture "Vietnamese" --all-dimensions --output-dir "./output/vietnamese_english_insider" --implicit-culture --trace --verbose


# ---------------------------------------------------------------------------
# 4. Shut down vLLM cleanly
# ---------------------------------------------------------------------------
echo "[$(date)] Pipeline done. Stopping vLLM (PID ${VLLM_PID})."
kill "${VLLM_PID}"
wait "${VLLM_PID}" 2>/dev/null || true
echo "[$(date)] Done."
