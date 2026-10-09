#!/usr/bin/env bash
# =============================================================================
# start_vllm_server.sh
# =============================================================================
# Start the vLLM inference service (OpenAI-compatible API).
# Usage:
#   ./scripts/start_vllm_server.sh [MODEL_PATH] [PORT] [GPU_IDS]
#
# Examples:
#   ./scripts/start_vllm_server.sh Qwen/Qwen2.5-7B-Instruct 8000 0,1,2,3
#   ./scripts/start_vllm_server.sh /path/to/local/model 8001 0
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Default parameters
# -----------------------------------------------------------------------------
DEFAULT_MODEL="Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PORT="8000"
DEFAULT_GPUS="0"

MODEL="${1:-$DEFAULT_MODEL}"
PORT="${2:-$DEFAULT_PORT}"
GPUS="${3:-$DEFAULT_GPUS}"

# -----------------------------------------------------------------------------
# Environment check
# -----------------------------------------------------------------------------
echo "[start_vllm_server] Starting vLLM service"
echo "  Model: ${MODEL}"
echo "  Port: ${PORT}"
echo "  GPU:  ${GPUS}"

if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

if ! python3 -c "import vllm" 2>/dev/null; then
    echo "Error: vllm package is not installed, please run first: pip install vllm"
    exit 1
fi

# -----------------------------------------------------------------------------
# Start service
# -----------------------------------------------------------------------------
# Set visible GPUs
export CUDA_VISIBLE_DEVICES="${GPUS}"

# Compute tensor-parallel-size (based on GPU count)
IFS=',' read -ra GPU_ARRAY <<< "${GPUS}"
TP_SIZE="${#GPU_ARRAY[@]}"

echo "[start_vllm_server] Tensor Parallel Size: ${TP_SIZE}"

# Start vLLM (in the background, logs written to a file)
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/vllm_server_$(date +%Y%m%d_%H%M%S).log"

echo "[start_vllm_server] Log file: ${LOG_FILE}"

# Note: the following parameters can be adjusted according to model size and GPU memory
python3 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --dtype bfloat16 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    >> "${LOG_FILE}" 2>&1 &

SERVER_PID=$!
echo "[start_vllm_server] Server PID: ${SERVER_PID}"
echo "${SERVER_PID}" > "${LOG_DIR}/vllm_server.pid"

# Wait for the service to be ready
echo "[start_vllm_server] Waiting for the service to be ready..."
for i in {1..60}; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        echo "[start_vllm_server] Service ready: http://localhost:${PORT}/v1"
        echo "[start_vllm_server] Test command:"
        echo "  curl http://localhost:${PORT}/v1/models"
        exit 0
    fi
    sleep 1
done

echo "[start_vllm_server] Warning: service startup timed out, please check the log: ${LOG_FILE}"
exit 1
