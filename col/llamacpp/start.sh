#!/bin/sh
set -eu

VLM_GGUF_REPO="${VLM_GGUF_REPO:-unsloth/Qwen3-VL-4B-Instruct-GGUF}"
VLM_GGUF_FILE="${VLM_GGUF_FILE:-Qwen3-VL-4B-Instruct-Q5_K_M.gguf}"
VLM_MMPROJ_FILE="${VLM_MMPROJ_FILE:-mmproj-BF16.gguf}"
VLM_MODEL_DIR="${VLM_MODEL_DIR:-/models/qwen3-vl-4b}"

mkdir -p "${VLM_MODEL_DIR}"

if [ ! -f "${VLM_MODEL_DIR}/${VLM_GGUF_FILE}" ]; then
  curl -L --retry 5 --retry-delay 3 \
    -o "${VLM_MODEL_DIR}/${VLM_GGUF_FILE}" \
    "https://huggingface.co/${VLM_GGUF_REPO}/resolve/main/${VLM_GGUF_FILE}"
fi

if [ ! -f "${VLM_MODEL_DIR}/${VLM_MMPROJ_FILE}" ]; then
  curl -L --retry 5 --retry-delay 3 \
    -o "${VLM_MODEL_DIR}/${VLM_MMPROJ_FILE}" \
    "https://huggingface.co/${VLM_GGUF_REPO}/resolve/main/${VLM_MMPROJ_FILE}"
fi

if [ -n "${LLAMA_SERVER_BIN:-}" ]; then
  BIN="${LLAMA_SERVER_BIN}"
elif [ -x "/app/llama-server" ]; then
  BIN="/app/llama-server"
elif [ -x "/llama-server" ]; then
  BIN="/llama-server"
elif command -v llama-server >/dev/null 2>&1; then
  BIN="$(command -v llama-server)"
else
  echo "llama-server binary not found"
  exit 127
fi

exec "${BIN}" \
  -m "${VLM_MODEL_DIR}/${VLM_GGUF_FILE}" \
  --mmproj "${VLM_MODEL_DIR}/${VLM_MMPROJ_FILE}" \
  --host 0.0.0.0 \
  --port 8000 \
  -c "${VLM_CTX_SIZE:-4096}" \
  -ngl "${LLAMA_N_GPU_LAYERS:-0}" \
  --parallel "${LLAMA_PARALLEL:-1}" \
  --threads "${LLAMA_THREADS:-8}"
