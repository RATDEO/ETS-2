#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-src/config/default.yaml}"
DATA_DIR="${DATA_DIR:-${ROOT_DIR}/Data_auto}"
SEED_DATA_DIR="${SEED_DATA_DIR:-${DATA_DIR}}"
TARGET_MODE="${TARGET_MODE:-returns}"
COT_METHOD="${COT_METHOD:-cot_rf}"
LLM_MAX_SAMPLES="${LLM_MAX_SAMPLES:-100000}"
LLM_MODEL="${LLM_MODEL:-qwen3-vl-4b-gpu}"
LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-deo}}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENAI_BASE_URL:-http://192.168.1.140:9877/v1}}"

python "${ROOT_DIR}/scripts/run_automated_pipeline_once.py" \
  --config "${CONFIG_PATH}" \
  --data-dir "${DATA_DIR}" \
  --seed-data-dir "${SEED_DATA_DIR}" \
  --target-mode "${TARGET_MODE}" \
  --cot-end-to-end \
  --cot-method "${COT_METHOD}" \
  --llm-max-samples "${LLM_MAX_SAMPLES}" \
  --llm-model "${LLM_MODEL}" \
  --llm-api-key "${LLM_API_KEY}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --no-paper-snapshot \
  "$@"
