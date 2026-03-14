#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "Virtual environment not found. Run ./init.sh first."
  exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

LLM_API_KEY="${LLM_API_KEY:-${OPENAI_API_KEY:-deo}}"
LLM_BASE_URL="${LLM_BASE_URL:-${OPENAI_BASE_URL:-http://192.168.1.140:9877/v1}}"
LLM_MODEL="${LLM_MODEL:-qwen3-vl-4b-gpu}"
LLM_MAX_SAMPLES="${LLM_MAX_SAMPLES:-100000}"
TARGET_MODE="${TARGET_MODE:-returns}"
COT_METHOD="${COT_METHOD:-cot_rf}"

python "${ROOT_DIR}/scripts/run_automated_pipeline_once.py" \
  --config "${ROOT_DIR}/src/config/default.yaml" \
  --data-dir "${ROOT_DIR}/Data_auto" \
  --seed-data-dir "${ROOT_DIR}/Data_auto" \
  --target-mode "${TARGET_MODE}" \
  --cot-end-to-end \
  --cot-method "${COT_METHOD}" \
  --llm-max-samples "${LLM_MAX_SAMPLES}" \
  --llm-model "${LLM_MODEL}" \
  --llm-api-key "${LLM_API_KEY}" \
  --llm-base-url "${LLM_BASE_URL}" \
  --no-paper-snapshot \
  "$@"
