#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS_FILE="${EDG_PATHS_FILE:-${REPO_ROOT}/configs/paths.env}"

if [[ ! -f "${PATHS_FILE}" ]]; then
  echo "Missing path configuration: ${PATHS_FILE}" >&2
  echo "Copy configs/paths.env.example to configs/paths.env first." >&2
  exit 2
fi

set -a
source "${PATHS_FILE}"
set +a

PYTHON_BIN="${PYTHON_BIN:-python}"

require_var() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "Required variable is unset: ${name}" >&2
    exit 2
  fi
}

require_file() {
  local name="$1"
  require_var "${name}"
  if [[ ! -f "${!name}" ]]; then
    echo "File does not exist (${name}): ${!name}" >&2
    exit 2
  fi
}

require_model_dir() {
  local name="$1"
  require_var "${name}"
  if [[ ! -f "${!name}/model.ckpt" || ! -f "${!name}/opt.json" ]]; then
    echo "${name} must contain model.ckpt and opt.json: ${!name}" >&2
    exit 2
  fi
}

check_common_data() {
  require_file TVR_VAL_JSONL
  require_file QUERY_H5
  require_file SUB_H5
  require_file SUB_INFO_JSON
  require_file DURATION_JSON
  require_file SLOWFAST_H5
}

check_training_data() {
  check_common_data
  require_var TRAIN_RESULTS_ROOT
  mkdir -p "${TRAIN_RESULTS_ROOT}"
}
