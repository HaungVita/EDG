#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_ROOT}/outputs/training/logs"

echo "Active EDG training processes:"
pgrep -af 'train_video_retrieval.py|train_single_video_moment_retrieval.py' || true

echo
echo "GPU status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader

echo
echo "Latest training log:"
latest_log="$(find "${LOG_DIR}" -maxdepth 1 -type f -name '*.log' -printf '%T@ %p\n' 2>/dev/null \
  | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -n "${latest_log}" ]]; then
  echo "${latest_log}"
  tail -n 20 "${latest_log}"
else
  echo "No training log found under ${LOG_DIR}."
fi
