#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

check_common_data
require_file FACE_H5
require_file PORTRAIT_H5
require_model_dir VR_MODEL_DIR

TASK_ROOT="${REPO_ROOT}/edg/event_driven_hybrid"
export PYTHONPATH="${TASK_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd /tmp
exec "${PYTHON_BIN}" "${TASK_ROOT}/crossmodal_moment_localization/inference.py" \
  --model_dir "${VR_MODEL_DIR}" \
  --eval_path "${TVR_VAL_JSONL}" \
  --desc_bert_path "${QUERY_H5}" \
  --sub_bert_path "${SUB_H5}" \
  --vid_feat_path "${SLOWFAST_H5}" \
  --face_feat_path "${FACE_H5}" \
  --portrait_feat_path "${PORTRAIT_H5}" \
  --video_duration_idx_path "${DURATION_JSON}" \
  --sub_info_path "${SUB_INFO_JSON}" \
  --tasks VR \
  --min_pred_l 1 --max_pred_l 24 \
  --eval_query_bsz 50 --eval_context_bsz 200 --num_workers 0 \
  --eval_id edg_repro
