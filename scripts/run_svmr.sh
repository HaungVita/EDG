#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

check_common_data
require_model_dir SVMR_MODEL_DIR

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m edg.cross_stage_guide.single_video_moment_retrieval.inference \
  --model_dir "${SVMR_MODEL_DIR}" \
  --eval_path "${TVR_VAL_JSONL}" \
  --desc_bert_path "${QUERY_H5}" \
  --sub_bert_path "${SUB_H5}" \
  --vid_feat_path "${SLOWFAST_H5}" \
  --video_duration_idx_path "${DURATION_JSON}" \
  --sub_info_path "${SUB_INFO_JSON}" \
  --tasks SVMR \
  --min_pred_l 1 --max_pred_l 24 \
  --eval_query_bsz 50 --eval_context_bsz 200 --num_workers 0 \
  --eval_id edg_repro
