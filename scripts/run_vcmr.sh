#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

check_common_data
require_file VCMR_VR_INPUT
require_model_dir VCMR_MODEL_DIR

TASK_ROOT="${REPO_ROOT}/edg/cross_stage_guide/video_corpus_moment_retrieval"
export PYTHONPATH="${TASK_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd /tmp
exec "${PYTHON_BIN}" "${TASK_ROOT}/crossmodal_moment_localization/inference_external.py" \
  --model_dir "${VCMR_MODEL_DIR}" \
  --eval_model moment \
  --external_inference_vr_res_path "${VCMR_VR_INPUT}" \
  --eval_path "${TVR_VAL_JSONL}" \
  --desc_bert_path "${QUERY_H5}" \
  --sub_bert_path "${SUB_H5}" \
  --vid_feat_path "${SLOWFAST_H5}" \
  --video_duration_idx_path "${DURATION_JSON}" \
  --sub_info_path "${SUB_INFO_JSON}" \
  --tasks VCMR VR \
  --min_pred_l 1 --max_pred_l 24 \
  --eval_query_bsz 50 --eval_context_bsz 200 --num_workers 0 \
  --eval_id edg_repro
