#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

check_training_data
require_file TVR_VR_TRAIN_JSONL
require_file FACE_H5
require_file PORTRAIT_H5

TASK_ROOT="${REPO_ROOT}/edg/event_driven_hybrid"
export PYTHONPATH="${TASK_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${TASK_ROOT}/crossmodal_moment_localization"
exec "${PYTHON_BIN}" train_video_retrieval.py \
  --dset_name tvr --exp_id EDG_video_retrieval \
  --results_root "${TRAIN_RESULTS_ROOT}" \
  --train_path "${TVR_VR_TRAIN_JSONL}" --eval_path "${TVR_VAL_JSONL}" \
  --desc_bert_path "${QUERY_H5}" --sub_bert_path "${SUB_H5}" \
  --vid_feat_path "${SLOWFAST_H5}" --face_feat_path "${FACE_H5}" \
  --portrait_feat_path "${PORTRAIT_H5}" \
  --video_duration_idx_path "${DURATION_JSON}" --sub_info_path "${SUB_INFO_JSON}" \
  --ctx_mode video_face_sub --vid_feat_size 4352 --face_feat_size 512 \
  --bsz 128 --n_epoch 100 --lr 1e-4 --wd 0.01 --seed 2018 \
  --hard_negtiave_start_epoch 20 --train_span_start_epoch 0 \
  --lw_st_ed 0.01 --lw_neg_ctx 1 --lw_neg_q 1 \
  --stop_task VCMR --eval_tasks_at_training VCMR SVMR VR \
  --num_workers "${NUM_WORKERS:-0}" "${@}"
