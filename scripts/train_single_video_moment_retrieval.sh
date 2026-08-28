#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

check_training_data
require_file TVR_MOMENT_TRAIN_JSONL
require_file TRAIN_VR_INPUT
require_file IDX2VIDEO_JSON

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -m edg.cross_stage_guide.single_video_moment_retrieval.train \
  --dset_name tvr --exp_id EDG_single_video_moment_retrieval \
  --results_root "${TRAIN_RESULTS_ROOT}" \
  --train_path "${TVR_MOMENT_TRAIN_JSONL}" --eval_path "${TVR_VAL_JSONL}" \
  --train_vr_input "${TRAIN_VR_INPUT}" --idx2video_path "${IDX2VIDEO_JSON}" \
  --desc_bert_path "${QUERY_H5}" --sub_bert_path "${SUB_H5}" \
  --vid_feat_path "${SLOWFAST_H5}" \
  --video_duration_idx_path "${DURATION_JSON}" --sub_info_path "${SUB_INFO_JSON}" \
  --ctx_mode video_sub --vid_feat_size 4352 --train_moment \
  --bsz 16 --n_epoch 100 --lr 1e-4 --wd 0.01 --seed 2018 \
  --hard_negtiave_start_epoch 20 --train_span_start_epoch 0 \
  --lw_st_ed 0.01 --lw_neg_ctx 1 --lw_neg_q 1 \
  --stop_task SVMR --eval_tasks_at_training SVMR \
  --num_workers "${NUM_WORKERS:-0}" "${@}"
