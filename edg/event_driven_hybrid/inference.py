import sys

sys.path.append(r'/data/hk/tvr_hk/baselines')
sys.path.append(r'/data/hk/tvr_hk')
sys.path.append(r'/opt/data/private/tvr_hk/baselines')

import os
import copy
import math
import time
import pprint
from tqdm import tqdm, trange
import numpy as np
from collections import Counter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from .config import TestOptions
from .event_driven_hybrid import EventDrivenHybrid
from .start_end_dataset_with_face import \
    start_end_collate, StartEndEvalDataset, prepare_batch_inputs
from edg.evaluation.postprocessing import \
    get_submission_top_n, post_processing_vcmr_nms, post_processing_svmr_nms
from edg.utils.basic_utils import save_json, load_json, load_jsonl
from edg.utils.tensor_utils import find_max_triples_from_upper_triangle_product
from edg.evaluation.tvr_eval import eval_retrieval

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO)


# 修改 get shot_msg of video
def get_shot(vid):
    """
    getting shots in spcific video
    args: vid --> str: video name
    return shot_msg --> list: shot message
    """
    for video in shot_msg:
        if video["vid_name"] == vid:
            shot = video["shot_msg"]
            return shot


# 修改
def shot_match_clip(shots):
    """
    matching clip-level with
    args: shots --> dict: message for one shots
    return: st_mat, ed_mat --> tuple(int) clip_level shots after matching
    """

    clip_length = 1.5
    for shot in shots:
        st = shot["start_time"]
        ed = shot["end_time"]

        # over max of clip(100)
        if shot["start_time"] > 1.5 * 100:
            st_mat = -1
            ed_mat = -2
            shot["st_clip"] = st_mat
            shot["ed_clip"] = ed_mat
            continue
        if shot["start_time"] < 1.5 * 100 and shot["end_time"] > 1.5 * 100:
            st_mat = math.floor(st / clip_length)
            ed_mat = 99
            shot["st_clip"] = st_mat
            shot["ed_clip"] = ed_mat
            continue

        # floor match
        st_mat = math.floor(st / clip_length)

        # ceil math original
        ed_mat = math.ceil(ed / clip_length)

        shot["st_clip"] = st_mat
        shot["ed_clip"] = ed_mat
    return None


# 修改
def clipmatchshot(batch, max_clip):
    bsz = len(batch)
    clip2shot = torch.zeros(bsz, max_clip,
                            39) - 1  # batch_size, max number of clip
    for batch_id, sample in enumerate(batch):
        shots = get_shot(sample["vid_name"])
        shot_match_clip(shots)
        clip_level_shot = get_shot(sample["vid_name"])
        for shot_id, shot in enumerate(clip_level_shot):
            if shot["st_clip"] != -1 and shot["ed_clip"] != -2:
                st_clip = shot["st_clip"]
                ed_clip = shot["ed_clip"]
                clip2shot[batch_id, st_clip:ed_clip + 1, shot_id] = shot_id
            elif shot["st_clip"] != -1 and shot["ed_clip"] == 99:
                st_clip = shot["st_clip"]
                ed_clip = shot["ed_clip"]
                clip2shot[batch_id, st_clip:ed_clip + 1, shot_id] = shot_id
            else:
                pass
    return clip2shot


def compute_context_info(model, eval_dataset, opt):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated 2200 (videos) * 100 (frm) * 500 (hsz) * 4 (B) * 2 (video/sub) * 2 (layers) / (1024 ** 2) = 1.76 GB
    max_n_videos: only consider max_n_videos videos for each query to return st_ed scores.
    """
    model.eval()
    eval_dataset.set_data_mode("context")
    context_dataloader = DataLoader(eval_dataset,
                                    collate_fn=start_end_collate,
                                    batch_size=opt.eval_context_bsz,
                                    num_workers=opt.num_workers,
                                    shuffle=False,
                                    pin_memory=opt.pin_memory)

    metas = []  # list(dicts)
    video_feat1 = []
    video_feat12 = []
    video_feat_1_mask = []
    video_feat2 = []
    video_feat_2_mask = []
    video_feat3 = []
    video_feat_3_mask = []
    sub_feat1 = []
    sub_feat12 = []
    sub_feat_1_mask = []
    sub_feat2 = []
    sub_feat_2_mask = []
    sub_feat3 = []
    sub_feat_3_mask = []
    VR_video_feat = []
    VR_sub_feat = []
    VR_video_feat_mask = []
    VR_sub_feat_mask = []
    VR_VLAD = []
    VR_VLAD_mask = []
    # video_feat_svmr = []
    # video_feat_svmr_mask = []
    # sub_feat_svmr = []
    # sub_feat_svmr_mask = []
    # context_cat_feat = []
    # context_cat_mask = []
    for idx, batch in tqdm(enumerate(context_dataloader),
                           desc="Computing query2video scores",
                           total=len(context_dataloader)):
        metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1],
                                            device=opt.device,
                                            non_blocking=opt.pin_memory)
        # max_clip = model_inputs["video_feat"].shape[1]
        # model_inputs["clip2shot"] = clipmatchshot(batch[0], max_clip)  # tensor:
        device = model_inputs["video_feat"].device

        feature = model.encode_context(
            model_inputs["video_feat"],
            model_inputs["video_mask"],
            model_inputs["sub_feat"],
            model_inputs["sub_mask"],
        )
        _video_feat1 = feature["video_feat_1"]
        _video_feat2 = feature["video_feat_2"]
        _video_feat3 = feature["video_feat_3"]
        _video_feat1_mask = feature["video_feat_1_mask"]
        _video_feat2_mask = feature["video_feat_2_mask"]
        _video_feat3_mask = feature["video_feat_3_mask"]
        _sub_feat1 = feature["sub_feat_1"]
        _sub_feat2 = feature["sub_feat_2"]
        _sub_feat3 = feature["sub_feat_3"]
        _sub_feat1_mask = feature["sub_feat_1_mask"]
        _sub_feat2_mask = feature["sub_feat_2_mask"]
        _sub_feat3_mask = feature["sub_feat_3_mask"]
        _VR_video_feat = feature["VR_video_feat"]
        _VR_video_feat_mask = feature["VR_video_feat_mask"]
        _VR_sub_feat = feature["VR_sub_feat"]
        _VR_sub_feat_mask = feature["VR_sub_feat_mask"]
        _VR_video_sub_feat = feature["VR_VLAD"]
        _VR_video_sub_feat_mask = feature["VR_VLAD_mask"]
        # _video_feat1, _video_feat2, _sub_feat1, _sub_feat2, _mask, _ml= model.encode_context(
        #     model_inputs["video_feat"],
        #     model_inputs["video_mask"],
        #     model_inputs["sub_feat"],
        #     model_inputs["sub_mask"],
        # )  # 修改

        video_feat1.append(_video_feat1)
        video_feat12.append(_video_feat1)
        video_feat2.append(_video_feat2)
        video_feat3.append(_video_feat3)
        video_feat_1_mask.append(_video_feat1_mask)
        video_feat_2_mask.append(_video_feat2_mask)
        video_feat_3_mask.append(_video_feat3_mask)
        sub_feat1.append(_sub_feat1)
        sub_feat12.append(_sub_feat1)
        sub_feat2.append(_sub_feat2)
        sub_feat3.append(_sub_feat3)
        sub_feat_1_mask.append(_sub_feat1_mask)
        sub_feat_2_mask.append(_sub_feat2_mask)
        sub_feat_3_mask.append(_sub_feat3_mask)
        VR_video_feat.append(_VR_video_feat)
        VR_sub_feat.append(_VR_sub_feat)
        VR_video_feat_mask.append(_VR_video_feat_mask)
        VR_sub_feat_mask.append(_VR_sub_feat_mask)
        VR_VLAD.append(_VR_video_sub_feat)
        VR_VLAD_mask.append(_VR_video_sub_feat_mask)
    def cat_tensor(tensor_list):
        if len(tensor_list) == 0:
            return None
        else:
            seq_l = [e.shape[1] for e in tensor_list]
            b_sizes = [e.shape[0] for e in tensor_list]
            b_sizes_cumsum = np.cumsum([0] + b_sizes)
            if len(tensor_list[0].shape) == 3:
                hsz = tensor_list[0].shape[2]
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l),
                                                      hsz)
            elif len(tensor_list[0].shape) == 2:
                res_tensor = tensor_list[0].new_zeros(sum(b_sizes), max(seq_l))
            else:
                raise ValueError("Only support 2/3 dimensional tensors")
            for i, e in enumerate(tensor_list):
                res_tensor[b_sizes_cumsum[i]:b_sizes_cumsum[i +
                                                            1], :seq_l[i]] = e
            return res_tensor

    return dict(
        video_metas=metas,
        video_feat_1=cat_tensor(video_feat1),
        video_feat_12=cat_tensor(video_feat1),
        video_feat_2=cat_tensor(video_feat2),
        video_feat_3=cat_tensor(video_feat3),
        video_feat_1_mask=cat_tensor(video_feat_1_mask),
        video_feat_2_mask=cat_tensor(video_feat_2_mask),
        video_feat_3_mask=cat_tensor(video_feat_3_mask),
        sub_feat_1=cat_tensor(sub_feat1),
        sub_feat_12=cat_tensor(sub_feat1),
        sub_feat_2=cat_tensor(sub_feat2),
        sub_feat_3=cat_tensor(sub_feat3),
        sub_feat_1_mask=cat_tensor(sub_feat_1_mask),
        sub_feat_2_mask=cat_tensor(sub_feat_2_mask),
        sub_feat_3_mask=cat_tensor(sub_feat_3_mask),
        VR_video_feat=cat_tensor(VR_video_feat),
        VR_video_feat_mask=cat_tensor(VR_video_feat_mask),
        VR_sub_feat=cat_tensor(VR_sub_feat),
        VR_sub_feat_mask=cat_tensor(VR_sub_feat_mask),
        VR_VLAD=cat_tensor(VR_VLAD),
        VR_VLAD_mask=cat_tensor(VR_VLAD_mask),
    )


def index_if_not_none(input_tensor, indices):
    if input_tensor is None:
        return input_tensor
    else:
        return input_tensor[indices]


def compute_query2ctx_info_svmr_only(model,
                                     eval_dataset,
                                     opt,
                                     ctx_info,
                                     max_before_nms=1000,
                                     max_n_videos=200,
                                     tasks=("SVMR", )):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 500 (hsz) * 4 / (1024**2) = 38.15 MB
    max_n_videos: int, use max_n_videos videos for computing VCMR results
    """
    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(True)
    query_eval_loader = DataLoader(eval_dataset,
                                   collate_fn=start_end_collate,
                                   batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=opt.pin_memory)
    video2idx = eval_dataset.video2idx
    video_metas = ctx_info["video_metas"]
    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz
    ctx_len = eval_dataset.max_ctx_len  # all pad to this length

    svmr_video2meta_idx = {
        e["vid_name"]: idx
        for idx, e in enumerate(video_metas)
    }
    svmr_gt_st_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)
    svmr_gt_ed_probs = np.zeros((n_total_query, ctx_len), dtype=np.float32)

    query_metas = []
    for idx, batch in tqdm(enumerate(query_eval_loader),
                           desc="Computing q embedding",
                           total=len(query_eval_loader)):
        _query_metas = batch[0]
        query_metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1],
                                            device=opt.device,
                                            non_blocking=opt.pin_memory)
        # query_context_scores (_N_q, N_videos), st_prob, ed_prob (_N_q, L)
        query2video_meta_indices = torch.LongTensor(
            [svmr_video2meta_idx[e["vid_name"]] for e in _query_metas])
        #_query_context_scores, _st_probs, _ed_probs = \
        #    model.get_pred_from_raw_query(model_inputs["query_feat"], model_inputs["query_mask"],
        #                                  index_if_not_none(ctx_info["video_feat1"], query2video_meta_indices),
        #                                  index_if_not_none(ctx_info["video_feat2"], query2video_meta_indices),
        #                                  index_if_not_none(ctx_info["video_mask"], query2video_meta_indices),
        #                                  index_if_not_none(ctx_info["sub_feat1"], query2video_meta_indices),
        #                                  index_if_not_none(ctx_info["sub_feat2"], query2video_meta_indices),
        #                                  index_if_not_none(ctx_info["sub_mask"], query2video_meta_indices),
        #                                  cross=False)
        _query_context_scores, _st_probs, _ed_probs, _, _  = \
            model.get_pred_from_triplet(model_inputs["query_feat"], model_inputs["query_mask"],
                                          index_if_not_none(ctx_info["video_feat1"], query2video_meta_indices),
                                          index_if_not_none(ctx_info["video_mask"], query2video_meta_indices),
                                          index_if_not_none(ctx_info["sub_feat1"], query2video_meta_indices),
                                          index_if_not_none(ctx_info["sub_mask"], query2video_meta_indices),
                                          cross=False)
        _query_context_scores = _query_context_scores + 1  # move cosine similarity to [0, 2]

        # normalize to get true probabilities!!!
        # the probabilities here are already (pad) masked, so only need to do softmax
        _st_probs = F.softmax(_st_probs, dim=-1)  # (_N_q, L)
        _ed_probs = F.softmax(_ed_probs, dim=-1)

        svmr_gt_st_probs[idx * bsz:(idx + 1) *
                         bsz, :_st_probs.shape[1]] = _st_probs.cpu().numpy()
        svmr_gt_ed_probs[idx * bsz:(idx + 1) *
                         bsz, :_ed_probs.shape[1]] = _ed_probs.cpu().numpy()

        if opt.debug:
            break
    svmr_res = get_svmr_res_from_st_ed_probs(svmr_gt_st_probs,
                                             svmr_gt_ed_probs,
                                             query_metas,
                                             video2idx,
                                             clip_length=opt.clip_length,
                                             min_pred_l=opt.min_pred_l,
                                             max_pred_l=opt.max_pred_l,
                                             max_before_nms=max_before_nms)
    return dict(SVMR=svmr_res)


def generate_min_max_length_mask(array_shape, min_l, max_l):
    """ The last two dimension denotes matrix of upper-triangle with upper-right corner masked,
    below is the case for 4x4.
    [[0, 1, 1, 0],
     [0, 0, 1, 1],
     [0, 0, 0, 1],
     [0, 0, 0, 0]]

    Args:
        array_shape: np.shape??? The last two dimensions should be the same
        min_l: int, minimum length of predicted span
        max_l: int, maximum length of predicted span

    Returns:

    """
    single_dims = (1, ) * (len(array_shape) - 2)
    mask_shape = single_dims + array_shape[-2:]
    extra_length_mask_array = np.ones(mask_shape,
                                      dtype=np.float32)  # (1, ..., 1, L, L)
    mask_triu = np.triu(extra_length_mask_array, k=min_l)
    mask_triu_reversed = 1 - np.triu(extra_length_mask_array, k=max_l)
    final_prob_mask = mask_triu * mask_triu_reversed
    return final_prob_mask  # with valid bit to be 1


def get_svmr_res_from_biaffine_probs(biaffine_probs, query_metas, video2idx,
                                     clip_length, min_pred_l, max_pred_l,
                                     max_before_nms):
    """
    Args:
        svmr_gt_st_probs: np.ndarray (N_queries, L, L), value range [0, 1]
        svmr_gt_ed_probs:
        query_metas:
        video2idx:
        clip_length: float, how long each clip is in seconds
        min_pred_l: int, minimum number of clips
        max_pred_l: int, maximum number of clips
        max_before_nms: get top-max_before_nms predictions for each query

    Returns:

    """
    svmr_res = []
    query_vid_names = [e["vid_name"] for e in query_metas]

    # masking very long ones! Since most are relatively short.
    #st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs, svmr_gt_ed_probs)  # (N, L, L)
    # extra_length_mask_array = np.ones(st_ed_prob_product.shape, dtype=np.bool)  # (N, L, L)
    # mask_triu = np.triu(extra_length_mask_array, k=min_pred_l)
    # mask_triu_reversed = np.logical_not(np.triu(extra_length_mask_array, k=max_pred_l))
    # final_prob_mask = np.logical_and(mask_triu, mask_triu_reversed)  # with valid bit to be 1
    #valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape, min_l=min_pred_l, max_l=max_pred_l)
    valid_prob_mask = generate_min_max_length_mask(biaffine_probs.shape,
                                                   min_l=min_pred_l,
                                                   max_l=max_pred_l)
    biaffine_probs *= valid_prob_mask  # invalid location will become zero!

    batched_sorted_triples = find_max_triples_from_upper_triangle_product(
        biaffine_probs, top_n=max_before_nms, prob_thd=None)
    #import pdb;pdb.set_trace()
    for i, q_vid_name in tqdm(
            enumerate(query_vid_names),
            desc="[SVMR_2D] Loop over queries to generate predictions",
            total=len(query_vid_names)):  # i is query_id
        q_m = query_metas[i]
        video_idx = video2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        _sorted_triples[:,
                        1] += 1  # as we redefined ed_idx, which is inside the moment.
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * clip_length
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = [[
            video_idx,
        ] + row for row in _sorted_triples.tolist()]
        cur_query_pred = dict(desc_id=q_m["desc_id"],
                              desc=q_m["desc"],
                              predictions=cur_ranked_predictions)
        svmr_res.append(cur_query_pred)
    return svmr_res


def get_svmr_res_from_st_ed_probs(svmr_gt_st_probs, svmr_gt_ed_probs,
                                  query_metas, video2idx, clip_length,
                                  min_pred_l, max_pred_l, max_before_nms):
    """
    Args:
        svmr_gt_st_probs: np.ndarray (N_queries, L, L), value range [0, 1]
        svmr_gt_ed_probs:
        query_metas:
        video2idx:
        clip_length: float, how long each clip is in seconds
        min_pred_l: int, minimum number of clips
        max_pred_l: int, maximum number of clips
        max_before_nms: get top-max_before_nms predictions for each query

    Returns:

    """
    svmr_res = []
    query_vid_names = [e["vid_name"] for e in query_metas]

    # masking very long ones! Since most are relatively short.
    st_ed_prob_product = np.einsum("bm,bn->bmn", svmr_gt_st_probs,
                                   svmr_gt_ed_probs)  # (N, L, L)
    # extra_length_mask_array = np.ones(st_ed_prob_product.shape, dtype=np.bool)  # (N, L, L)
    # mask_triu = np.triu(extra_length_mask_array, k=min_pred_l)
    # mask_triu_reversed = np.logical_not(np.triu(extra_length_mask_array, k=max_pred_l))
    # final_prob_mask = np.logical_and(mask_triu, mask_triu_reversed)  # with valid bit to be 1
    valid_prob_mask = generate_min_max_length_mask(st_ed_prob_product.shape,
                                                   min_l=min_pred_l,
                                                   max_l=max_pred_l)
    st_ed_prob_product *= valid_prob_mask  # invalid location will become zero!

    batched_sorted_triples = find_max_triples_from_upper_triangle_product(
        st_ed_prob_product, top_n=max_before_nms, prob_thd=None)
    for i, q_vid_name in tqdm(
            enumerate(query_vid_names),
            desc="[SVMR] Loop over queries to generate predictions",
            total=len(query_vid_names)):  # i is query_id
        q_m = query_metas[i]
        video_idx = video2idx[q_vid_name]
        _sorted_triples = batched_sorted_triples[i]
        _sorted_triples[:,
                        1] += 1  # as we redefined ed_idx, which is inside the moment.
        _sorted_triples[:, :2] = _sorted_triples[:, :2] * clip_length
        # [video_idx(int), st(float), ed(float), score(float)]
        cur_ranked_predictions = [[
            video_idx,
        ] + row for row in _sorted_triples.tolist()]
        cur_query_pred = dict(desc_id=q_m["desc_id"],
                              desc=q_m["desc"],
                              predictions=cur_ranked_predictions)
        svmr_res.append(cur_query_pred)
    return svmr_res


def load_external_vr_res2(external_vr_res_path, top_n_vr_videos=5):
    """return a mapping from desc_id to top retrieved video info"""
    external_vr_res = load_json(external_vr_res_path)
    external_vr_res = get_submission_top_n(external_vr_res,
                                           top_n=top_n_vr_videos)["VR"]
    query2video = {e["desc_id"]: e["predictions"] for e in external_vr_res}
    return query2video


def score2d_to_moments_scores(score2d, num_clips, duration):
    grids = score2d.nonzero()
    scores = score2d[grids[:, 0], grids[:, 1]]
    grids[:, 1] += 1
    moments = grids * duration / num_clips
    return moments, scores


def compute_query2ctx_info(model,
                           eval_dataset,
                           opt,
                           ctx_info,
                           max_before_nms=1000,
                           max_n_videos=100,
                           tasks=("SVMR", )):
    """Use val set to do evaluation, remember to run with torch.no_grad().
    estimated size 20,000 (query) * 500 (hsz) * 4 / (1024**2) = 38.15 MB
    max_n_videos: int, use max_n_videos videos for computing VCMR/VR results
    """
    #max_n_videos=10
    # max_triplet_videos == max_n_videos
    max_triplet_videos = 100
    max_n_videos = 100
    is_svmr = "SVMR" in tasks
    is_vr = "VR" in tasks
    is_vcmr = "VCMR" in tasks

    video2idx = eval_dataset.video2idx
    video_metas = ctx_info["video_metas"]
    if opt.external_inference_vr_res_path is not None:
        video_idx2meta_idx = {
            video2idx[m["vid_name"]]: i
            for i, m in enumerate(video_metas)
        }
        external_query2video = \
            load_external_vr_res2(opt.external_inference_vr_res_path, top_n_vr_videos=max_n_videos)
        # 「query idx： [video meta idx]」
        external_query2video_meta_idx = \
            {k: [video_idx2meta_idx[e[0]] for e in v] for k, v in external_query2video.items()}
    else:
        external_query2video = None
        external_query2video_meta_idx = None

    model.eval()
    eval_dataset.set_data_mode("query")
    eval_dataset.load_gt_vid_name_for_query(is_svmr)
    query_eval_loader = DataLoader(eval_dataset,
                                   collate_fn=start_end_collate,
                                   batch_size=opt.eval_query_bsz,
                                   num_workers=opt.num_workers,
                                   shuffle=False,
                                   pin_memory=opt.pin_memory)
    n_total_videos = len(video_metas)
    n_total_query = len(eval_dataset)
    bsz = opt.eval_query_bsz

    is_biaffine = model.is_biaffine

    if is_vcmr:
        flat_st_ed_scores_sorted_indices = np.empty(
            (n_total_query, max_before_nms), dtype=np.int32)
        flat_st_ed_sorted_scores = np.zeros((n_total_query, max_before_nms),
                                            dtype=np.float32)
        flat_2d_scores_sorted_indices = np.empty(
            (n_total_query, max_before_nms), dtype=np.int32)
        flat_2d_sorted_scores = np.zeros((n_total_query, max_before_nms),
                                         dtype=np.float32)

    if is_vr or is_vcmr:
        sorted_q2c_indices = np.empty((n_total_query, 1 * max_n_videos),
                                      dtype=np.int32)
        sorted_q2c_scores = np.empty((n_total_query, 1 * max_n_videos),
                                     dtype=np.float32)

    if is_svmr:
        svmr_video2meta_idx = {
            e["vid_name"]: idx
            for idx, e in enumerate(video_metas)
        }
        # 100+64+32+16=212
        # svmr_gt_st_probs = np.zeros((n_total_query, 212), dtype=np.float32)
        # svmr_gt_ed_probs = np.zeros((n_total_query, 212), dtype=np.float32)
        # svmr_gt_2d_probs = np.zeros((n_total_query, 212, opt.max_ctx_l),
        #                             dtype=np.float32)
        # 修改
        svmr_gt_st_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
        svmr_gt_ed_probs = np.zeros((n_total_query, opt.max_ctx_l), dtype=np.float32)
        svmr_gt_2d_probs = np.zeros((n_total_query, opt.max_ctx_l, opt.max_ctx_l), dtype=np.float32)

    query_metas = []
    for idx, batch in tqdm(enumerate(query_eval_loader),
                           desc="Computing q embedding",
                           total=len(query_eval_loader)):
        _query_metas = batch[0]
        query_metas.extend(batch[0])
        model_inputs = prepare_batch_inputs(batch[1],
                                            device=opt.device,
                                            non_blocking=opt.pin_memory)
        # query_context_scores (_N_q, N_videos), st_prob, ed_prob (_N_q, N_videos, L)
        my_query_feat, my_query_mask = model.query_embedding(
            model_inputs["query_feat"], model_inputs["query_mask"])
        # 修改##################################################
        model_inputs["query_face_feat"] = None  # query 不加入人脸信息
        if not model_inputs["query_face_feat"] == None:
            my_query_face_hidden = model.query_face2token_linear(
                model_inputs["query_face_feat"])
            my_query_face_hidden = my_query_face_hidden.unsqueeze(1)
        else:
            my_query_face_hidden = None
        # 待定是否加入人脸
        if not my_query_face_hidden == None:
            my_query_feat, my_query_mask = model.extend_query_with_face(
                my_query_feat, my_query_mask, my_query_face_hidden)
        # query 分层
        word_feat = my_query_feat
        word_mask = my_query_mask
        _query_context_scores, _st_probs, _ed_probs, tmp_query1, tmp_query2, _out_2d = \
            model.get_pred_from_triplet(word_feat, word_mask,
                                        ctx_info, my_query_face_hidden,
                                          cross=True, is_eval=True, max_triplet_videos=max_triplet_videos)

        if is_biaffine:
            _2d_probs = torch.nn.Softmax(2)(_out_2d.view(
                *_out_2d.size()[:2], -1)).view_as(_out_2d)
        #_2d_probs = _out_2d
        #score2d_to_moments_scores(_out_2d, num_clips, duration)
        if is_svmr:  # collect SVMR data
            row_indices = torch.arange(0, len(_st_probs))
            query2video_meta_indices = torch.LongTensor(
                [svmr_video2meta_idx[e["vid_name"]] for e in _query_metas])

            _, _gt_st_probs, _gt_ed_probs, tmp_query1, tmp_query2, _gt_2d_probs  = \
                model.get_pred_from_triplet(word_feat, word_mask,
                                        ctx_info, my_query_face_hidden,
                                              cross=True, is_eval=True, max_triplet_videos=max_triplet_videos, gt_idx=query2video_meta_indices)

            _gt_st_probs = F.softmax(_gt_st_probs,
                                     dim=-1)  # (_N_q, N_videos, L)
            _gt_ed_probs = F.softmax(_gt_ed_probs, dim=-1)
            #torch.nn.Softmax(dim=1)(a.reshape(a.shape[0],-1)).reshape(a.shape)
            _gt_st_probs = _gt_st_probs.squeeze(1)
            _gt_ed_probs = _gt_ed_probs.squeeze(1)
            svmr_gt_st_probs[idx * bsz:(idx + 1) * bsz, :_st_probs.shape[2]] = \
                _gt_st_probs[row_indices].cpu().numpy()
            svmr_gt_ed_probs[idx * bsz:(idx + 1) * bsz, :_ed_probs.shape[2]] = \
                _gt_ed_probs[row_indices].cpu().numpy()
            #import pdb; pdb.set_trace()
            if is_biaffine:
                #print(torch.argmax(_gt_2d_probs[0,0]))
                _gt_2d_probs = torch.nn.Softmax(1)(_gt_2d_probs.view(
                    *_gt_2d_probs.size()[:1], -1)).view_as(_gt_2d_probs)
                #import pdb; pdb.set_trace()
                #print(torch.argmax(_gt_2d_probs[0,0]))
                #_gt_2d_probs = _gt_2d_probs
                svmr_gt_2d_probs[idx * bsz:(idx + 1) * bsz, :_2d_probs.shape[2], :_2d_probs.shape[3]] = \
                    _gt_2d_probs[row_indices].cpu().numpy()
        if not (is_vr or is_vcmr):
            continue

        # Get top-max_n_videos videos for each query
        # _sorted_q2c_scores, _sorted_q2c_indices = \
        # torch.sort(_query_context_scores, descending=True)  # (_N_q, N_videos)
        # _sorted_q2c_scores = _sorted_q2c_scores[:, :max_n_videos]  # (N_q, max_n_videos)
        # _sorted_q2c_indices = _sorted_q2c_indices[:, :max_n_videos]

        # TODO CHANGING HERE !!!!
        if external_query2video is None:
            _sorted_q2c_scores, _sorted_q2c_indices = \
                torch.topk(_query_context_scores, max_n_videos * 1, dim=1, largest=True)
            ### 修改 注释
            # _sorted_q2c_scores = F.softmax(_sorted_q2c_scores, dim=1)
        else:
            relevant_video_info = [
                external_query2video[qm["desc_id"]] for qm in _query_metas
            ]
            _sorted_q2c_indices = _query_context_scores.new(
                [[video_idx2meta_idx[sub_e[0]] for sub_e in e]
                 for e in relevant_video_info]).long()
            _sorted_q2c_scores = _query_context_scores.new(
                [[sub_e[3] for sub_e in e] for e in relevant_video_info])
            _sorted_q2c_scores = torch.exp(opt.q2c_alpha * _sorted_q2c_scores)
        # collect data for vr and vcmr
        sorted_q2c_indices[idx * bsz:(idx + 1) *
                           bsz] = _sorted_q2c_indices.cpu().numpy()
        sorted_q2c_scores[idx * bsz:(idx + 1) *
                          bsz] = _sorted_q2c_scores.cpu().numpy()

        if not is_vcmr:
            continue

        # Get VCMR results
        # compute combined scores
        row_indices = torch.arange(0, len(_st_probs),
                                   device=opt.device).unsqueeze(1)
        # _st_probs = _st_probs[row_indices, _sorted_q2c_indices]  # (_N_q, max_n_videos, L)
        # _ed_probs = _ed_probs[row_indices, _sorted_q2c_indices]
        #import pdb; pdb.set_trace()
        # (_N_q, max_n_videos, L, L)
        if is_biaffine:
            valid_prob_mask = generate_min_max_length_mask(
                _2d_probs.shape, min_l=opt.min_pred_l, max_l=opt.max_pred_l)
            _2d_scores = _2d_probs * torch.from_numpy(valid_prob_mask).to(
                _2d_probs.device)  # invalid location will become zero!
            _n_q = _2d_scores.shape[0]
            _flat_2d_scores = _2d_scores.reshape(_n_q,
                                                 -1)  # (N_q, max_n_videos*L*L)
            _flat_2d_sorted_scores, _flat_2d_scores_sorted_indices = \
                torch.sort(_flat_2d_scores, dim=1, descending=True)
            flat_2d_sorted_scores[idx * bsz:(idx + 1) * bsz] = \
                _flat_2d_sorted_scores[:, :max_before_nms].cpu().numpy()
            flat_2d_scores_sorted_indices[idx * bsz:(idx + 1) * bsz] = \
                _flat_2d_scores_sorted_indices[:, :max_before_nms].cpu().numpy()

        else:
            #TODO st, ed probs and q2c scores are not compatible
            # 修改: 改变 score 计算方式
            n_q, n_v, lv = _st_probs.shape
            _st_probs = _st_probs.contiguous().view(-1, n_v * lv)
            _st_probs = F.softmax(_st_probs, dim=-1)
            _ed_probs = _ed_probs.contiguous().view(-1, n_v * lv)
            _ed_probs = F.softmax(_ed_probs, dim=-1)
            _st_probs = _st_probs.view(n_q, n_v, lv)
            _ed_probs = _ed_probs.view(n_q, n_v, lv)
            _st_probs = _st_probs.unsqueeze(-1)
            _ed_probs = _ed_probs.unsqueeze(-2)
            _sorted_q2c_scores = F.softmax(_sorted_q2c_scores, dim=1)
            # _sorted_q2c_scores = F.softmax(_sorted_q2c_scores, dim=1)
            # infer 10 先测试 VR
            # _sorted_q2c_scores = _sorted_q2c_scores.unsqueeze(-1).unsqueeze(-1)
            # _st_ed_scores = _st_probs + _sorted_q2c_scores + _ed_probs
            # 连乘计算_st_ed_scores
            # _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs, _sorted_q2c_scores, _ed_probs)
            _st_ed_scores = _st_probs + _ed_probs

            # _st_ed_scores = torch.einsum("qvm,qv,qvn->qvmn", _st_probs,
            #                              _sorted_q2c_scores, _ed_probs)
            valid_prob_mask = generate_min_max_length_mask(
                _st_ed_scores.shape,
                min_l=opt.min_pred_l,
                max_l=opt.max_pred_l)
            _st_ed_scores *= torch.from_numpy(valid_prob_mask).to(
                _st_ed_scores.device)  # invalid location will become zero!
            # sort across the top-max_n_videos videos (by flatten from the 2nd dim)
            # the indices here are local indices, not global indices
            _n_q = _st_ed_scores.shape[0]
            _flat_st_ed_scores = _st_ed_scores.reshape(
                _n_q, -1)  # (N_q, max_n_videos*L*L)
            _flat_st_ed_sorted_scores, _flat_st_ed_scores_sorted_indices = \
                torch.sort(_flat_st_ed_scores, dim=1, descending=True)
            # collect data
            flat_st_ed_sorted_scores[idx * bsz:(idx + 1) * bsz] = \
                _flat_st_ed_sorted_scores[:, :max_before_nms].cpu().numpy()
            flat_st_ed_scores_sorted_indices[idx * bsz:(idx + 1) * bsz] = \
                _flat_st_ed_scores_sorted_indices[:, :max_before_nms].cpu().numpy()

        if opt.debug:
            break
    # Numpy starts here!!!
    svmr_res = []
    #import pdb;pdb.set_trace()
    if is_svmr:
        if is_biaffine:
            svmr_2d_res = get_svmr_res_from_biaffine_probs(
                svmr_gt_2d_probs,
                query_metas,
                video2idx,
                clip_length=opt.clip_length,
                min_pred_l=opt.min_pred_l,
                max_pred_l=opt.max_pred_l,
                max_before_nms=max_before_nms)
        else:
            svmr_res = get_svmr_res_from_st_ed_probs(
                svmr_gt_st_probs,
                svmr_gt_ed_probs,
                query_metas,
                video2idx,
                clip_length=opt.clip_length,
                min_pred_l=opt.min_pred_l,
                max_pred_l=opt.max_pred_l,
                max_before_nms=max_before_nms)
    vr_res = []
    ########## Testing Code of restoring name entity of anonymous text query, via checking whether the hidden name appears in the sub ##########
    """
    if is_vr:
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(zip(sorted_q2c_scores[:, :100], sorted_q2c_indices[:, :100])),
                desc="[VR] Loop over queries to generate predictions", total=n_total_query):
            cur_vr_redictions = []
            for j, (v_score, v_meta_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                video_sub = video_metas[v_meta_idx]["sub"]
                video_appear = []
                for sub in video_sub:
                    for sub_appear in sub["appear"]:
                        video_appear.append(sub_appear)
                video_appear = Counter(video_appear)
                for query_appear in query_metas[i]["appear"]:
                    if query_appear in video_appear:
                        cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
                        break
            cur_vr_redictions = cur_vr_redictions[:100]
            pred_video_idx = [cur[0] for cur in cur_vr_redictions]
            for j, (v_score, v_meta_idx) in enumerate(zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                if len(cur_vr_redictions)<100:
                    if video_idx not in pred_video_idx:
                        cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
                        pred_video_idx.append(video_idx)
                else:
                    break
            #import pdb
            #pdb.set_trace()
            cur_query_pred = dict(
                desc_id=query_metas[i]["desc_id"],
                desc=query_metas[i]["desc"],
                predictions=cur_vr_redictions
            )
            vr_res.append(cur_query_pred)
    """

    if is_vr:
        #import pdb;pdb.set_trace()
        for i, (_sorted_q2c_scores_row, _sorted_q2c_indices_row) in tqdm(
                enumerate(
                    zip(sorted_q2c_scores[:, :100],
                        sorted_q2c_indices[:, :100])),
                desc="[VR] Loop over queries to generate predictions",
                total=n_total_query):
            cur_vr_redictions = []
            for j, (v_score, v_meta_idx) in enumerate(
                    zip(_sorted_q2c_scores_row, _sorted_q2c_indices_row)):
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                cur_vr_redictions.append([video_idx, 0, 0, float(v_score)])
            #import pdb;pdb.set_trace()
            cur_query_pred = dict(desc_id=query_metas[i]["desc_id"],
                                  desc=query_metas[i]["desc"],
                                  predictions=cur_vr_redictions)
            vr_res.append(cur_query_pred)

    vcmr_res = []
    vcmr_2d_res = []
    flat_score_sorted_indices = None
    flat_sorted_scores = None
    if is_vcmr:
        if is_biaffine:
            flat_score_sorted_indices = flat_2d_scores_sorted_indices
            flat_sorted_scores = flat_2d_sorted_scores
        else:
            flat_score_sorted_indices = flat_st_ed_scores_sorted_indices
            flat_sorted_scores = flat_st_ed_sorted_scores
        for i, (_flat_scores_sorted_indices, _flat_sorted_scores) in tqdm(
                enumerate(zip(flat_score_sorted_indices, flat_sorted_scores)),
                desc="[VCMR] Loop over queries to generate predictions",
                total=n_total_query):  # i is query_idx
            # list([video_idx(int), st(float), ed(float), score(float)])
            video_meta_indices_local, pred_st_indices, pred_ed_indices = \
                np.unravel_index(_flat_scores_sorted_indices,
                                 shape=(max_n_videos, opt.max_ctx_l, opt.max_ctx_l))
            # video_meta_indices_local refers to the indices among the top-max_n_videos
            # video_meta_indices refers to the indices in all the videos, which is the True indices
            video_meta_indices = sorted_q2c_indices[i,
                                                    video_meta_indices_local]

            pred_st_in_seconds = pred_st_indices.astype(
                np.float32) * opt.clip_length
            pred_ed_in_seconds = pred_ed_indices.astype(
                np.float32) * opt.clip_length + opt.clip_length
            cur_vcmr_redictions = []
            for j, (v_meta_idx, v_score) in enumerate(
                    zip(video_meta_indices, _flat_sorted_scores)):  # videos
                video_idx = video2idx[video_metas[v_meta_idx]["vid_name"]]
                cur_vcmr_redictions.append([
                    video_idx,
                    float(pred_st_in_seconds[j]),
                    float(pred_ed_in_seconds[j]),
                    float(v_score)
                ])

            cur_query_pred = dict(desc_id=query_metas[i]["desc_id"],
                                  desc=query_metas[i]["desc"],
                                  predictions=cur_vcmr_redictions)
            vcmr_res.append(cur_query_pred)
    res = dict(SVMR=svmr_res, VCMR=vcmr_res, VR=vr_res)

    return {k: v for k, v in res.items() if len(v) != 0}


def get_eval_res(model, eval_dataset, opt, tasks, max_after_nms):
    """compute and save query and video proposal embeddings"""
    context_info = compute_context_info(model, eval_dataset, opt)
    if "VCMR" in tasks or "VR" in tasks:
        logger.info("Inference with full-script.")
        eval_res = compute_query2ctx_info(model,
                                          eval_dataset,
                                          opt,
                                          context_info,
                                          max_before_nms=opt.max_before_nms,
                                          max_n_videos=opt.max_vcmr_video,
                                          tasks=tasks)
    else:
        logger.info("Inference at [SVMR only] mode. This script is different.")
        eval_res = compute_query2ctx_info_svmr_only(
            model,
            eval_dataset,
            opt,
            context_info,
            max_before_nms=opt.max_before_nms,
            max_n_videos=max_after_nms,
            tasks=tasks)
    eval_res["video2idx"] = eval_dataset.video2idx
    return eval_res


POST_PROCESSING_MMS_FUNC = {
    "SVMR": post_processing_svmr_nms,
    "VCMR": post_processing_vcmr_nms
}


def eval_epoch(model,
               eval_dataset,
               opt,
               save_submission_filename,
               tasks=("SVMR", ),
               max_after_nms=100):
    """max_after_nms: always set to 100, since the eval script only evaluate top-100"""
    model.eval()
    logger.info("Computing scores")
    # logger.info("Start timing")
    # times = []
    # for _ in range(3):
    #     st_time = time.time()
    #     eval_submission_raw = get_eval_res(model, eval_dataset, opt, tasks, max_after_nms=max_after_nms)
    #     times += [time.time() - st_time]
    # times = torch.FloatTensor(times)
    eval_submission_raw = get_eval_res(model,
                                       eval_dataset,
                                       opt,
                                       tasks,
                                       max_after_nms=max_after_nms)

    IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    eval_submission = get_submission_top_n(eval_submission_raw,
                                           top_n=max_after_nms)
    save_json(eval_submission, submission_path)

    if opt.eval_split_name == "val":  # since test_public has no GT
        metrics = eval_retrieval(eval_submission,
                                 eval_dataset.query_data,
                                 iou_thds=IOU_THDS,
                                 match_number=not opt.debug,
                                 verbose=opt.debug,
                                 use_desc_type=opt.dset_name == "tvr")
        save_metrics_path = submission_path.replace(".json", "_metrics.json")
        save_json(metrics,
                  save_metrics_path,
                  save_pretty=True,
                  sort_keys=False)
        latest_file_paths = [submission_path, save_metrics_path]
    else:
        metrics = None
        latest_file_paths = [
            submission_path,
        ]
    # metrics["time_avg"] = float(times.mean())
    # metrics["time_std"] = float(times.std())

    if opt.nms_thd != -1:
        logger.info("Performing nms with nms_thd {}".format(opt.nms_thd))
        eval_submission_after_nms = dict(
            video2idx=eval_submission_raw["video2idx"])
        for k, nms_func in POST_PROCESSING_MMS_FUNC.items():
            if k in eval_submission_raw:
                eval_submission_after_nms[k] = nms_func(
                    eval_submission_raw[k],
                    nms_thd=opt.nms_thd,
                    max_before_nms=opt.max_before_nms,
                    max_after_nms=max_after_nms)

        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(
            ".json", "_nms_thd_{}.json".format(opt.nms_thd))
        save_json(eval_submission_after_nms, submission_nms_path)
        if opt.eval_split_name == "val":
            metrics_nms = eval_retrieval(eval_submission_after_nms,
                                         eval_dataset.query_data,
                                         iou_thds=IOU_THDS,
                                         match_number=not opt.debug,
                                         verbose=opt.debug)
            save_metrics_nms_path = submission_nms_path.replace(
                ".json", "_metrics.json")
            save_json(metrics_nms,
                      save_metrics_nms_path,
                      save_pretty=True,
                      sort_keys=False)
            latest_file_paths += [submission_nms_path, save_metrics_nms_path]
        else:
            metrics_nms = None
            latest_file_paths = [
                submission_nms_path,
            ]
    else:
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths


def setup_model(opt):
    """Load model from checkpoint and move to specified device"""
    checkpoint = torch.load(opt.ckpt_filepath)
    loaded_model_cfg = checkpoint["model_cfg"]
    loaded_model_cfg["stack_conv_predictor_conv_kernel_sizes"] = -1
    loaded_model_cfg["type_vocab_size"] = 2

    model = EventDrivenHybrid(loaded_model_cfg)
    model.load_state_dict(checkpoint["model"])
    logger.info("Loaded model saved at epoch {} from checkpoint: {}".format(
        checkpoint["epoch"], opt.ckpt_filepath))

    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        if len(opt.device_ids) > 1:
            logger.info("Use multi GPU", opt.device_ids)
            model = torch.nn.DataParallel(
                model, device_ids=opt.device_ids)  # use multi GPU
    return model


def start_inference():
    logger.info("Setup config, data and model...")
    opt = TestOptions().parse()
    cudnn.benchmark = False
    cudnn.deterministic = True

    assert opt.eval_path is not None
    eval_dataset = StartEndEvalDataset(
        dset_name=opt.dset_name,
        eval_split_name=opt.eval_split_name,  # should only be val set
        data_path=opt.eval_path,
        desc_bert_path_or_handler=opt.desc_bert_path,
        sub_bert_path_or_handler=opt.sub_bert_path,
        max_desc_len=opt.max_desc_l,
        max_ctx_len=opt.max_ctx_l,
        video_duration_idx_path=opt.video_duration_idx_path,
        sub_info_path=opt.sub_info_path,
        vid_feat_path_or_handler=opt.vid_feat_path,
        face_feat_path_or_handler=opt.face_feat_path,
        portrait_feat_path_or_handler=opt.portrait_feat_path,
        clip_length=opt.clip_length,
        ctx_mode=opt.ctx_mode,
        data_mode="query",
        h5driver=opt.h5driver,
        data_ratio=opt.data_ratio,
        normalize_vfeat=not opt.no_norm_vfeat,
        normalize_tfeat=not opt.no_norm_tfeat)

    model = setup_model(opt)
    save_submission_filename = "inference_{}_{}_{}_predictions_{}.json".format(
        opt.dset_name, opt.eval_split_name, opt.eval_id, "_".join(opt.tasks))
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename,
                       tasks=opt.tasks, max_after_nms=100)
    logger.info("metrics_no_nms \n{}".format(
        pprint.pformat(metrics_no_nms, indent=4)))
    logger.info("metrics_nms \n{}".format(pprint.pformat(metrics_nms,
                                                         indent=4)))


if __name__ == '__main__':
    start_inference()
