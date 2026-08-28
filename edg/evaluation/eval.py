import numpy as np
from collections import OrderedDict, defaultdict
import json
import time
import copy
from tqdm import tqdm
import multiprocessing as mp
from edg.evaluation.utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, get_ap

TASK_TYPES = OrderedDict([
    ("VCMR", "Video Corpus Moment Retrieval"),
    ("SVMR", "Single Video Moment Retrieval"),
    ("VR", "regular Video Retrieval")
])

def get_rounded_percentage(float_number, n_floats=2):
    return round(float_number * 100, n_floats)

def eval_retrieval(submission, ground_truth, iou_thds=(0.5, 0.7), verbose=True, match_number=True, use_desc_type=True):
    #import pdb; pdb.set_trace()
    video2idx = submission['video2idx']
    submitted_task_types = [k for k in TASK_TYPES if k in submission]
    if verbose:
        print("Evaluating for task {}".format(submitted_task_types))
    eval_metrics = OrderedDict()
    metrics_raw_dict = {}
    #import pdb;pdb.set_trace()
    for task_type in submitted_task_types:
        metrics, metrics_by_type = eval_by_task_type(
            submission[task_type], video2idx, ground_truth,
            iou_thds=iou_thds, recall_topks=(1, 5, 10, 100),
            task_type=task_type, max_pred_per_query=100,
            match_number=match_number, verbose=verbose, use_desc_type=use_desc_type)
        metrics_raw_dict[task_type] = metrics
        metrics_raw_dict[task_type+"_by_type"] = metrics_by_type

    for task_type in submitted_task_types:
        eval_metrics[task_type] = metrics_raw_dict[task_type]
    if use_desc_type:
        for task_type in submitted_task_types:
            eval_metrics[task_type+"_by_type"] = metrics_raw_dict[task_type+"_by_type"]
    return eval_metrics

def pad_sequences_1d_np(sequences, dtype=np.float32):

    """ Pad a single-nested list or a sequence of n-d array (torch.tensor or np.ndarray)
    into a (n+1)-d array, only allow the first dim has variable lengths.
    Args:
        sequences: list(n-d tensor or list)
        dtype: np.dtype or torch.dtype
    Returns:
        padded_seqs: ((n+1)-d tensor) padded with zeros
        mask: (2d tensor) of the same shape as the first two dims of padded_seqs,
              1 indicate valid, 0 otherwise
    Examples:
        >>> test_data_list = [[1,2,3], [1,2], [3,4,7,9]]
        >>> pad_sequences_1d(test_data_list, dtype=np.float32)
        >>> test_data_3d = [np.random.randn(2,3,4), np.random.randn(4,3,4), np.random.randn(1,3,4)]
        >>> pad_sequences_1d(test_data_3d, dtype=np.float32)
    """
    if isinstance(sequences[0], list):
        sequences = [np.asarray(s, dtype=dtype) for s in sequences]

    extra_dims = sequences[0].shape[1:]  # the extra dims should be the same for all elements
    lengths = [len(seq) for seq in sequences]
    assert "numpy" in str(dtype), "dtype and input type does not match"
    padded_seqs = np.zeros((len(sequences), max(lengths)) + extra_dims, dtype=dtype)
    mask = np.zeros((len(sequences), max(lengths)), dtype=np.float32)

    for idx, seq in enumerate(sequences):
        end = lengths[idx]
        padded_seqs[idx, :end] = seq
        mask[idx, :end] = 1
    return padded_seqs, mask

def eval_by_task_type(moment_predictions, video2idx, ground_truth,
                     iou_thds=(0.5, 0.7), recall_topks=(1, 5, 10, 100),
                     task_type="SVMR", max_pred_per_query=100, match_number=True, verbose=True, use_desc_type=True):
    """ a predicted triplet is positive only if:
    1) its vid_name matches the GT vid_name
    2) IoU between its timestamp and GT timestamp is higher than the given threshold

    moment_predictions w.r.t. different task_type:
        For each query, evaluated on top max_pred_per_query [vid_name, st, ed] triplets. (score entry ignored)
        VCMR: vid_name might be repeating.
        SVMR: vid_name is fixed to be the GT vid_name.
        VR: vid_name is not repeating, st and ed will not be used.

    Args:
        video2idx: {vid_name (str): index (int), ...}
        moment_predictions: list(dict), each dict is {
            "desc": str,
            "desc_id": int,
            "predictions": [vid_name_idx (int), st (float), ed (float), score (float)] * n_pred,
                sorted predictions, n_pred could be different for all dicts. For each prediction,
                only the first 3 elements [vid_name (str), st (float), ed (float),] are used,
                any other following elements are ignored. We leave score here for record.
        }
        ground_truth: list(dict), each dict is {
            "desc": str,
            "desc_id": int,
            "type": str, one of [v, t, vt]
            "vid_name": str
            "ts": [st (float), ed (float)], or list([st (float), ed (float)]), len == 4.
            ...
        }
        iou_thds: temporal IoU thresholds
        recall_topks: recall at different top k
        task_type: str, could be: ["VCMR", "SVMR", "VR"], see TASK_TYPES for definition.
        max_pred_per_query: int, only top max_pred_per_query predictions for each query are used.
        match_number: bool, must set to True if when do evaluation, False is only used for debug.
        verbose:
        use_desc_type: only TVR has desc type
    Returns:

    """
    assert task_type in TASK_TYPES, "task_type must be one of {}".format(list(TASK_TYPES.keys()))
    if verbose:
        print("Running evaluation with task_type {}, n results {}; n gt {}"
              .format(task_type, len(moment_predictions), len(ground_truth)))

    predictions_by_desc_id = {e["desc_id"]: e for e in moment_predictions}
    gt_by_desc_id = {e["desc_id"]: e for e in ground_truth}
    desc_type2idx = {"v": 0, "t": 1, "vt": 2}
    desc_types = []  # n_desc
    if match_number:
        assert set(gt_by_desc_id.keys()) == set(predictions_by_desc_id.keys()), \
            "desc_ids in predictions and ground_truth must match"
    # assert len(set([len(e["predictions"]) for e in predictions_by_desc_id.values()])) == 1, \
    #     "all queries must have the same number of predictions"
    pred_info_matrix_collection = []
    for k, gt_item in tqdm(gt_by_desc_id.items(), desc="Loop over moments", leave=False):
        if not match_number and k not in predictions_by_desc_id:
            continue
        pred_info_matrix = np.array(
            [e[:3] for e in predictions_by_desc_id[k]["predictions"]][:max_pred_per_query],
            dtype=np.float32)  # (n_pred, 3)
        if use_desc_type:
            desc_types.append(desc_type2idx[gt_item["type"]])
        vid_name_matched_pred = pred_info_matrix[:, 0] == video2idx[gt_item["vid_name"]]  # bool, (n_pred, )
        pred_info_matrix = np.concatenate([pred_info_matrix, vid_name_matched_pred[:, None]], axis=1)  # (n_pred, 4)

        # add 1 + len(iou_thds) columns, iou_scores, iou_corrects for each iou_thd.
        iou_thd_corrects_columns = []
        if len(gt_item["ts"]) >= 4:  # didemo, fro all 3 splits, at least 4 ts for each, < 0.5% has more than 4.
            least_n_overlap = 2  # True if overlapped with at least least_n_overlap GT ts.
            iou_corrects_dict = defaultdict(list)
            for single_gt_ts in gt_item["ts"]:
                single_gt_ts = np.array(single_gt_ts, dtype=np.float32)  # (2, )
                # iou scores of the predictions that have wrong vid_name are set to 0.
                iou_scores = compute_temporal_iou_batch(pred_info_matrix[:, 1:3], single_gt_ts) * vid_name_matched_pred
                for iou_thd in iou_thds:
                    iou_corrects_dict[iou_thd].append(iou_scores >= iou_thd)
            for iou_thd in iou_thds:
                iou_corrects = sum(iou_corrects_dict[iou_thd]) >= least_n_overlap  # bool, (n_pred, )
                iou_thd_corrects_columns.append(iou_corrects[:, None])

        else:  # should be 2, len([st, ed]) == 2
            single_gt_ts = np.array(gt_item["ts"][0], dtype=np.float32)  # (2, )
            # iou scores of the predictions that have wrong vid_name are set to 0.
            iou_scores = compute_temporal_iou_batch(pred_info_matrix[:, 1:3], single_gt_ts) * vid_name_matched_pred

            for iou_thd in iou_thds:
                iou_corrects = iou_scores >= iou_thd  # bool, (n_pred, )
                iou_thd_corrects_columns.append(iou_corrects[:, None])

        pred_info_matrix = np.concatenate([pred_info_matrix, ] + iou_thd_corrects_columns, axis=1)  # (n_pred, 6)
        pred_info_matrix_collection.append(pred_info_matrix)

    # column header [vid_name_idx (int), st (float), ed (float), is_vid_name_match (bool),
    # iou_scores>=iou_thd0 (bool), iou_scores>=iou_thd1 (bool)]
    pred_info_matrix_collection = pad_sequences_1d_np(pred_info_matrix_collection)[0]  # (n_desc, n_pred, 6)
    if use_desc_type:
        desc_types = np.array(desc_types)  # (n_desc)

    # results wrapper
    metrics = OrderedDict()
    metrics_by_type = OrderedDict()


    iou_c_offset = 4  # iou_corrects column index starts here
    if task_type == "VCMR":
        for iou_idx, iou_thd in enumerate(iou_thds):
            iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(np.bool)  # (n_desc, n_pred)
            # 1) there might be more than one positive clip, so use `>= 1`
            for k in recall_topks:
                metrics["{}-r{}".format(iou_thd, k)] = \
                    get_rounded_percentage(np.mean(np.sum(iou_corrects[:, :k], axis=1) >= 1))
        if use_desc_type:
            for desc_type in desc_type2idx:
                type_corrects = desc_types == desc_type2idx[desc_type]  # (n_desc)
                n_desc_in_type = np.sum(type_corrects)  # (n_desc)
                for iou_idx, iou_thd in enumerate(iou_thds):
                    # (n_desc, n_pred)
                    iou_corrects = pred_info_matrix_collection[:, :, iou_c_offset + iou_idx].astype(np.bool)
                    for k in recall_topks:
                        metrics_by_type["{}-{}-r{}".format(desc_type, iou_thd, k)] = get_rounded_percentage(
                            1.0 * np.sum(np.logical_and(np.sum(iou_corrects[:, :k], axis=1) >= 1, type_corrects))
                            / n_desc_in_type
                        )
    elif task_type == "VR":
        vid_name_matched = pred_info_matrix_collection[:, :, 3].astype(np.bool)  # (n_desc, n_pred)
        for k in recall_topks:
            metrics["r{}".format(k)] = \
                get_rounded_percentage(np.mean(np.sum(vid_name_matched[:, :k], axis=1) >= 1))
        if use_desc_type:
            for desc_type in desc_type2idx:
                type_corrects = desc_types == desc_type2idx[desc_type]  # (n_desc)
                n_desc_in_type = np.sum(type_corrects)  # (n_desc)
                for k in recall_topks:
                    metrics_by_type["{}-r{}".format(desc_type, k)] = get_rounded_percentage(
                        1.0 * np.sum(np.logical_and(np.sum(vid_name_matched[:, :k], axis=1) >= 1, type_corrects))
                        / n_desc_in_type)
    else:
        raise ValueError("task_type wrong.")
    if use_desc_type:
        metrics_by_type["desc_type_ratio"] = "v {} t {} vt {}"\
            .format(*[get_rounded_percentage(1.0 * np.sum(desc_types == desc_type2idx[k]) / len(desc_types))
                      for k in ["v", "t", "vt"]])
    return metrics, metrics_by_type



def compute_temporal_iou_batch(preds, gt):
    """ compute intersection-over-union along temporal axis
    This function is significantly faster than `compute_temporal_iou`,
    the result should be the same.
    Args:
        preds: np.ndarray, (N, 2), [st (float), ed (float)] * N
        gt: [st (float), ed (float)]
    Returns:
        iou (float): np.ndarray, (N, )

    References:
        for np.divide with zeros, see https://stackoverflow.com/a/37977222
    """
    intersection = np.maximum(0, np.minimum(preds[:, 1], gt[1]) - np.maximum(preds[:, 0], gt[0]))
    union = np.maximum(preds[:, 1], gt[1]) - np.minimum(preds[:, 0], gt[0])  # not the correct union though
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)

def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores


def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["desc_id"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["desc_id"],  # in order to use the API
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["ts"][:max_gt_windows] \
            if max_gt_windows is not None else d["ts"]
        qid = d["desc_id"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["desc_id"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    iou_thd2ap["average"] = np.mean(ap_thds)
    # formatting
    iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return iou_thd2ap


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10)):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["desc_id"]: d["pred_relevant_windows"][0][:2] for d in submission}  # :2 rm scores
    gt_qid2window = {d["desc_id"]: d["ts"][0] for d in ground_truth}
    #gt_qid2window = {}
    #for d in ground_truth:
    #    cur_gt_windows = d["ts"]
    #    cur_qid = d["desc_id"]
    #    cur_max_iou_idx = 0
    #    if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
    #        cur_ious = compute_temporal_iou_batch_cross(
    #            np.array([pred_qid2window[cur_qid]]), np.array(d["ts"])
    #        )[0]
    #        cur_max_iou_idx = np.argmax(cur_ious)
    #    gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    iou_thd2recall_at_one = {}
    for thd in iou_thds:
        iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")
    return iou_thd2recall_at_one


def get_window_len(window):
    return window[1] - window[0]


def get_data_by_range(submission, ground_truth, len_range):
    """ keep queries with ground truth window length in the specified length range.
    Args:
        submission:
        ground_truth:
        len_range: [min_l (int), max_l (int)]. the range is (min_l, max_l], i.e., min_l < l <= max_l
    """
    min_l, max_l = len_range
    if min_l == 0 and max_l == 150:  # min and max l in dataset
        return submission, ground_truth

    # only keep ground truth with windows in the specified length range
    # if multiple GT windows exists, we only keep the ones in the range
    ground_truth_in_range = []
    gt_qids_in_range = set()
    #import pdb;pdb.set_trace()
    for d in ground_truth:
        rel_windows_in_range = [
            w for w in d["ts"] if min_l < get_window_len(w) <= max_l]
        if len(rel_windows_in_range) > 0:
            d = copy.deepcopy(d)
            d["ts"] = rel_windows_in_range
            ground_truth_in_range.append(d)
            gt_qids_in_range.add(d["desc_id"])

    # keep only submissions for ground_truth_in_range
    submission_in_range = []
    for d in submission:
        if d["desc_id"] in gt_qids_in_range:
            submission_in_range.append(copy.deepcopy(d))

    return submission_in_range, ground_truth_in_range


def eval_moment_retrieval(submission, ground_truth, verbose=True):
    length_ranges = [[0, 3], [3, 8], [8, 150], [0, 150], ]  #
    range_names = ["short", "middle", "long", "full"]

    ret_metrics = {}
    for l_range, name in zip(length_ranges, range_names):
        if verbose:
            start_time = time.time()
        _submission, _ground_truth = get_data_by_range(submission, ground_truth, l_range)
        print(f"{name}: {l_range}, {len(_ground_truth)}/{len(ground_truth)}="
              f"{100*len(_ground_truth)/len(ground_truth):.2f} examples.")
        iou_thd2average_precision = compute_mr_ap(_submission, _ground_truth, num_workers=8, chunksize=50)
        iou_thd2recall_at_one = compute_mr_r1(_submission, _ground_truth)
        ret_metrics[name] = {"MR-mAP": iou_thd2average_precision, "MR-R1": iou_thd2recall_at_one}
        if verbose:
            print(f"[eval_moment_retrieval] [{name}] {time.time() - start_time:.2f} seconds")
    return ret_metrics


def eval_all_submission(submission, ground_truth, verbose=True):
    """
    Args:
        submission:
        ground_truth:
        verbose:
    """
    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if True:
        video_ret_scores = eval_retrieval(submission, ground_truth, iou_thds=(0.5, 0.7), verbose=True, match_number=True, use_desc_type=True)
        eval_metrics.update(video_ret_scores)
        video_ret_scores_brief = {
            "VR-1": video_ret_scores["VR"]["r1"],
            "VR-5": video_ret_scores["VR"]["r5"],
            "VR-10": video_ret_scores["VR"]["r10"],
            "VR-100": video_ret_scores["VR"]["r100"],
        }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in video_ret_scores_brief.items()], key=lambda x: x[0]))
    """
    if "pred_saliency_scores" in submission[0]:
        highlight_det_scores = eval_highlight(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(highlight_det_scores)
        highlight_det_scores_brief = dict([
            (f"{k}-{sub_k.split('-')[1]}", v[sub_k])
            for k, v in highlight_det_scores.items() for sub_k in v])
        eval_metrics_brief.update(highlight_det_scores_brief)
    """
    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["vr_brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))

    return final_eval_metrics

def eval_vr_submission(submission, ground_truth, verbose=True):
    """
    Args:
        submission:
        ground_truth:
        verbose:
    """
    qid2preds = {d["desc_id"]: d for d in submission}
    qid2gt = {d["desc_id"]: d for d in ground_truth}
    eval_retrieval(submission, ground_truth, iou_thds=(0.5, 0.7), verbose=True, match_number=True, use_desc_type=True)

def eval_submission(submission, ground_truth, verbose=True, match_number=True):
    """
    Args:
        submission: list(dict), each dict is {
            qid: str,
            query: str,
            vid: str,
            pred_relevant_windows: list([st, ed]),
            pred_saliency_scores: list(float), len == #clips in video.
                i.e., each clip in the video will have a saliency score.
        }
        ground_truth: list(dict), each dict is     {
          "qid": 7803,
          "query": "Man in gray top walks from outside to inside.",
          "duration": 150,
          "vid": "RoripwjYFp8_360.0_510.0",
          "relevant_clip_ids": [13, 14, 15, 16, 17]
          "saliency_scores": [[4, 4, 2], [3, 4, 2], [2, 2, 3], [2, 2, 2], [0, 1, 3]]
               each sublist corresponds to one clip in relevant_clip_ids.
               The 3 elements in the sublist are scores from 3 different workers. The
               scores are in [0, 1, 2, 3, 4], meaning [Very Bad, ..., Good, Very Good]
        }
        verbose:
        match_number:

    Returns:

    """
    #for i, gt in enumerate(ground_truth):
    #    ground_truth[i]['ts'] = ground_truth[i]['ts']
    pred_qids = set([e["desc_id"] for e in submission])
    gt_qids = set([e["desc_id"] for e in ground_truth])
    if match_number:
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        submission = [e for e in submission if e["desc_id"] in shared_qids]
        ground_truth = [e for e in ground_truth if e["desc_id"] in shared_qids]

    eval_metrics = {}
    eval_metrics_brief = OrderedDict()
    if "pred_relevant_windows" in submission[0]:
        moment_ret_scores = eval_moment_retrieval(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(moment_ret_scores)
        moment_ret_scores_brief = {
            "MR-full-mAP": moment_ret_scores["full"]["MR-mAP"]["average"],
            "MR-full-mAP@0.5": moment_ret_scores["full"]["MR-mAP"]["0.5"],
            "MR-full-mAP@0.75": moment_ret_scores["full"]["MR-mAP"]["0.75"],
            "MR-short-mAP": moment_ret_scores["short"]["MR-mAP"]["average"],
            "MR-middle-mAP": moment_ret_scores["middle"]["MR-mAP"]["average"],
            "MR-long-mAP": moment_ret_scores["long"]["MR-mAP"]["average"],
            "MR-full-R1@0.5": moment_ret_scores["full"]["MR-R1"]["0.5"],
            "MR-full-R1@0.7": moment_ret_scores["full"]["MR-R1"]["0.7"],
        }
        eval_metrics_brief.update(
            sorted([(k, v) for k, v in moment_ret_scores_brief.items()], key=lambda x: x[0]))
    """
    if "pred_saliency_scores" in submission[0]:
        highlight_det_scores = eval_highlight(
            submission, ground_truth, verbose=verbose)
        eval_metrics.update(highlight_det_scores)
        highlight_det_scores_brief = dict([
            (f"{k}-{sub_k.split('-')[1]}", v[sub_k])
            for k, v in highlight_det_scores.items() for sub_k in v])
        eval_metrics_brief.update(highlight_det_scores_brief)
    """
    # sort by keys
    final_eval_metrics = OrderedDict()
    final_eval_metrics["brief"] = eval_metrics_brief
    final_eval_metrics.update(sorted([(k, v) for k, v in eval_metrics.items()], key=lambda x: x[0]))
    return final_eval_metrics


def eval_main():
    import argparse
    parser = argparse.ArgumentParser(description="Moments and Highlights Evaluation Script")
    parser.add_argument("--submission_path", type=str, help="path to generated prediction file")
    parser.add_argument("--gt_path", type=str, help="path to GT file")
    parser.add_argument("--save_path", type=str, help="path to save the results")
    parser.add_argument("--not_verbose", action="store_true")
    args = parser.parse_args()

    verbose = not args.not_verbose
    submission = load_jsonl(args.submission_path)
    gt = load_jsonl(args.gt_path)
    results = eval_submission(submission, gt, verbose=verbose)
    if verbose:
        print(json.dumps(results, indent=4))

    with open(args.save_path, "w") as f:
        f.write(json.dumps(results, indent=4))


if __name__ == '__main__':
    eval_main()
