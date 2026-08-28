"""Prediction truncation and temporal NMS used by EDG evaluation."""

from collections import defaultdict

from edg.utils.temporal_nms import temporal_non_maximum_suppression


def get_submission_top_n(submission, top_n=100):
    output = {"video2idx": submission["video2idx"]}
    for task, predictions in submission.items():
        if task != "video2idx":
            output[task] = [
                {**item, "predictions": item["predictions"][:top_n]}
                for item in predictions
            ]
    return output


def _filter_vcmr_by_nms(predictions, threshold, max_before, max_after):
    grouped = defaultdict(list)
    for video_idx, start, end, score in predictions[:max_before]:
        grouped[video_idx].append([start, end, score])

    output = []
    for video_idx, moments in grouped.items():
        for moment in temporal_non_maximum_suppression(moments, threshold):
            output.append([video_idx] + moment)
    return sorted(output, key=lambda item: item[3], reverse=True)[:max_after]


def post_processing_vcmr_nms(results, nms_thd=0.6,
                             max_before_nms=1000, max_after_nms=100):
    for item in results:
        item["predictions"] = _filter_vcmr_by_nms(
            item["predictions"], nms_thd, max_before_nms, max_after_nms)
    return results


def post_processing_svmr_nms(results, nms_thd=0.6,
                             max_before_nms=1000, max_after_nms=100):
    for item in results:
        predictions = item["predictions"]
        if not predictions:
            continue
        video_idx = predictions[0][0]
        moments = temporal_non_maximum_suppression(
            [prediction[1:] for prediction in predictions[:max_before_nms]],
            nms_threshold=nms_thd,
        )[:max_after_nms]
        item["predictions"] = [[video_idx] + moment for moment in moments]
    return results
