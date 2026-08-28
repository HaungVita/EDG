"""
Dataset for clip model
"""
import sys
# sys.path.append(r'/data/hk/tvr_hk/baselines')
sys.path.append(r'/opt/data/private/tvr_hk/baselines')
sys.path.append(r'/opt/data/private/tvr_hk/')
# sys.path.append(r'/data/hk/tvr_hk')
sys.path.append(r'/hy-tmp/datasets/')
import json
import logging
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py
from crossmodal_moment_localization.lmdb_feature import open_video_features
import time
import math
import random
from tqdm import tqdm
from utils.basic_utils import load_jsonl, load_json, l2_normalize_np_array, flat_list_of_lists, merge_dicts
from utils.tensor_utils import pad_sequences_1d, pad_sequences_2d
from clip_alignment_with_language.local_utils.compute_proposal_upper_bound import \
    get_didemo_agreed_ts

logger = logging.getLogger(__name__)


class StartEndDataset(Dataset):
    """
    Args:
        dset_name, str, ["tvr"]
        ctx_mode: str,
    Return:
        a dict: {
            "meta": {
                "desc_id": int,
                "desc": str,
                "appear": list(str)
                "vid_name": str,
                "duration": float,
                "ts": [st (float), ed (float)], seconds, ground_truth timestamps
            }
            "model_inputs": {
                "query_pos_id": torch.tensor, (30, )
                "query_token_id": torch.tensor,(30, )
                "query_feat": torch.tensor, (L, D_q)
                “query_face_feat”： torch.tensor, (1, 512)
                "video_pos_id" = torch.tensor, (100, )
                "video_token_id" = torch.tensor, (100, )
                "video_feat": torch.tensor, (n_clip_in_moment, D_video)
                "sub_pos_id" = sub_feat_pos_id, (100, )
                "sub_token_id" = sub_feat_token_id, (100, )
                "sub_feat": torch.tensor, (n_clip_in_moment, D_sub)
                "st_ed_indices": torch.LongTensor, (2, )
                "iou2d": float
                "span_mask": torch.tensor, (n_clip_in_moment, )
            }
        }
    """
    def __init__(self, dset_name, data_path, desc_bert_path_or_handler, sub_bert_path_or_handler,
                 max_desc_len, max_ctx_len,
                 vid_feat_path_or_handler, face_feat_path_or_handler, portrait_feat_path_or_handler, clip_length, train_moment, ctx_mode="video",
                 normalize_vfeat=True, normalize_tfeat=True, h5driver=None, data_ratio=2.0,):
        self.train_moment = train_moment
        # 利用 1 阶段的结果
        if train_moment:
            # retrieval_file = "/opt/data/private/tvr_hk/baselines/crossmodal_moment_localization/train_retrieval.json"
            # retrieval_file = "/opt/data/private/tvr_hk/baselines/crossmodal_moment_localization/train_retrieval.json"
            # retrieval_file = "/root/hk_tmp_data/train_retrieval.json"  # /root目录下更快
            # retrieval_file = "/root/hk_tmp_data/inference_tvr_train_9999_predictions_VCMR_SVMR_VR.json"  # /root目录下更快
            retrieval_file = "/root/hk_tmp_data/inference_tvr_train_2_predictions_VCMR_SVMR_VR.json"
            video2idx_file = "/opt/data/private/tvr_hk/data/tvr_video2dur_idx.json"
            idx2video_file = "/opt/data/private/tvr_hk/data/idx2video.json"
            with open(retrieval_file, 'r') as f1:
                with open(video2idx_file, "r") as f2:
                    with open(idx2video_file, "r") as f3:
                        self.retrieval = json.load(f1)["VR"]
                        self.video2idx = json.load(f2)
                        self.idx2video = json.load(f3)["train"]

        self.dset_name = dset_name
        self.data_path = data_path
        self.data_ratio = data_ratio

        self.desc_bert_path_or_handler = desc_bert_path_or_handler
        self.max_desc_len = max_desc_len

        self.sub_bert_path_or_handler = sub_bert_path_or_handler
        self.max_ctx_len = max_ctx_len
        self.vid_feat_path_or_handler = vid_feat_path_or_handler
        self.face_feat_path_or_handler = face_feat_path_or_handler

        self.portrait_feat_path_or_handler = portrait_feat_path_or_handler
        self.clip_length = clip_length
        self.ctx_mode = ctx_mode

        # prepare desc data
        self.data = load_jsonl(data_path)
        if self.data_ratio != 1:
            n_examples = int(len(self.data) * data_ratio)
            self.data = self.data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(data_ratio * 100, n_examples))

        self.use_video = "video" in self.ctx_mode
        self.use_sub = "sub" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode
        self.use_face = "face" in self.ctx_mode

        # 修改
        if True and self.portrait_feat_path_or_handler:

            # 修改: 执行 video_sub --> Query端加入我的ch20， video不加入
            if isinstance(portrait_feat_path_or_handler, h5py.File):
                self.portrait_feat_h5 = portrait_feat_path_or_handler
            else:
                self.portrait_feat_h5 = h5py.File(portrait_feat_path_or_handler, "r", driver=h5driver)


        if self.use_video:
            if isinstance(vid_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = vid_feat_path_or_handler
            else:  # str path
                self.vid_feat_h5 = open_video_features(vid_feat_path_or_handler, h5driver)

        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r", driver=h5driver)

        if self.use_sub:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:  # str path
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r", driver=h5driver)

        if self.use_face:
            if isinstance(face_feat_path_or_handler, h5py.File):
                self.face_feat_h5 = face_feat_path_or_handler
            else: # str path
                self.face_feat_h5 = h5py.File(face_feat_path_or_handler, "r", driver=h5driver)
            if isinstance(portrait_feat_path_or_handler, h5py.File):
                self.portrait_feat_h5 = portrait_feat_path_or_handler
            else:
                self.portrait_feat_h5 = h5py.File(portrait_feat_path_or_handler, "r", driver=h5driver)

        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat
        self.visual_token_id = 0
        self.text_token_id = 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        raw_data = self.data[index]

        # initialize with basic data
        # 如果"appear"不存在
        if "appear" not in raw_data:
            raw_data["appear"] = []
        meta = dict(
            desc_id=raw_data["desc_id"],
            desc=raw_data["desc"],
            appear=raw_data["appear"],
            vid_name=raw_data["vid_name"],
            duration=raw_data["duration"],
            ts=raw_data["ts"] if self.dset_name != "didemo" else get_didemo_agreed_ts(raw_data["ts"]),
        )
        model_inputs = dict()
        query_feat_pos_id = torch.arange(self.max_desc_len, dtype=torch.long)
        query_feat_token_id = torch.full((self.max_desc_len,), self.text_token_id, dtype=torch.long)
        model_inputs["query_pos_id"] = query_feat_pos_id
        model_inputs["query_token_id"] = query_feat_token_id
        model_inputs["query_feat"] = self.get_query_feat_by_desc_id(meta["desc_id"])
        # 修改 在Query加入feat
        model_inputs["query_face_feat"] = self.get_face_feat_by_nid(meta["appear"])

        ctx_l = 0
        if self.use_video:
            video_feat = self.vid_feat_h5[meta["vid_name"]][:self.max_ctx_len]  # (N_clip, D)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            visual_feat_pos_id = torch.arange(self.max_ctx_len, dtype=torch.long)
            visual_feat_token_id = torch.full((self.max_ctx_len,), self.visual_token_id, dtype=torch.long)
            span_mask = torch.zeros(video_feat.shape[0], dtype=torch.long)
            model_inputs["video_feat"] = torch.from_numpy(video_feat)
            model_inputs["video_pos_id"] = visual_feat_pos_id
            model_inputs["video_token_id"] = visual_feat_token_id
            ctx_l = len(video_feat)

        else:
            model_inputs["video_feat"] = torch.zeros((2, 2))

        if self.use_sub:  # no need for ctx feature, as the features are already contextulized
            sub_feat = self.sub_bert_h5[meta["vid_name"]][:self.max_ctx_len]  # (N_clips, D_t)
            if self.normalize_tfeat:
                sub_feat = l2_normalize_np_array(sub_feat)
            sub_feat_pos_id = torch.arange(self.max_ctx_len, dtype=torch.long)
            sub_feat_token_id = torch.full((self.max_ctx_len,), self.text_token_id, dtype=torch.long)
            model_inputs["sub_feat"] = torch.from_numpy(sub_feat)
            model_inputs["sub_pos_id"] = sub_feat_pos_id
            model_inputs["sub_token_id"] = sub_feat_token_id
            ctx_l = len(sub_feat)
        else:
            model_inputs["sub_feat"] = torch.zeros((2, 2))

        if self.train_moment:
            if self.retrieval:
                # 排除 positive 找到前 topk negtive sample
                pos_video = raw_data["vid_name"]
                pos_video_id = self.video2idx["train"][pos_video][1]  # [duration, idx]
                # 使用 next() 和 enumerate() 查找 训练集中检索得分最高的 video
                index = next((i for i, item in enumerate(self.retrieval) if item["desc_id"] == raw_data["desc_id"]), None)
                vr_result = self.retrieval[index]  # {desc_id: , "desc":, "preictions": }
                vr_prediction = vr_result["predictions"]
                vr_neg_video_id = []
                vr_neg_video = []
            # gt 的 rank
            neg_video_num = 5
            # search_depth = 80
            search_depth = 500
            location = 100
            # negative_video_pool_list = [self.idx2video[str(item[0])] for item in vr_prediction if meta["vid_name"] != self.idx2video[str(item[0])]]
            for rank, prediction in enumerate(vr_prediction):
                if prediction[0] == pos_video_id:
                    location = rank
                    break
                if rank == 100:
                    break
            # 排除掉 vr_prediction[location]
            sampled_negative_video_pool = random.sample(vr_prediction[:location] + \
                                              vr_prediction[location + 1:location + search_depth], neg_video_num)
            # 正确的视频first engine 的 得分
            rp = vr_prediction[location][3]
            rn = [item[3] for item in sampled_negative_video_pool]
            # weight_hard = 1 - (rp / (rp + sum(rn)))
            weight_hard = (math.exp(rp) / (math.exp(rp) + sum([math.exp(i) for i in rn])))
            model_inputs["weight_hard"] = weight_hard
            # sampled_negative_video_pool = random.sample(vr_prediction[:location + search_depth], neg_video_num)
            vr_neg_video_id = [neg_video[0] for neg_video in sampled_negative_video_pool]
            vr_neg_video = [self.idx2video[str(video_id)] for video_id in vr_neg_video_id]
            # for prediction in vr_prediction:
            #     # 负采样 4 个
            #     if len(vr_neg_video_id) < 5:
            #         if prediction[0] = pos_video_id:
            #             vr_neg_video_id.append(prediction[0])
            #             vr_neg_video.append(self.idx2video[str(prediction[0])])
            # vr_neg_video = raw_data["neg_video"][:2]
            neg_video_feat = []
            neg_sub_feat = []
            neg_visual_feat_pos_id = []
            neg_visual_feat_token_id = []
            neg_sub_feat_pos_id = []
            neg_sub_feat_token_id = []
            for name in vr_neg_video:
                _neg_video_feat = torch.from_numpy(l2_normalize_np_array(self.vid_feat_h5[name][:self.max_ctx_len]))
                _neg_sub_feat = torch.from_numpy(l2_normalize_np_array(self.sub_bert_h5[name][:self.max_ctx_len]))
                _neg_visual_feat_pos_id = torch.arange(self.max_ctx_len, dtype=torch.long)
                _neg_visual_feat_token_id = torch.full((self.max_ctx_len,), self.visual_token_id, dtype=torch.long)
                _neg_sub_feat_pos_id = torch.arange(self.max_ctx_len, dtype=torch.long)
                _neg_sub_feat_token_id = torch.full((self.max_ctx_len,), self.text_token_id, dtype=torch.long)
                neg_video_feat.append(_neg_video_feat)
                neg_visual_feat_pos_id.append(_neg_visual_feat_pos_id)
                neg_visual_feat_token_id.append(_neg_visual_feat_token_id)
                neg_sub_feat.append(_neg_sub_feat)
                neg_sub_feat_pos_id.append(_neg_sub_feat_pos_id)
                neg_sub_feat_token_id.append(_neg_sub_feat_token_id)
            meta["vr_neg_video"] = vr_neg_video
            model_inputs["neg_video_feat"] = neg_video_feat
            model_inputs["neg_visual_feat_pos_id"] = torch.stack(neg_visual_feat_pos_id)
            model_inputs["neg_visual_feat_token_id"] = torch.stack(neg_visual_feat_token_id)
            model_inputs["neg_sub_feat"] = neg_sub_feat
            model_inputs["neg_sub_feat_pos_id"] = torch.stack(neg_sub_feat_pos_id)
            model_inputs["neg_sub_feat_token_id"] = torch.stack(neg_sub_feat_token_id)
            # model_inputs["neg_video_feat"] = torch.stack(neg_video_feat)
            # model_inputs["neg_visual_feat_pos_id"] = torch.stack(neg_visual_feat_pos_id)
            # model_inputs["negvisual_feat_token_id"] = torch.stack(neg_visual_feat_token_id)
            # model_inputs["neg_sub_feat"] = torch.stack(neg_sub_feat)
            # model_inputs["neg_sub_feat_pos_id"] = torch.stack(neg_sub_feat_pos_id)
            # model_inputs["negsub_feat_token_id"] = torch.stack(neg_sub_feat_token_id)
            # model_inputs["neg_video_feat"] = self.vid_feat_h5[meta["vid_name"]][:self.max_ctx_len]
        if self.use_face:
            # (n_clusters, hidden_dim)
            face_feat = torch.from_numpy(self.get_face_feat_by_video_name(meta["vid_name"]))

            model_inputs["face_feat"] = face_feat
        # 被我注释
        if self.use_face:
            face_feat = self.face_feat_h5[meta["vid_name"]][:self.max_ctx_len]  # (N_clip, D)
            face_feat = face_feat[:video_feat.shape[0]]
            #if self.normalize_vfeat:
            #    video_feat = l2_normalize_np_array(video_feat)
            model_inputs["face_feat"] = torch.from_numpy(face_feat)
            ctx_l = len(face_feat)
        else:
            model_inputs["face_feat"] = torch.zeros((2, 2))

        if self.use_tef:
            # note the tef features here are normalized clip indices (1.5 secs), instead of the original time (1 sec)
            ctx_l = meta["duration"] // self.clip_length + 1 if ctx_l == 0 else ctx_l
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (N_clips, 2)
            model_inputs["tef_feat"] = tef
        else:
            model_inputs["tef_feat"] = torch.zeros((2, 2))

        if self.use_video and self.use_tef:
            model_inputs["video_feat"] = torch.cat(
                [model_inputs["video_feat"], model_inputs["tef_feat"]], dim=1)  # (N_clips, D+2)
        if self.use_sub and self.use_tef:
            model_inputs["sub_feat"] = torch.cat(
                [model_inputs["sub_feat"], model_inputs["tef_feat"]], dim=1)  # (N_clips, D_t+2)

        model_inputs["st_ed_indices"] = self.get_st_ed_label(meta["ts"], max_idx=ctx_l-1)
        moment = torch.tensor([max(meta["ts"][0], 0), min(meta["ts"][1], meta["duration"])])
        iou2d = self.moment_to_iou2d(moment=moment, num_clips=model_inputs["video_feat"].shape[0], duration=meta["duration"])
        model_inputs["iou2d"] = iou2d

        # 修改 gt
        model_inputs["span_mask"] = self.get_span_label(span_mask, model_inputs["st_ed_indices"])
        return dict(meta=meta, model_inputs=model_inputs)

    # 修改
    def get_face_feat_by_video_name(self, vid_name):
        face_feat = self.face_feat_h5[vid_name][:]
        if False:  # 恒定为 False self.normalize_vfeat = True
            face_feat = l2_normalize_np_array(face_feat)
        return face_feat
    # 修改


    def get_st_ed_label(self, ts, max_idx):
        """
        Args:
            ts: [st (float), ed (float)] in seconds, ed > st
            max_idx: length of the video

        Returns:
            [st_idx, ed_idx]: int,

        Given ts = [3.2, 7.6], st_idx = 2, ed_idx = 6,
        clips should be indexed as [2: 6), the translated back ts should be [3:9].
        # TODO which one is better, [2: 5] or [2: 6)
        """
        st_idx = min(math.floor(ts[0] / self.clip_length), max_idx)
        # Given ts = [3.2, 7.6], st_idx = 2, ed_idx = 5,
        # clips should be indexed as [2: 5], the translated back ts should be [3:9].
        ed_idx = min(math.floor(ts[1] / self.clip_length), max_idx)
        # 修改: 注释
        # ed_idx = min(math.ceil(ts[1] / self.clip_length), max_idx)
        return torch.LongTensor([st_idx, ed_idx])

    def get_span_label(self, span_mask, st_ed_indices):
        span_mask[st_ed_indices[0]:st_ed_indices[1]+1]=1
        return span_mask

    def get_query_feat_by_desc_id(self, desc_id):
        #print(desc_id)
        #import pdb; pdb.set_trace()
        query_feat = self.desc_bert_h5[str(desc_id)][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)

    def get_face_feat_by_nid(self, appear):
        '''
        输入: 一个clip内出现的人物列表 e.g "appear": ["nm0001455", "nm0001612", "nm0001710"]
        输出: (1, 512)一个clip内平均人物的特征, 没有检测到人物则用zeor vector 替代
        '''
        appear_id = []  # 记录appear中出现人的id
        appear_feat = []  # 记录人脸信息
        for ap in appear:
            appear_id.append(ap)
            appear_feat.append(self.portrait_feat_h5[ap])
        if len(appear_feat)==0:  # 如果没有检测到人，则cat vector zeor
            appear_feat.append(np.zeros((512,)))
        return torch.from_numpy(np.expand_dims(np.mean(appear_feat, axis=0), axis=0))


    def score2d_to_moments_scores(self, score2d, num_clips, duration):
        grids = score2d.nonzero()
        scores = score2d[grids[:,0], grids[:,1]]
        grids[:, 1] += 1
        moments = grids * duration / num_clips
        return moments, scores

    def iou(self, candidates, gt):
        start, end = candidates[:,0], candidates[:,1]
        s, e = gt[0].float(), gt[1].float()
        # print(s.dtype, start.dtype)
        inter = end.min(e) - start.max(s)
        union = end.max(e) - start.min(s)
        return inter.clamp(min=0) / union

    def moment_to_iou2d(self, moment, num_clips, duration):
        iou2d = torch.ones(num_clips, num_clips)
        candidates, _ = self.score2d_to_moments_scores(iou2d, num_clips, duration)
        iou2d = self.iou(candidates, moment).reshape(num_clips, num_clips)
        return iou2d



class StartEndEvalDataset(Dataset):
    """
    init_data_mode: `video_query` or `video_only` or `query_only`,
        it indicates which data to load when initialize the Dataset object.
    data_mode: `context` or `query`, it indicates which data to return for self.__get_item__()
    desc_bert_path_or_handler: h5py.File object or str path
    vid_feat_path_or_handler: h5py.File object or str path
    eval_proposal_bsz: the proposals for a single video will be sorted in length and batched here with
        max batch size to be eval_proposal_bsz. A single video might have multiple batches of proposals.
    load_gt_video: load GroundTruth Video, useful when evaluating single video moment retrieval.
    data_ratio: percentage of query data to use.
    """
    def __init__(self, dset_name, eval_split_name, data_path=None,
                 desc_bert_path_or_handler=None, max_desc_len=None,  max_ctx_len=None, sub_info_path=None,
                 sub_bert_path_or_handler=None, vid_feat_path_or_handler=None, face_feat_path_or_handler=None,
                 portrait_feat_path_or_handler=None, video_duration_idx_path=None, clip_length=None,
                 ctx_mode="video", data_mode="context",
                 h5driver=None, data_ratio=1.0, normalize_vfeat=True, normalize_tfeat=True):
        self.dset_name = dset_name
        self.eval_split_name = eval_split_name
        self.ctx_mode = ctx_mode
        self.load_gt_video = False
        self.data_ratio = data_ratio  # only affect query data
        self.normalize_vfeat = normalize_vfeat
        self.normalize_tfeat = normalize_tfeat
        self.portrait_feat_path_or_handler = portrait_feat_path_or_handler

        self.data_mode = None
        self.set_data_mode(data_mode)

        # 修改
        if True and self.portrait_feat_path_or_handler:
            # 指定位置
            # cli2shot_path = "/opt/data/private/tvr_hk/hk/tests/data/shot_msg/shot_msgV1.json"
            # self.clip2shot = load_json(cli2shot_path)

            #  修改
            if isinstance(portrait_feat_path_or_handler, h5py.File):
                self.portrait_feat_h5 = portrait_feat_path_or_handler
            else:
                self.portrait_feat_h5 = h5py.File(portrait_feat_path_or_handler, "r", driver=h5driver)

        self.max_desc_len = max_desc_len
        self.max_ctx_len = max_ctx_len
        self.data_path = data_path
        self.query_data = load_jsonl(data_path)
        if data_ratio != 1:
            n_examples = int(len(self.query_data) * data_ratio)
            self.query_data = self.query_data[:n_examples]
            logger.info("Using {}% of the data: {} examples".format(data_ratio * 100, n_examples))
        if isinstance(desc_bert_path_or_handler, h5py.File):
            self.desc_bert_h5 = desc_bert_path_or_handler
        else:
            self.desc_bert_h5 = h5py.File(desc_bert_path_or_handler, "r", driver=h5driver)

        video_data = load_json(video_duration_idx_path)[self.eval_split_name]
        self.video_data = [{"vid_name": k, "duration": v[0]} for k, v in video_data.items()]
        self.video2idx = {k: v[1] for k, v in video_data.items()}
        self.clip_length = clip_length

        self.use_video = "video" in self.ctx_mode
        self.use_sub = "sub" in self.ctx_mode
        self.use_tef = "tef" in self.ctx_mode
        self.use_face = "face" in self.ctx_mode

        if self.use_video:
            if isinstance(vid_feat_path_or_handler, h5py.File):
                self.vid_feat_h5 = vid_feat_path_or_handler
            else:  # str path
                self.vid_feat_h5 = open_video_features(vid_feat_path_or_handler, h5driver)

        if self.use_sub:
            if isinstance(sub_bert_path_or_handler, h5py.File):
                self.sub_bert_h5 = sub_bert_path_or_handler
            else:  # str path
                self.sub_bert_h5 = h5py.File(sub_bert_path_or_handler, "r", driver=h5driver)

        if self.use_face:
            if isinstance(face_feat_path_or_handler, h5py.File):
                self.face_feat_h5 = face_feat_path_or_handler
            else: # str path
                self.face_feat_h5 = h5py.File(face_feat_path_or_handler, "r", driver=h5driver)
            if isinstance(portrait_feat_path_or_handler, h5py.File):
                self.portrait_feat_h5 = portrait_feat_path_or_handler
            else:
                self.portrait_feat_h5 = h5py.File(portrait_feat_path_or_handler, "r", driver=h5driver)

        self.visual_token_id = 0
        self.text_token_id = 1

    def set_data_mode(self, data_mode):
        """context or query"""
        assert data_mode in ["context", "query"]
        self.data_mode = data_mode

    def load_gt_vid_name_for_query(self, load_gt_video):
        """load_gt_video: bool, affect the returned value of self._get_item_query"""
        if load_gt_video:
            assert "vid_name" in self.query_data[0]
        self.load_gt_video = load_gt_video

    def __len__(self):
        if self.data_mode == "context":
            return len(self.video_data)
        else:
            return len(self.query_data)

    def __getitem__(self, index):
        if self.data_mode == "context":
            return self._get_item_context(index)
        else:
            return self._get_item_query(index)

    def get_query_feat_by_desc_id(self, desc_id):
        query_feat = self.desc_bert_h5[str(desc_id)][:self.max_desc_len]
        if self.normalize_tfeat:
            query_feat = l2_normalize_np_array(query_feat)
        return torch.from_numpy(query_feat)

    def get_face_feat_by_nid(self, appear):
        appear_id = []
        appear_feat = []
        for ap in appear:
            appear_id.append(ap)
            appear_feat.append(self.portrait_feat_h5[ap])
        if len(appear_feat)==0:
            appear_feat.append(np.zeros((512,)))
        return torch.from_numpy(np.mean(appear_feat, axis=0))

    def _get_item_query(self, index):
        """Need to batch"""
        raw_data = self.query_data[index]

        meta = dict(
            desc_id=raw_data["desc_id"],
            desc=raw_data["desc"],
            appear=raw_data["appear"] if "appear" in raw_data else [],
            vid_name=raw_data["vid_name"] if self.load_gt_video else None
        )

        model_inputs = dict()
        query_feat_pos_id = torch.arange(self.max_desc_len, dtype=torch.long)
        query_feat_token_id = torch.full((self.max_desc_len,), self.text_token_id, dtype=torch.long)
        model_inputs["query_pos_id"] = query_feat_pos_id
        model_inputs["query_token_id"] = query_feat_token_id
        model_inputs["query_feat"] = self.get_query_feat_by_desc_id(meta["desc_id"])
        model_inputs["query_face_feat"] = None
        if self.portrait_feat_path_or_handler:
            # 修改 在query加入face
            model_inputs["query_face_feat"] = self.get_face_feat_by_nid(meta["appear"])

        # 修改被我注释
        # if self.use_face:
        #     model_inputs["query_face_feat"] = self.get_face_feat_by_nid(meta["appear"])
        # else:
        #     model_inputs["query_face_feat"] = None
        return dict(meta=meta, model_inputs=model_inputs)

    def get_st_ed_label(self, ts, max_idx):
        """
        Args:
            ts: [st (float), ed (float)] in seconds, ed > st
            max_idx: length of the video

        Returns:
            [st_idx, ed_idx]: int,

        Given ts = [3.2, 7.6], st_idx = 2, ed_idx = 6,
        clips should be indexed as [2: 6), the translated back ts should be [3:9].
        Given ts = [5, 9], st_idx = 3, ed_idx = 6,
        clips should be indexed as [3: 6), the translated back ts should be [4.5:9].
        # TODO which one is better, [2: 5] or [2: 6)
        """
        # TODO ed_idx -= 1, should also modify relevant code in inference.py
        st_idx = min(math.floor(ts[0] / self.clip_length), max_idx)
        ed_idx = min(math.ceil(ts[1] / self.clip_length) - 1, max_idx)  # st_idx could be the same as ed_idx
        return torch.LongTensor([st_idx, ed_idx])

    def _get_item_context(self, index):
        """No need to batch, since it has already been batched here"""
        raw_data = self.video_data[index]

        # initialize with basic data
        meta = dict(
            vid_name=raw_data["vid_name"],
            duration=raw_data["duration"],
        )

        model_inputs = dict()
        ctx_l = 0

        # 修改####################################################################################################
        # k = 20
        # if True:
        #     k = 20
        #     appear_feat = self.appear_feat_h5[meta["vid_name"]][:k]  # (N_clip, D)
        #     if self.normalize_vfeat:
        #         appear_feat = l2_normalize_np_array(appear_feat)
        #     appear_feat_pos_id = torch.arange(k, dtype=torch.long)
        #     appear_feat_token_id = torch.full((k,), 3, dtype=torch.long)  # token type = 3
        #     model_inputs["appear_feat"] = torch.from_numpy(appear_feat)
        #     model_inputs["appear_pos_id"] = appear_feat_pos_id
        #     model_inputs["appear_token_id"] = appear_feat_token_id
        #     ctx_l = len(appear_feat)

        # else:
        #     model_inputs["appear_feat"] = torch.zeros((2, 2))


        #########################################################################################################


        if self.use_video:
            video_feat = self.vid_feat_h5[meta["vid_name"]][:self.max_ctx_len]  # (N_clip, D)
            if self.normalize_vfeat:
                video_feat = l2_normalize_np_array(video_feat)
            visual_feat_pos_id = torch.arange(self.max_ctx_len, dtype=torch.long)
            visual_feat_token_id = torch.full((self.max_ctx_len,), self.visual_token_id, dtype=torch.long)
            model_inputs["video_feat"] = torch.from_numpy(video_feat)
            model_inputs["video_pos_id"] = visual_feat_pos_id
            model_inputs["video_token_id"] = visual_feat_token_id
            ctx_l = len(video_feat)

        else:
            model_inputs["video_feat"] = torch.zeros((2, 2))

        if self.use_sub:  # no need for ctx feature, as the features are already contextulized
            sub_feat = self.sub_bert_h5[meta["vid_name"]][:self.max_ctx_len]  # (N_clips, D_t)
            if self.normalize_tfeat:
                sub_feat = l2_normalize_np_array(sub_feat)
            sub_feat_pos_id = torch.arange(self.max_ctx_len, dtype=torch.long)
            sub_feat_token_id = torch.full((self.max_ctx_len,), self.text_token_id, dtype=torch.long)
            model_inputs["sub_feat"] = torch.from_numpy(sub_feat)
            model_inputs["sub_pos_id"] = sub_feat_pos_id
            model_inputs["sub_token_id"] = sub_feat_token_id
            ctx_l = len(sub_feat)
        else:
            model_inputs["sub_feat"] = torch.zeros((2, 2))

        if self.use_tef:
            ctx_l = meta["duration"] // self.clip_length + 1 if ctx_l == 0 else ctx_l
            tef_st = torch.arange(0, ctx_l, 1.0) / ctx_l
            tef_ed = tef_st + 1.0 / ctx_l
            tef = torch.stack([tef_st, tef_ed], dim=1)  # (N_clips, 2)
            model_inputs["tef_feat"] = tef
        else:
            model_inputs["tef_feat"] = torch.zeros((2, 2))


        if self.use_face:
            # (n_clusters, hidden_dim)
            face_feat = torch.from_numpy(self.get_face_feat_by_video_name(meta["vid_name"]))

            model_inputs["face_feat"] = face_feat
        # 修改被我注释
        if self.use_face:
            face_feat = self.face_feat_h5[meta["vid_name"]][:self.max_ctx_len]  # (N_clip, D)
            face_feat = face_feat[:video_feat.shape[0]]
            #if self.normalize_vfeat:
            #    video_feat = l2_normalize_np_array(video_feat)
            model_inputs["face_feat"] = torch.from_numpy(face_feat)
            ctx_l = len(face_feat)
        else:
            model_inputs["face_feat"] = torch.zeros((2, 2))

        if self.use_video and self.use_tef:
            model_inputs["video_feat"] = torch.cat(
                [model_inputs["video_feat"], model_inputs["tef_feat"]], dim=1)  # (N_clips, D+2)
        if self.use_sub and self.use_tef:
            model_inputs["sub_feat"] = torch.cat(
                [model_inputs["sub_feat"], model_inputs["tef_feat"]], dim=1)  # (N_clips, D_t+2)

        # 以下被我注释
        # if self.use_video and self.use_face:
        #     model_inputs["video_feat"] = torch.cat(
        #         [model_inputs["video_feat"], model_inputs["face_feat"]], dim=1) #(N_clips, D+512)
        return dict(meta=meta, model_inputs=model_inputs)

    def get_face_feat_by_video_name(self, vid_name):
        face_feat = self.face_feat_h5[vid_name][:]
        if False:  # 恒定为 False self.normalize_vfeat = False opt传入的时候就是 false 不看默认值
            face_feat = l2_normalize_np_array(face_feat)
        return face_feat


def start_end_collate(batch):
    batch_meta = [e["meta"] for e in batch]  # seems no need to collate ?

    model_inputs_keys = batch[0]["model_inputs"].keys()
    batched_data = dict()
    max_fixed_length = 100
    for k in model_inputs_keys:
        if "face" in k and batch[0]["model_inputs"][k]==None:
            batched_data[k] = None
            continue
        if "feat" in k and "neg" not in k:
            if "query" in k:
                batched_data[k] = pad_sequences_1d(
                    [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
            if "query" not in k:
                # 我修改: 每个批次video feat 都填充到 100
                batched_data[k] = pad_sequences_1d(
                    [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=100)
                # batched_data[k] = pad_sequences_1d(
                #     [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
        #   生成负样本
        elif "neg_video_feat" in k or "neg_sub_feat" in k:
            batched_data[k] = [pad_sequences_1d(e["model_inputs"][k], dtype=torch.float32, fixed_length=max_fixed_length) for e in batch]

        elif "span" in k:
            #import pdb;pdb.set_trace()
            tmp_paded_span = pad_sequences_1d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32, fixed_length=None)
            batched_data[k] = tmp_paded_span[0]
        elif "iou2d" in k:
            #import pdb;pdb.set_trace()
            tmp_paded_span = pad_sequences_2d(
                [e["model_inputs"][k] for e in batch], dtype=torch.float32)
            batched_data[k] = tmp_paded_span[0]

        #elif k == "st_ed_indices":
        #    batched_data["st_ed_indices"] = torch.stack(
        #        [e["model_inputs"]["st_ed_indices"] for e in batch], dim=0)
        else:
            if "neg" not in k and "weight_hard" not in k:
                batched_data[k] = torch.stack(
                [e["model_inputs"][k] for e in batch], dim=0)
            if "neg" in k and "weight_hard" not in k:
                batched_data[k] = [e["model_inputs"][k] for e in batch]
            if "weight_hard" in k:
                batched_data[k] = torch.tensor([e["model_inputs"][k] for e in batch], dtype=torch.float32)

    return batch_meta, batched_data


def prepare_batch_inputs(batched_model_inputs, device, non_blocking=False, video_msg=None, sub_msg=None, face_msg=None):
    model_inputs = {}
    for k, v in batched_model_inputs.items():
        if v == None:
            model_inputs[k] = None
            if "feat" in k:
                model_inputs[k.replace("feat", "mask")] = None
            continue
        if "feat" in k:
            if "neg_video_feat" == k or "neg_sub_feat" == k:
                model_inputs[k] = [i[0].to(device, non_blocking=non_blocking) for i in v]
                model_inputs[k] = torch.stack(model_inputs[k])
                model_inputs[k.replace("feat", "mask")] = torch.stack([i[1].to(device, non_blocking=non_blocking) for i in v])
            if "neg" not in k:
                model_inputs[k] = v[0].to(device, non_blocking=non_blocking)
                model_inputs[k.replace("feat", "mask")] = v[1].to(device, non_blocking=non_blocking)
            if "neg_visual_token_id" in k or "neg_sub_token_id" in k or "neg_visual_pos_id" in k or "neg_sub_pos_id" in k:
                model_inputs[k] = [i.to(device, non_blocking=non_blocking) for i in v]
        else:
            model_inputs[k] = v.to(device, non_blocking=non_blocking)
    if video_msg != None :
        model_inputs["video_feat"] = video_msg[0]
        model_inputs["video_mask"] = video_msg[1]  # vid2shotid
        model_inputs["video_pos_id"] = video_msg[2]
        model_inputs["sub_feat"] = sub_msg[0]
        model_inputs["sub_mask"] = sub_msg[1]  # vid2shotid
        model_inputs["sub_pos_id"] = sub_msg[2]
        model_inputs["face_G"] = face_msg[0]
        model_inputs["face_S"] = face_msg[1]  # face2shotid
        model_inputs["face_pos_id"] = face_msg[2]

    return model_inputs

if __name__ == '__main__':
    from baselines.crossmodal_moment_localization.config import BaseOptions
    options = BaseOptions().parse()
