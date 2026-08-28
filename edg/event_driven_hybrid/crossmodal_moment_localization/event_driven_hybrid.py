import sys
from venv import logger
from pathlib import Path

import numpy as np
import h5py

sys.path.append(r'/data/hk/tvr_hk/baselines')
sys.path.append(r'/data/hk/tvr_hk')
sys.path.append(r'/hy-tmp/datasets')
sys.path.append(r'/opt/data/private/tvr_hk/baselines/crossmodal_moment_localization')
sys.path.append(r'/opt/data/private/tvr_hk/baselines')
sys.path.append(r'/opt/data/private/tvr_hk')
sys.path.append(r'/opt/data/private')

import yaml
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from types import SimpleNamespace
from scipy.optimize import linear_sum_assignment
from crossmodal_moment_localization.model_components import \
    BertAttention, PositionEncoding, LinearLayer, BertSelfAttention, TrainablePositionalEncoding, ConvEncoder
from edg.utils.model_utils import RNNEncoder, MLP
from edg.utils.basic_utils import load_config, load_jsonl
from crossmodal_moment_localization.transformer.bert import BertEncoder
from crossmodal_moment_localization.transformer.bert_embed import BertEmbeddings
from crossmodal_moment_localization.contrastive import batch_video_query_loss, hard_video_query_loss
from start_end_dataset_with_face import pad_sequences_1d

from third_party.actionformer.libs.modeling import backbones, expend_mask
from cluster_merge import *
from check_series import *

visual_action_config = {
    "n_in": 256,  # input feature dimension
    "n_embd": 256,  # embedding dimension (after convolution)
    "n_head": 8,  # number of head for self-attention in transformers
    "n_embd_ks": 3,  # conv kernel size of the embedding network
    "max_len": 128,  # max sequence length
    # (#convs, #stem transformers, #branch transformers)
    # "arch": (0, 2, 5),  # 大娃
    # "arch": (0, 2, 2),  # 二娃最棒
    # "arch": (0, 1, 2),  # 四娃
    "arch": (0, 1, 3),  # 模拟 XML
    # "mha_win_size": [-1]*6,  # size of local window for mha
    # "mha_win_size": [-1, 16, 16, 8, 8, 8],  # size of local window for mha
    # "mha_win_size": [-1, 16, 8],  # size of local window for mha
    # "mha_win_size": [-1, 8, 4, 4],  # size of local window for mha
    "mha_win_size": [4, 4, 4, 4],  # size of local window for mha
    "scale_factor": 2,  # dowsampling rate for the branch
    # "with_ln": False,       # if to attach layernorm after conv
    "with_ln": True,  # if to attach layernorm after conv
    "attn_pdrop": 0.0,  # dropout rate for the attention map
    "proj_pdrop": 0.0,  # dropout rate for the projection / MLP
    "path_pdrop": 0.0,  # droput rate for drop path
    "use_abs_pe": False,  # use absolute position embedding
    "use_rel_pe": False,  # use relative position embedding
}
sub_action_config = {
    "n_in": 256,  # input feature dimension
    "n_embd": 256,  # embedding dimension (after convolution)
    "n_head": 8,  # number of head for self-attention in transformers
    "n_embd_ks": 3,  # conv kernel size of the embedding network
    "max_len": 128,  # max sequence length
    # (#convs, #stem transformers, #branch transformers)
    # "arch": (0, 2, 5),  # 大娃
    # "arch": (0, 2, 2),  # 二娃最棒
    # "arch": (0, 1, 2),  # 四娃
    "arch": (0, 1, 3),  # 模拟 XML
    # "mha_win_size": [-1]*6,  # size of local window for mha
    # "mha_win_size": [-1, 16, 16, 8, 8, 8],  # size of local window for mha
    # "mha_win_size": [-1, 16, 8],  # size of local window for mha
    # "mha_win_size": [-1, 8, 4, 4],  # size of local window for mha
    "mha_win_size": [4, 4, 4, 4],  # size of local window for mha
    "scale_factor": 2,  # dowsampling rate for the branch
    # "with_ln": False,       # if to attach layernorm after conv
    "with_ln": True,  # if to attach layernorm after conv
    "attn_pdrop": 0.0,  # dropout rate for the attention map
    "proj_pdrop": 0.0,  # dropout rate for the projection / MLP
    "path_pdrop": 0.0,  # droput rate for drop path
    "use_abs_pe": False,  # use absolute position embedding
    "use_rel_pe": False,  # use relative position embedding
}

base_bert_layer_config = dict(
    hidden_size=768,
    intermediate_size=768,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_attention_heads=4,
)

# 修改
base_bert_layer_config_1 = dict(
    hidden_size=256,
    intermediate_size=768,
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,
    num_attention_heads=4,
)

xml_base_config = edict(
    merge_two_stream=True,  # merge only the scores
    cross_att=True,  # cross-attention for video and subtitles
    span_predictor_type="conv",
    encoder_type="transformer",  # cnn, transformer, lstm, gru
    add_pe_rnn=False,  # add positional encoding for RNNs, (LSTM and GRU)
    visual_input_size=2048,  # changes based on visual input type
    query_input_size=768,
    sub_input_size=768,
    hidden_size=500,  #
    conv_kernel_size=5,  # conv kernel_size for st_ed predictor
    stack_conv_predictor_conv_kernel_sizes=-1,  # Do not use
    conv_stride=1,  #
    max_ctx_l=100,
    max_desc_l=30,
    input_drop=0.1,  # dropout for input
    drop=0.1,  # dropout for other layers
    n_heads=4,  # self attention heads
    ctx_mode=
    "video_sub",  # which context are used. 'video', 'sub' or 'video_sub'
    margin=0.1,  # margin for ranking loss
    ranking_loss_type="hinge",  # loss type, 'hinge' or 'lse'
    lw_neg_q=1,  # loss weight for neg. query and pos. context
    lw_neg_ctx=1,  # loss weight for pos. query and neg. context
    lw_st_ed=1,  # loss weight for st ed prediction
    use_hard_negative=
    False,  # use hard negative at video level, we may change it during training.
    hard_pool_size=20,
    use_self_attention=True,
    no_modular=False,
    pe_type="none",  # no positional encoding
    initializer_range=0.02,
)

HBI_config = edict(
    sample_ratio = 1,
    embed_dim = 256,
    dim_out = 256,
    k = 3,
    num_heads = 8
)


class EventDrivenHybrid(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)

        # 修改
        self.sub_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)

        self.ctx_token_pos_embed = TrainablePositionalEncoding(
            # max_position_embeddings=config.max_ctx_l,
            max_position_embeddings=212,  # fit 拼接过后的 video
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)
        # self.ctx_token_pos_embed = TrainablePositionalEncoding(
        #     max_position_embeddings=config.max_ctx_l,
        #     type_vocab_size=2,
        #     hidden_size=config.hidden_size,
        #     dropout=config.input_drop)
        self.V1ctx_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)
        self.V2ctx_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)

        # 用于 pad 适应 Actionformer 的位置、类型embedding
        self.myctx_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=128,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)

        self.query_vid2clip_linear = LinearLayer(in_hsz=config.hidden_size,
                                                 out_hsz=config.hidden_size)
        self.query_vid2clip_linear2 = LinearLayer(in_hsz=config.hidden_size,
                                                  out_hsz=config.hidden_size)
        self.query_face2token_linear = LinearLayer(in_hsz=512,
                                                   out_hsz=config.hidden_size)
        self.merge_query_and_face_linear = LinearLayer(
            in_hsz=config.hidden_size * 2, out_hsz=config.hidden_size)
        #self.face_linear = LinearLayer(
        #    in_hsz=512, out_hsz=config.hidden_size)
        #self.span_conv_vq = LinearLayer(
        #    in_hsz=2*config.hidden_size, out_hsz=1)
        #self.span_conv_sq = LinearLayer(
        #    in_hsz=2*config.hidden_size, out_hsz=1)
        self.span_conv_vq = nn.Conv1d(in_channels=2 * config.hidden_size,
                                      out_channels=1,
                                      kernel_size=5,
                                      padding=2)
        self.span_conv_sq = nn.Conv1d(in_channels=2 * config.hidden_size,
                                      out_channels=1,
                                      kernel_size=5,
                                      padding=2)
        #self.conv_st_ctx = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=5, padding=2)
        #self.conv_ed_ctx = nn.Conv1d(in_channels=config.hidden_size, out_channels=config.hidden_size, kernel_size=5, padding=2)
        self.metric_weight = nn.Parameter(torch.ones(3))
        #######
        self.is_biaffine = False
        if self.is_biaffine:
            self.start_layer = MLP(n_in=config.hidden_size,
                                   n_out=config.hidden_size,
                                   dropout=0.33)
            self.end_layer = MLP(n_in=config.hidden_size,
                                 n_out=config.hidden_size,
                                 dropout=0.33)
        """
        self.query_input_proj = LinearLayer(config.query_input_size,
                                            config.hidden_size,
                                            layer_norm=True,
                                            dropout=config.input_drop,
                                            relu=True)
        if config.encoder_type == "transformer":  # self-att encoder
            self.query_encoder = BertAttention(edict(
                hidden_size=config.hidden_size,
                intermediate_size=config.hidden_size,
                hidden_dropout_prob=config.drop,
                attention_probs_dropout_prob=config.drop,
                num_attention_heads=config.n_heads,
            ))
        elif config.encoder_type == "cnn":
            self.query_encoder = ConvEncoder(
                kernel_size=5,
                n_filters=config.hidden_size,
                dropout=config.drop
            )
        elif config.encoder_type in ["gru", "lstm"]:
            self.query_encoder = RNNEncoder(
                word_embedding_size=config.hidden_size,
                hidden_size=config.hidden_size // 2,
                bidirectional=True,
                n_layers=1,
                rnn_type=config.encoder_type,
                return_outputs=True,
                return_hidden=False
            )
        """
        conv_cfg = dict(in_channels=1,
                        out_channels=1,
                        kernel_size=config.conv_kernel_size,
                        stride=config.conv_stride,
                        padding=config.conv_kernel_size // 2,
                        bias=False)

        cross_att_cfg = edict(hidden_size=config.hidden_size,
                              num_attention_heads=config.n_heads,
                              attention_probs_dropout_prob=config.drop)

        config_dir = Path(__file__).resolve().parent
        model_config = load_config(str(config_dir / "model_config.json"))
        print(model_config)
        my_model_config = load_config(str(config_dir / "my_model_config.json"))
        self.t2vVLAD = NVLDModel(my_model_config.netvlad_config)
        SQAN_config = config_dir / "LG_config.yaml"
        # 加载配置文件
        with open(SQAN_config, 'r') as f:
            SQAN_config = yaml.safe_load(f)
        self.HBI_video_pool = HBIPooling(HBI_config)
        self.HBI_sub_pool = HBIPooling(HBI_config)
        self.bidirect = BidirectionalAttention(video_dim=256)
        # 初始化 QuerySequenceEncoder 实例
        self.l2g_query_video_encoder = SequentialQueryAttention(SQAN_config)
        self.l2g_query_sub_encoder = SequentialQueryAttention(SQAN_config)
        self.v_DQALoss = DQALoss(SQAN_config)
        self.s_DQALoss = DQALoss(SQAN_config)
        self.kl = KL()
        self.cal_video_score = CalEventLevel(hidden_size=256)
        self.cal_sub_score = CalEventLevel(hidden_size=256)
        self.video_grounding = FineGrainGround(config)
        self.sub_grounding = FineGrainGround(config)
        self.share_norm_loss = ShareNormLoss()
        self.conquer_cross_visual = CrossAttention(dim=256, context_dim=256)
        self.conquer_cross_sub = CrossAttention(dim=256, context_dim=256)
        self.vs_score = VS_score()
        self.consistency_loss = TwoStageConpare()
        self.use_sub = "sub" in config.ctx_mode
        self.use_face = "face" in config.ctx_mode
        self.triplet = True  # triplet using ThreeModalEncoder
        #self.triplet = False # sepration using video_query_Encoder and sub_query_Encoder

        # 修改
        # self.local = LocalAttention(action_config=visual_action_config, config=config)
        self.videolocal = VideoLocalAttention(
            action_config=visual_action_config)
        self.subglobal = SubLocalAttention(action_config=sub_action_config)
        self.adapt = AdaptFace(input_dim=256, output_dim=256, n_heads=8)
        if self.use_sub:
            # 修改 不用改
            # if self.use_face:
            #    config.visual_input_size += 512
            self.videoEncoder = TwoModalEncoder(
                config=model_config.bert_config,
                img_dim=config.visual_input_size,
                text_dim=config.sub_input_size,
                hidden_dim=config.hidden_size,
                split_num=config.max_ctx_l,
                with_emb=True,
            )
            self.myvideoEncoder = MyTwoModalEncoder(
                config=model_config.bert_config,
                split_num=config.max_ctx_l,
                with_emb=True,
            )

            if self.triplet:
                self.triple_Encoder = ThreeModalEncoder(
                    config=model_config.triplet_config,
                    img_dim=config.hidden_size,
                    text_dim=config.hidden_size,
                    query_dim=config.hidden_size,
                    hidden_dim=config.hidden_size,
                    split_num=config.max_ctx_l,
                )
                self.my_triple_Encoder = SVMR_Train(
                    config=config,
                    model_config=model_config,
                )
            else:
                self.video_query_Encoder = TwoModalEncoder(
                    config=model_config.triplet_config,
                    img_dim=config.hidden_size,
                    text_dim=config.hidden_size,
                    hidden_dim=config.hidden_size,
                    split_num=config.max_ctx_l,
                    with_emb=False,
                )

                self.sub_query_Encoder = TwoModalEncoder(
                    config=model_config.triplet_config,
                    img_dim=config.hidden_size,
                    text_dim=config.hidden_size,
                    hidden_dim=config.hidden_size,
                    split_num=config.max_ctx_l,
                    with_emb=False,
                )
        else:
            self.videoEncoder = OneModalEncoder(
                config=model_config.bert_config,
                input_dim=config.visual_input_size,
                hidden_dim=config.hidden_size,
            )

            self.subEncoder = OneModalEncoder(
                config=model_config.bert_config,
                input_dim=config.sub_input_size,
                hidden_dim=config.hidden_size,
            )
        # 修改
        self.subEncoder = OneModalEncoder(
            config=model_config.bert_config,
            input_dim=config.sub_input_size,
            hidden_dim=config.hidden_size,
        )
        self.mysubEncoder = MyOneModalEncoder(
            config=model_config.bert_config,
            input_dim=config.sub_input_size,
            hidden_dim=config.hidden_size,
        )
        self.queryEncoder = OneModalEncoder(
            config=model_config.query_bert_config,
            input_dim=config.query_input_size,
            hidden_dim=config.hidden_size,
        )
        #self.biaffineLayer = BiaffineLayer(in_size1=config.hidden_size, in_size2=config.hidden_size, class_size=1)
        if self.is_biaffine:
            self.biaffineLayer = Biaffine(n_in=config.hidden_size,
                                          bias_x=False,
                                          bias_y=False)
        self.use_video = "video" in config.ctx_mode
        """
        if self.use_video:
            self.video_input_proj = LinearLayer(config.visual_input_size,
                                                config.hidden_size,
                                                layer_norm=True,
                                                dropout=config.input_drop,
                                                relu=True)
            self.video_encoder1 = copy.deepcopy(self.query_encoder)
            self.video_encoder2 = copy.deepcopy(self.query_encoder)
            if self.config.cross_att:
                self.video_cross_att = BertSelfAttention(cross_att_cfg)
                self.video_cross_layernorm = nn.LayerNorm(config.hidden_size)
            else:
                if self.config.encoder_type == "transformer":
                    self.video_encoder3 = copy.deepcopy(self.query_encoder)
        """
        self.video_query_linear = nn.Linear(config.hidden_size,
                                            config.hidden_size)
        if config.span_predictor_type == "conv":
            if not config.merge_two_stream:
                self.video_st_predictor = nn.Conv1d(**conv_cfg)
                self.video_ed_predictor = nn.Conv1d(**conv_cfg)
        elif config.span_predictor_type == "cat_linear":
            self.video_st_predictor = nn.ModuleList(
                [nn.Linear(config.hidden_size, 1) for _ in range(6)])
            self.video_ed_predictor = nn.ModuleList(
                [nn.Linear(config.hidden_size, 1) for _ in range(6)])
        """
        if self.use_sub:
            self.sub_input_proj = LinearLayer(config.sub_input_size,
                                              config.hidden_size,
                                              layer_norm=True,
                                              dropout=config.input_drop,
                                              relu=True)
            self.sub_encoder1 = copy.deepcopy(self.query_encoder)
            self.sub_encoder2 = copy.deepcopy(self.query_encoder)
            if self.config.cross_att:
                self.sub_cross_att = BertSelfAttention(cross_att_cfg)
                self.sub_cross_layernorm = nn.LayerNorm(config.hidden_size)
            else:
                if self.config.encoder_type == "transformer":
                    self.sub_encoder3 = copy.deepcopy(self.query_encoder)
        """
        self.sub_query_linear = nn.Linear(config.hidden_size,
                                          config.hidden_size)
        if config.span_predictor_type == "conv":
            if not config.merge_two_stream:
                self.sub_st_predictor = nn.Conv1d(**conv_cfg)
                self.sub_ed_predictor = nn.Conv1d(**conv_cfg)
        elif config.span_predictor_type == "cat_linear":
            self.sub_st_predictor = nn.ModuleList(
                [nn.Linear(config.hidden_size, 1) for _ in range(2)])
            self.sub_ed_predictor = nn.ModuleList(
                [nn.Linear(config.hidden_size, 1) for _ in range(2)])

        self.modular_vector_mapping = nn.Linear(in_features=config.hidden_size,
                                                out_features=self.use_sub +
                                                self.use_video,
                                                bias=False)
        self.modular_vector_mapping_word = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.use_sub + self.use_video,
            bias=False)
        self.modular_vector_mapping_word2 = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.use_sub + self.use_video,
            bias=False)
        self.modular_vector_mapping_word3 = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.use_sub + self.use_video,
            bias=False)
        self.modular_vector_mapping_word_v_s = nn.Linear(
            in_features=config.hidden_size, out_features=2, bias=False)
        self.modular_vector_mapping_phrase = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.use_sub + self.use_video,
            bias=False)
        self.modular_vector_mapping_sentence = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.use_sub + self.use_video,
            bias=False)
        self.modular_vector_mapping2 = nn.Linear(
            in_features=config.hidden_size,
            out_features=self.use_sub + self.use_video,
            bias=False)

        self.temporal_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.span_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.biaffine_criterion = nn.BCEWithLogitsLoss(reduction="none")
        self.l2_criterion = nn.PairwiseDistance(p=2)
        self.nce_criterion = MILNCELoss(reduction='mean')

        if config.merge_two_stream and config.span_predictor_type == "conv":
            if self.config.stack_conv_predictor_conv_kernel_sizes == -1:
                self.merged_st_predictor = nn.Conv1d(**conv_cfg)
                self.merged_ed_predictor = nn.Conv1d(**conv_cfg)
            else:
                print("Will be using  multiple Conv layers for prediction.")
                self.merged_st_predictors = nn.ModuleList()
                self.merged_ed_predictors = nn.ModuleList()
                num_convs = len(
                    self.config.stack_conv_predictor_conv_kernel_sizes)
                for k in self.config.stack_conv_predictor_conv_kernel_sizes:
                    conv_cfg = dict(in_channels=1,
                                    out_channels=1,
                                    kernel_size=k,
                                    stride=config.conv_stride,
                                    padding=k // 2,
                                    bias=False)
                    self.merged_st_predictors.append(nn.Conv1d(**conv_cfg))
                    self.merged_ed_predictors.append(nn.Conv1d(**conv_cfg))
                self.combine_st_conv = nn.Linear(num_convs, 1, bias=False)
                self.combine_ed_conv = nn.Linear(num_convs, 1, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """ Initialize the weights."""

        def re_init(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                # cf https://github.com/pytorch/pytorch/pull/5617
                module.weight.data.normal_(mean=0.0,
                                           std=self.config.initializer_range)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            elif isinstance(module, nn.Conv1d):
                module.reset_parameters()
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()

        self.apply(re_init)

    def set_hard_negative(self, use_hard_negative, hard_pool_size):
        """use_hard_negative: bool; hard_pool_size: int, """
        self.config.use_hard_negative = use_hard_negative
        self.config.hard_pool_size = hard_pool_size

    def set_train_st_ed(self, lw_st_ed):
        """pre-train video retrieval then span prediction"""
        self.config.lw_st_ed = lw_st_ed

    def bce_rescale_loss(self, scores, masks, targets):

        min_iou, max_iou, bias, multiple = 0.5, 1.0, 0.5, 1.5
        joint_prob = torch.sigmoid(scores) * masks
        #print("joint, ", joint_prob[0].sum(), joint_prob[0,:5,5:10])
        target_prob = (targets - min_iou) * (1 - bias) / (max_iou - min_iou)
        #print("target, ", target_prob[0].sum(), target_prob[0,:5,5:10])
        target_prob[target_prob > 0] += bias
        #target_prob[target_prob > 0] *= multiple
        target_prob[target_prob > 1] = 1
        target_prob[target_prob < 0] = 0
        pos_prob = torch.where(targets > 0, joint_prob, (targets > 0).float())
        pos_prob_mask = targets > 0
        #print("pos, ", pos_prob[0].sum())
        loss = F.binary_cross_entropy(
            joint_prob, target_prob, reduction='none') * masks
        #import pdb;pdb.set_trace()
        #weights = torch.where(pos_prob_mask==0, pos_prob_mask+.05, pos_prob_mask+1.)
        #loss = loss * weights
        #loss_biaffine = self.biaffine_criterion(out_2d, self.iou2d_mask)
        #loss_biaffine = loss_biaffine * weights
        #loss_biaffine = torch.sum(loss_biaffine) / (torch.count_nonzero(self.iou2d_mask)+1e-12)
        loss_value = torch.sum(loss) / (torch.sum(masks) + 1e-12)
        return loss_value, joint_prob

    #def __init__(self, config, video_modality,
    #             visual_dim=4352, text_dim= 768,
    #             query_dim=768, hidden_dim = 768,split_num=100,):
    #    super(VideoQueryEncoder, self).__init__()
    #    self.use_sub = len(video_modality) > 1
    def aggregate_vlad_base_hungarian(self,
                                      vlad_ind_local_ind_list,
                                      vlad_feat,
                                      local_feat,
                                      local_feat_mask,
                                      method="Hadamard"):
        """基于匈牙利算法进行聚合"""
        # assert vlad_ind.size() == local_ind.size(), "聚类中心不对"
        device = vlad_feat.device
        # 每个batch内匈牙利算法分配方案下标
        vlad_ind = [item[0] for item in vlad_ind_local_ind_list]
        local_ind = [item[1] for item in vlad_ind_local_ind_list]
        # 这里的索引还很奇怪: 0代表行，1代表列
        vlad_feat = [
            vlad_feat[i].to("cpu").index_select(0, item)
            for i, item in enumerate(vlad_ind)
        ]
        local_feat = [
            local_feat[i].to("cpu").index_select(0, item)
            for i, item in enumerate(local_ind)
        ]
        local_feat_mask = [
            local_feat_mask[i].to("cpu").index_select(0, item)
            for i, item in enumerate(local_ind)
        ]
        vlad_feat = torch.stack(vlad_feat).to(device=device)
        local_feat = torch.stack(local_feat).to(device=device)
        local_feat_mask = torch.stack(local_feat_mask).to(device=device)
        # vlad_feat = vlad_feat.index_select(1, vlad_ind)
        # local_feat = local_feat.index_select(1, local_ind)
        # local_feat_mask = local_feat_mask.index_select(1, local_ind)
        # 聚合方法
        # 使用Hadamard乘积聚合
        if method == "Hadamard":
            c, l = vlad_feat.size()[1], local_feat.size()[1]
            # 应用掩码，当局部特征失效时（mask==0），使用vlad_feat本身
            masked_local_feat = local_feat * local_feat_mask.unsqueeze(-1).float() + \
                vlad_feat * (1 - local_feat_mask.unsqueeze(-1).float())
            # fusion: [bsz, c, d]
            fusion_feat = vlad_feat * masked_local_feat
            return fusion_feat
        elif method == "Average":
            # Average：简单平均，同时应用掩码确保无效特征不影响计算
            valid_counts = local_feat_mask.unsqueeze(-1).float()
            masked_local_feat = local_feat * valid_counts
            fusion_feat = (vlad_feat + masked_local_feat) / (1 + valid_counts
                                                             )  # 防止除以0
            return fusion_feat
        else:
            raise ValueError("Invalid aggregation method")

    def pad_few_video(self, vlad_feat, local_feat, local_mask):
        # 填充较少token的feat适应Hungary
        c, l = vlad_feat.size()[1], local_feat.size()[1]
        padding_size = c - l
        if padding_size == 0:
            return local_feat, local_mask
        local_feat = F.pad(local_feat, (0, 0, 0, padding_size), "constant", 0)
        local_mask = F.pad(local_mask, (0, padding_size), "constant", 0)
        return local_feat, local_mask

    def batch_cosine_similarity(self, t1, t2, t2_mask):
        """计算两个批次的余弦相似度

        Args:
            t1 (torch.tensor): one feature tensor
            t2 (torch.tensor): another feature tensor

        Returns:
            torch: cosine similarity matrix
        """
        # t1.shape == (bsz, c, d)
        # t2.shape == (bsz, l, d)
        if len(t1.size()) == 2:
            bsz = t2.size()[0]
            t1 = t1.unsqueeze(0).repeat(bsz, 1, 1)

        # Normalize each vector in t1 and t2 along the last dimension
        t1_norm = t1 / t1.norm(dim=-1, keepdim=True)
        # Mask invalid t2 data by setting them to zero before normalization
        t2 = t2 * t2_mask.unsqueeze(
            -1)  # Ensure t2_mask is broadcasted correctly

        # Avoid division by zero for t2_norm by adding a very small value to the norm
        t2_norm = t2 / (t2.norm(dim=-1, keepdim=True) + 1e-8)
        # Compute cosine similarity
        # We use einsum to compute dot products between all pairs (c, l) in each batch
        # einsum allows summing over the last dimension (d) of both tensors
        cosine_sim = torch.einsum('bcd,bld->bcl', t1_norm, t2_norm)

        # Normalized to the interval [0,1]
        cosine_sim = (cosine_sim + 1) / 2
        return cosine_sim

    def vlad_align_Local(self, vlad_feat, local_feat, local_mask):
        """计算NetVLAD和local feat 之间的匹配

        Args:
            vlad_feat (torch.tensor): gotten from NetVLAD feature extractor
            local_feat (torch.tensor): gotten from local feature extractor
        """
        # 计算两个特征之间的相似度[0, 1]
        pad_local_feat, pad_local_mask = self.pad_few_video(
            vlad_feat, local_feat, local_mask)
        cos_sim = self.batch_cosine_similarity(vlad_feat, pad_local_feat,
                                               pad_local_mask)

        # 计算距离矩阵的最小值和最大值
        min_val = torch.min(cos_sim)
        max_val = torch.max(cos_sim)

        # 对距离矩阵进行归一化
        normalized_cos_sim = (cos_sim - min_val) / (max_val - min_val)

        # 匈牙利算法目标是优化cost最小值, 所以我们需要将距离转换为cost
        cost = 1 - normalized_cos_sim

        # 使用匈牙利算法找到最小权匹配
        # 假设vlad_feat(t1)是人, local_feat(t2)是任务，人多任务少的匈牙利算法
        vlad_ind_local_ind_list = self.Hungarian(cost, )
        return vlad_ind_local_ind_list, pad_local_mask

    def Hungarian(
        self,
        cost,
    ):
        """匈牙利算法

        Args:
            cost (torch.tensor): 代价矩阵, 每一行表示NetVLAD特征和local特征之间的相似度
        """
        # linear_sum_assignment只支持2维矩阵计算
        n = cost.size()[0]
        assignments = []
        for i in range(n):
            vlad_ind, local_ind = linear_sum_assignment(cost[i].cpu().detach())
            vlad_ind = torch.from_numpy(vlad_ind)
            local_ind = torch.from_numpy(local_ind)
            assignments.append((vlad_ind, local_ind))
        return assignments
        # # 创建一个全零矩阵
        # valid_matrix = torch.zeros((n, n))

        # # 在相应的位置上设置为 1
        # valid_matrix[vlad_ind, local_ind] = 1
        # if return_absolute:
        #     cos_sim_matrix = valid_matrix
        # else:
        #     valid_matrix = valid_matrix.bool()
        #     cos_sim_matrix = cos_sim[valid_matrix]

        # if return_loss:
        #     total_cost = cost[valid_matrix.bool()].sum()
        #     return vlad_ind, local_ind, cos_sim_matrix, total_cost
        # return vlad_ind, local_ind, cos_sim_matrix
    def fine_grain_svmr(self, one_query_feat, topk_video_feat, topk_sub_feat,
                        one_query_mask, topk_video_mask, topk_sub_mask):
        # video_fine_grain_w2ctx: [qbsz, vbsz, w, c]
        video_fine_grain_w2ctx = self.cal_video_score(
            one_query_feat.unsqueeze(0).repeat(topk_video_feat.size()[0], 1,
                                               1),
            topk_video_feat,
            one_query_mask.unsqueeze(0).repeat(topk_video_feat.size()[0], 1),
            topk_video_mask,
            is_eval=True)
        sub_fine_grain_w2ctx = self.cal_sub_score(
            one_query_feat.unsqueeze(0).repeat(topk_video_feat.size()[0], 1,
                                               1),
            topk_sub_feat,
            one_query_mask.unsqueeze(0).repeat(topk_video_feat.size()[0], 1),
            topk_sub_mask,
            is_eval=True)
        # video_fine_grain_ctx_scores: [1, vbsz, c](infer)
        video_fine_grain_ctx_scores_st, video_fine_grain_ctx_scores_ed = self.video_grounding(
            video_fine_grain_w2ctx, topk_video_mask)
        sub_fine_grain_ctx_scores_st, sub_fine_grain_ctx_scores_ed = self.sub_grounding(
            sub_fine_grain_w2ctx, topk_sub_mask)
        # _st: [1, vbsz, c]
        _st = (video_fine_grain_ctx_scores_st +
               sub_fine_grain_ctx_scores_st) / 2
        _ed = (video_fine_grain_ctx_scores_ed +
               sub_fine_grain_ctx_scores_ed) / 2
        # _st: [1, vbsz, c]
        return _st.unsqueeze(0), _ed.unsqueeze(0)

    def fine_grain_svmr_train(self, word_feat, topk_video_feat, topk_sub_feat,
                              word_mask, topk_video_mask, topk_sub_mask):
        # video_fine_grain_w2ctx: [qbsz, vbsz, w, c]
        video_fine_grain_w2ctx = self.cal_video_score(word_feat,
                                                      topk_video_feat,
                                                      word_mask,
                                                      topk_video_mask,
                                                      is_eval=True)
        sub_fine_grain_w2ctx = self.cal_sub_score(word_feat,
                                                  topk_sub_feat,
                                                  word_mask,
                                                  topk_sub_mask,
                                                  is_eval=True)

        # video_fine_grain_ctx_scores: [1, vbsz, c](infer)
        video_fine_grain_ctx_scores_st, video_fine_grain_ctx_scores_ed = self.video_grounding(
            video_fine_grain_w2ctx, topk_video_mask)
        sub_fine_grain_ctx_scores_st, sub_fine_grain_ctx_scores_ed = self.sub_grounding(
            sub_fine_grain_w2ctx, topk_sub_mask)
        # _st: [1, vbsz, c]
        _st = (video_fine_grain_ctx_scores_st +
               sub_fine_grain_ctx_scores_st) / 2
        _ed = (video_fine_grain_ctx_scores_ed +
               sub_fine_grain_ctx_scores_ed) / 2
        if word_feat.size()[0] == 1:
            # _st: [1, vbsz, c]
            return _st.unsqueeze(0), _ed.unsqueeze(0)
        if word_feat.size()[0] == topk_video_feat.size()[0]:
            return _st, _ed

    def forward(
        self,
        query_feat,
        query_pos_id,
        query_token_id,
        query_mask,
        video_feat,
        video_pos_id,
        video_token_id,
        video_mask,
        face_feat,
        face_mask,
        query_face_feat,
        query_face_mask,
        sub_feat,
        sub_pos_id,
        sub_token_id,
        sub_mask,
        tef_feat,
        tef_mask,
        st_ed_indices,
        span_mask,
        iou2d,
        appear_feat=None,
        appear_pos_id=None,
        appear_token_id=None,
        appear_mask=None,
        appear_ind=None,
    ):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            face_feat:  (N, Lv, Dface) or None
            face_mask:  (N, Lv) or None
            query_face_feat:  (N, 1, Dface) or None
            query_face_mask:  (N, 1) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            tef_feat: (N, Lv, 2) or None,
            tef_mask: (N, Lv) or None,
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
            span_mask: (N, Lv), torch.LongTensor, 1s between 1st and 2nd idx, 0s others
            iou2d: (N, Lv, Lv), torch.FloatTensor
        """
        _, ml = video_mask.size()
        video_feat, video_mask = self.videolocal(video_feat, video_mask,
                                                 self.V1ctx_token_pos_embed)
        sub_feat, sub_mask = self.subglobal(sub_feat, sub_mask,
                                            self.V2ctx_token_pos_embed)
        features = {
            "video_feat_1": video_feat[0][
                :,
                :ml,
            ],
            "video_feat_2": video_feat[1],
            "video_feat_3": video_feat[2],
            "video_feat_1_mask": video_mask[0][
                :,
                :ml,
            ],
            "video_feat_2_mask": video_mask[1],
            "video_feat_3_mask": video_mask[2],
            "sub_feat_1": sub_feat[0][
                :,
                :ml,
            ],
            "sub_feat_2": sub_feat[1],
            "sub_feat_3": sub_feat[2],
            "sub_feat_1_mask": sub_mask[0][
                :,
                :ml,
            ],
            "sub_feat_2_mask": sub_mask[1],
            "sub_feat_3_mask": sub_mask[2],
        }
        features["VR_video_feat"], features[
            "VR_video_feat_mask"] = self.HBI_video_pool(
                features["video_feat_1"],
                features["video_feat_1_mask"],
            )
        features["VR_sub_feat"], features[
            "VR_sub_feat_mask"] = self.HBI_sub_pool(
                features["sub_feat_1"],
                features["sub_feat_1_mask"],
            )
        v_s_feat = torch.cat(
            [features["VR_video_feat"], features["VR_sub_feat"]], dim=1)
        v_s_mask = torch.cat(
            [features["VR_video_feat_mask"], features["VR_sub_feat_mask"]],
            dim=1)
        features["VR_VLAD"] = self.t2vVLAD(v_s_feat, v_s_mask)
        features["VR_VLAD_mask"] = features["VR_VLAD"].new_ones(
            features["VR_VLAD"].size()[:-1])
        # video_feat1 = video_feat[:, :ml]
        # sub_feat1 = sub_feat[:, :ml]
        # video_mask = video_mask[:, :ml]
        # sub_mask = sub_mask[:, :ml]
        # 不对 query 的 face 进行 concate
        query_face_feat = None
        if query_face_feat == None:
            query_face_hidden = None
        else:
            query_face_hidden = self.query_face2token_linear(query_face_feat)

        # query 分层
        # 修改: 被注释
        query_feat1, self.query_input_proj = self.query_embedding(
            query_feat, query_mask)
        # 待定是否加入人脸
        if not query_face_hidden == None:
            query_feat1, query_mask = self.extend_query_with_face(
                query_feat1, query_mask, query_face_hidden)
        word_feat = query_feat1
        word_mask = query_mask


        # 修改
        iou2d_label = iou2d
        query_context_scores, st_prob, ed_prob, tmp_querys1, tmp_querys2, span_prob1, span_prob2, video_query, sub_query, video_sub_query, out_2d, \
            loss_vid_kl, loss_sub_kl, loss_vid_kl_cross, loss_sub_kl_cross, sample_mask, vs_loss, triplet_loss, \
                 consistency_loss, query_diverse_loss, v2q_ce_loss = \
            self.get_pred_from_triplet(word_feat, word_mask, features, query_face_hidden, span_mask)
        query_diverse_loss = 0.05 * query_diverse_loss
        loss_biaffine = 0
        # (N, L, L) the lower part becomes zeros, start_idx < ed_idx
        #upper_product = np.triu(self.iou2d_label.shape, k=1)
        iou2d_mask = torch.ones_like(iou2d_label)
        iou2d_mask = torch.triu(iou2d_mask, diagonal=1)
        if self.is_biaffine:

            loss_biaffine = self.bce_rescale_loss(out_2d, iou2d_mask,
                                                  iou2d_label)[0]

        loss_vcl = 0
        if True:
            mid_video_q2ctx_scores = self.my_get_unnormalized_video_level_scores(
                video_query,
                features["VR_video_feat"],
            )
            mid_sub_q2ctx_scores = self.my_get_unnormalized_video_level_scores(
                sub_query, features["VR_sub_feat"])
            # mid_video_sub_q2ctx_scores = self.my_get_unnormalized_video_level_scores(
            #     video_sub_query, features["VR_VLAD"])
            mid_video_q2ctx_scores, _ = torch.max(mid_video_q2ctx_scores,
                                                  dim=1)
            mid_sub_q2ctx_scores, _ = torch.max(mid_sub_q2ctx_scores, dim=1)
            # mid_video_sub_q2ctx_scores, _ = torch.max(
            #     mid_video_sub_q2ctx_scores, dim=1)
            mid_q2ctx_scores = (mid_video_q2ctx_scores + mid_sub_q2ctx_scores +
                                0) / 3.0
            loss_vcl = self.nce_criterion.my_forward(mid_q2ctx_scores)

        self.with_span = False
        loss_span1 = 0
        loss_span2 = 0
        #import pdb;pdb.set_trace()
        if self.with_span:
            weights = torch.where(span_mask == 0, span_mask + 1.,
                                  span_mask * 2.)
            loss_span1 = self.span_criterion(span_prob1, span_mask)
            loss_span1 = loss_span1 * weights
            loss_span1 = torch.sum(loss_span1) / (torch.sum(video_mask) +
                                                  1e-12)
            loss_span2 = self.span_criterion(span_prob2, span_mask)
            loss_span2 = loss_span2 * weights
            loss_span2 = torch.sum(loss_span2) / (torch.sum(video_mask) +
                                                  1e-12)

        loss_query_self_simi = 0
        if not self.triplet:
            tmp_querys1 = F.normalize(tmp_querys1, dim=-1)
            tmp_querys2 = F.normalize(tmp_querys2, dim=-1)
            loss_query_self_simi = self.l2_criterion(
                tmp_querys1.reshape((-1, tmp_querys1.shape[-1])),
                tmp_querys2.reshape(-1, tmp_querys2.shape[-1]))
            loss_query_self_simi = torch.mean(loss_query_self_simi)

        loss_st_ed = 0
        if self.config.lw_st_ed != 0:

            # _loss_st, _loss_ed = self.share_norm_loss.moment_share_loss(st_prob, ed_prob, st_ed_indices, sample_mask)
            _loss_st, _loss_ed = self.share_norm_loss.loss(st_prob, ed_prob, st_ed_indices, sample_mask)
            loss_st = _loss_st * 0.01
            loss_ed = _loss_ed * 0.01
            # loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            # loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed
            # 新增细粒度
            # loss_st_fine_grain = self.temporal_criterion(fine_grain_st_prob, st_ed_indices[:, 0])
            # loss_ed_fine_grain = self.temporal_criterion(fine_grain_ed_prob, st_ed_indices[:, 1])
            # fine_grain_svmr_rate = 1e-2
            # loss_st_ed_fine_grain = fine_grain_svmr_rate * (
            #     loss_st_fine_grain + loss_ed_fine_grain)
        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(
                query_context_scores)

        loss_st_ed = self.config.lw_st_ed * loss_st_ed
        loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
        loss_neg_q = self.config.lw_neg_q * loss_neg_q

        # span都是0 下面两句其实没用
        loss_span1 = 0.01 * loss_span1
        loss_span2 = 0.01 * loss_span2
        #loss_biaffine = 0.1*loss_biaffine

        # 修改
        # loss_fcl = 0.05*loss_fcl
        loss_vcl = 0.05 * loss_vcl

        # loss_query_self_simi是0 下面这句没用
        loss_query_self_simi = 0.001 * loss_query_self_simi
        """
        loss_neg_ctx: shot-query -> VR *
        loss_neg_q: shot-query -> VR *
        shot_loss_vcl: shot-query -> VR NCE *
        loss_vcl: clip-query -> VR NCE
        loss_fcl: clip-query -> SVMR
        loss_scl: shot-query -> SVMR *
        loss_st_ed: clip-query -> SVMR
        """

        query_diverse_loss = 0
        consistency_loss = 0
        v2q_ce_loss = 0
        loss = loss_st_ed + loss_neg_ctx + loss_neg_q + loss_vid_kl + loss_sub_kl + loss_vid_kl_cross + loss_sub_kl_cross + loss_vcl + vs_loss + triplet_loss + query_diverse_loss + v2q_ce_loss # + loss_fcl
        # loss = loss_st_ed + loss_neg_ctx + loss_neg_q + loss_span1 + loss_span2 + loss_query_self_simi + loss_fcl + loss_vcl # + loss_biaffine
        # print(loss_neg_ctx, loss_neg_q)
        return loss, {
            "loss_st_ed": float(loss_st_ed),
            "loss_neg_ctx": float(loss_neg_ctx),
            "loss_neg_q": float(loss_neg_q),
            "loss_vid_kl": float(loss_vid_kl),
            "loss_sub_kl": float(loss_sub_kl),
            "loss_vid_kl_cross": float(loss_vid_kl_cross),
            "loss_sub_kl_cross": float(loss_sub_kl_cross),
            "loss_vcl": float(loss_vcl),
            "triplet_loss": float(triplet_loss),
            "consistency_loss": float(consistency_loss),
            "query_diverse_loss": float(query_diverse_loss),
            "v2q_ce_loss": float(v2q_ce_loss),
            # "consistency_loss": float(consistency_loss),
            "vs_loss": float(vs_loss),
            # "loss_fcl": float(loss_fcl),
            # "loss_st_ed_fine_grain": float(loss_st_ed_fine_grain),
            # "dqal_v_loss": float(dqal_v_loss),
            # "dqal_s_loss": float(dqal_s_loss),
            # "dqal_v_loss": float(dqal_v_loss),
            # "dqal_s_loss": float(dqal_s_loss),
            #   "loss_span1": float(loss_span1),
            #   "loss_span2": float(loss_span2),
            #   "loss_query_self_simi": float(loss_query_self_simi),
            #   "loss_biaffine": float(loss_biaffine),
            # "shot_loss_vcl": float(shot_loss_vcl),
            "loss_overall": float(loss),
        }

    def forward_back(self, query_feat, query_mask, video_feat, video_mask,
                     face_feat, face_mask, sub_feat, sub_mask, tef_feat,
                     tef_mask, st_ed_indices):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat: (N, Lv, Dv) or None
            video_mask: (N, Lv) or None
            face_feat:  (N, Lv, Dv) or None
            face_mask:  (N, Lv) or None
            sub_feat: (N, Lv, Ds) or None
            sub_mask: (N, Lv) or None
            tef_feat: (N, Lv, 2) or None,
            tef_mask: (N, Lv) or None,
            st_ed_indices: (N, 2), torch.LongTensor, 1st, 2nd columns are st, ed labels respectively.
        """
        video_feat1, video_feat2, sub_feat1, sub_feat2 = \
            self.encode_context(video_feat, video_mask, sub_feat, sub_mask)

        query_context_scores, st_prob, ed_prob = \
            self.get_pred_from_raw_query(query_feat, query_mask,
                                         video_feat1, video_feat2, video_mask,
                                         sub_feat1, sub_feat2, sub_mask, cross=False)

        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed

        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(
                query_context_scores)

        loss_st_ed = self.config.lw_st_ed * loss_st_ed
        loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
        loss_neg_q = self.config.lw_neg_q * loss_neg_q
        loss = loss_st_ed + loss_neg_ctx + loss_neg_q
        return loss, {
            "loss_st_ed": float(loss_st_ed),
            "loss_neg_ctx": float(loss_neg_ctx),
            "loss_neg_q": float(loss_neg_q),
            "loss_overall": float(loss)
        }

    def get_visualization_data(self, query_feat, query_mask, video_feat,
                               video_mask, sub_feat, sub_mask, tef_feat,
                               tef_mask, st_ed_indices):
        assert self.config.merge_two_stream and self.use_video and self.use_sub and not self.config.no_modular
        video_feat1, video_feat2, sub_feat1, sub_feat2 = \
            self.encode_context(video_feat, video_mask, sub_feat, sub_mask)
        encoded_query = self.encode_input(query_feat, query_mask,
                                          self.query_input_proj,
                                          self.query_encoder,
                                          self.query_pos_embed)  # (N, Lq, D)
        # (N, D), (N, D), (N, L, 2)
        video_query, sub_query, modular_att_scores = \
            self.get_modularized_queries(encoded_query, query_mask, self.modular_vector_mapping, return_modular_att=True)
        # (N, L), (N, L), (N, L)
        st_prob, ed_prob, similarity_scores, video_similarity, sub_similarity = self.get_merged_st_ed_prob(
            video_query,
            video_feat2,
            sub_query,
            sub_feat2,
            video_mask,
            cross=False,
            return_similaity=True)

        # clean up invalid bits
        data = dict(
            modular_att_scores=modular_att_scores.cpu().numpy(
            ),  # (N, Lq, 2), row 0, 1 are video, sub.
            st_prob=st_prob.cpu().numpy(),  # (N, L)
            ed_prob=ed_prob.cpu().numpy(),  # (N, L)
            similarity_scores=similarity_scores.cpu().numpy(),  # (N, L)
            video_similarity=video_similarity.cpu().numpy(),  # (N, L)
            sub_similarity=sub_similarity.cpu().numpy(),  # (N, L)
            st_ed_indices=st_ed_indices.cpu().numpy())  # (N, L)
        query_lengths = query_mask.sum(1).to(
            torch.long).cpu().tolist()  # (N, )
        ctx_lengths = video_mask.sum(1).to(torch.long).cpu().tolist()  # (N, )
        # print("query_lengths {}".format((type(query_lengths), len(query_lengths), query_lengths[:10])))
        for k, v in data.items():
            if k == "modular_att_scores":
                # print(k, v, v.shape, type(v))
                data[k] = [e[:l] for l, e in zip(query_lengths, v)
                           ]  # list(e) where e is  (Lq_i, 2)
            else:
                data[k] = [e[:l] for l, e in zip(ctx_lengths, v)
                           ]  # list(e) where e is (Lc_i)

        # aggregate info for each example
        datalist = []
        for idx in range(len(data["modular_att_scores"])):
            datalist.append({k: v[idx] for k, v in data.items()})
        return datalist  # list(dicts) of length N

    # def encode_query(self, query_feat, query_mask):
    #     #encoded_query = self.encode_input(query_feat, query_mask,
    #     #                                  self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
    #     #video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
    #     video_query, sub_query = self.get_modularized_queries(
    #         query_feat, query_mask, self.modular_vector_mapping)  # (N, D) * 2
    #     return video_query, sub_query
    def encode_query_word(self, word_feat, word_mask):
        #encoded_query = self.encode_input(query_feat, query_mask,
        #                                  self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        #video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        video_query, sub_query = self.get_modularized_queries(
            word_feat, word_mask,
            self.modular_vector_mapping_word)  # (N, D) * 2
        return video_query, sub_query

    # SVMR
    def encode_query_word2(self, word_feat, word_mask):
        #encoded_query = self.encode_input(query_feat, query_mask,
        #                                  self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        #video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        video_query, sub_query = self.get_modularized_queries(
            word_feat, word_mask,
            self.modular_vector_mapping_word2)  # (N, D) * 2
        return video_query, sub_query

    def encode_query_word3(self, word_feat, word_mask):
        #encoded_query = self.encode_input(query_feat, query_mask,
        #                                  self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        #video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        video_query, sub_query = self.get_modularized_queries(
            word_feat, word_mask,
            self.modular_vector_mapping_word3)  # (N, D) * 2
        return video_query, sub_query

    def encode_query_word_v_s(self, word_feat, word_mask):
        #encoded_query = self.encode_input(query_feat, query_mask,
        #                                  self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        #video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        video_sub_query = self.get_modularized_queries(
            word_feat, word_mask,
            self.modular_vector_mapping_word_v_s)  # (N, D) * 1
        return video_sub_query

    def encode_query_phrase(self, phrase_feat, phrase_mask):
        #encoded_query = self.encode_input(query_feat, query_mask,
        #                                  self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        #video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        video_query, sub_query = self.get_modularized_queries(
            phrase_feat, phrase_mask,
            self.modular_vector_mapping_phrase)  # (N, D) * 2
        return video_query, sub_query

    def encode_query_sentence(self, sentence_feat, sentence_mask):
        #encoded_query = self.encode_input(query_feat, query_mask,
        #                                  self.query_input_proj, self.query_encoder, self.query_pos_embed)  # (N, Lq, D)
        #video_query, sub_query = self.get_modularized_queries(encoded_query, query_mask)  # (N, D) * 2
        video_query, sub_query = self.get_modularized_queries(
            sentence_feat, sentence_mask,
            self.modular_vector_mapping_sentence)  # (N, D) * 2
        return video_query, sub_query

    def encode_query_fc(self, video_query, sub_query):
        video_query2 = self.query_vid2clip_linear(video_query)
        sub_query2 = self.query_vid2clip_linear2(sub_query)
        return video_query2, sub_query2

    # def encode_query2(self, query_feat, query_mask):
    #     video_query2, sub_query2 = self.get_modularized_queries(
    #         query_feat, query_mask, self.modular_vector_mapping2)
    #     #video_query2 = self.query_vid2clip_linear(video_query)
    #     #sub_query2 = self.query_vid2clip_linear2(sub_query)
    #     return video_query2, sub_query2

    def extend_query_with_face(self, query_feat, query_mask,
                               query_face_hidden):
        query_feat = torch.cat([query_feat, query_face_hidden], dim=-2)
        query_mask = torch.cat([
            query_mask,
            torch.ones((query_mask.shape[0], 1), dtype=query_mask.dtype).to(
                query_mask.device)
        ],
                               dim=-1)
        return query_feat, query_mask

    def non_cross_encode_context(self,
                                 context_feat,
                                 context_mask,
                                 module_name="video"):
        encoder_layer3 = getattr(self, module_name + "_encoder3") \
            if self.config.encoder_type == "transformer" else None
        return self._non_cross_encode_context(
            context_feat,
            context_mask,
            input_proj_layer=getattr(self, module_name + "_input_proj"),
            encoder_layer1=getattr(self, module_name + "_encoder1"),
            encoder_layer2=getattr(self, module_name + "_encoder2"),
            encoder_layer3=encoder_layer3)

    def _non_cross_encode_context(self,
                                  context_feat,
                                  context_mask,
                                  input_proj_layer,
                                  encoder_layer1,
                                  encoder_layer2,
                                  encoder_layer3=None):
        """
        Args:
            context_feat: (N, L, D)
            context_mask: (N, L)
            input_proj_layer:
            encoder_layer1:
            encoder_layer2:
            encoder_layer3
        """
        context_feat1 = self.encode_input(context_feat, context_mask,
                                          input_proj_layer, encoder_layer1,
                                          self.ctx_pos_embed)  # (N, L, D)
        if self.config.encoder_type in ["transformer", "cnn"]:
            context_mask = context_mask.unsqueeze(
                1)  # (N, 1, L), torch.FloatTensor
            context_feat2 = encoder_layer2(context_feat1,
                                           context_mask)  # (N, L, D)
            if self.config.encoder_type == "transformer":
                context_feat2 = encoder_layer3(context_feat2, context_mask)
        elif self.config.encoder_type in ["gru", "lstm"]:
            context_mask = context_mask.sum(
                1).long()  # (N, ), torch.LongTensor
            context_feat2 = encoder_layer2(context_feat1,
                                           context_mask)[0]  # (N, L, D)
        else:
            raise NotImplementedError
        return context_feat1, context_feat2

    def encode_context(self, video_feat, video_mask, sub_feat, sub_mask):
        if self.config.cross_att:
            ## 为了匹配video-face维度，修改
            assert self.use_video and self.use_sub
            return self.cross_encode_context(video_feat, video_mask, sub_feat,
                                             sub_mask)
        else:
            video_feat1, video_feat2 = (None, ) * 2
            if self.use_video:
                video_feat1, video_feat2 = self.non_cross_encode_context(
                    video_feat, video_mask, module_name="video")
            sub_feat1, sub_feat2 = (None, ) * 2
            if self.use_sub:
                sub_feat1, sub_feat2 = self.non_cross_encode_context(
                    sub_feat, sub_mask, module_name="sub")
            return video_feat1, video_feat2, sub_feat1, sub_feat2

    def cross_encode_context(
        self,
        video_feat,
        video_mask,
        sub_feat,
        sub_mask,
    ):
        _, ml = video_mask.size()
        video_feat, video_mask = self.videolocal(video_feat, video_mask,
                                                 self.V1ctx_token_pos_embed)
        sub_feat, sub_mask = self.subglobal(sub_feat, sub_mask,
                                            self.V2ctx_token_pos_embed)

        features = {
            "video_feat_1": video_feat[0][
                :,
                :ml,
            ],
            "video_feat_2": video_feat[1],
            "video_feat_3": video_feat[2],
            "video_feat_1_mask": video_mask[0][
                :,
                :ml,
            ],
            "video_feat_2_mask": video_mask[1],
            "video_feat_3_mask": video_mask[2],
            "sub_feat_1": sub_feat[0][
                :,
                :ml,
            ],
            "sub_feat_2": sub_feat[1],
            "sub_feat_3": sub_feat[2],
            "sub_feat_1_mask": sub_mask[0][
                :,
                :ml,
            ],
            "sub_feat_2_mask": sub_mask[1],
            "sub_feat_3_mask": sub_mask[2],
        }
        features["VR_video_feat"], features[
            "VR_video_feat_mask"] = self.HBI_video_pool(
                features["video_feat_1"], features["video_feat_1_mask"])
        features["VR_sub_feat"], features[
            "VR_sub_feat_mask"] = self.HBI_sub_pool(
                features["sub_feat_1"], features["sub_feat_1_mask"])
        v_s_feat = torch.cat(
            [features["VR_video_feat"], features["VR_sub_feat"]], dim=1)
        v_s_mask = torch.cat(
            [features["VR_video_feat_mask"], features["VR_sub_feat_mask"]],
            dim=1)
        features["VR_VLAD"] = self.t2vVLAD(v_s_feat, v_s_mask)
        features["VR_VLAD_mask"] = features["VR_VLAD"].new_ones(
            features["VR_VLAD"].size()[:-1])
        return features

    def query_embedding(self, query_feat, query_mask):
        query_feat1, self.query_input_proj = self.queryEncoder(
            query_feat, query_mask, self.query_token_pos_embed)
        return query_feat1, query_mask

    def back_cross_encode_context(self, video_feat, video_mask, sub_feat,
                                  sub_mask):
        encoded_video_feat = self.encode_input(video_feat, video_mask,
                                               self.video_input_proj,
                                               self.video_encoder3,
                                               self.ctx_pos_embed)
        encoded_sub_feat = self.encode_input(sub_feat, sub_mask,
                                             self.sub_input_proj,
                                             self.sub_encoder1,
                                             self.ctx_pos_embed)
        x_encoded_video_feat = self.cross_context_encoder(
            encoded_video_feat, video_mask, encoded_sub_feat, sub_mask,
            self.video_cross_att, self.video_cross_layernorm,
            self.video_encoder2)  # (N, L, D)
        x_encoded_sub_feat = self.cross_context_encoder(
            encoded_sub_feat, sub_mask, encoded_video_feat, video_mask,
            self.sub_cross_att, self.sub_cross_layernorm,
            self.sub_encoder2)  # (N, L, D)
        return encoded_video_feat, x_encoded_video_feat, encoded_sub_feat, x_encoded_sub_feat

    def cross_context_encoder(self, main_context_feat, main_context_mask,
                              side_context_feat, side_context_mask,
                              cross_att_layer, norm_layer, self_att_layer):
        """
        Args:
            main_context_feat: (N, Lq, D)
            main_context_mask: (N, Lq)
            side_context_feat: (N, Lk, D)
            side_context_mask: (N, Lk)
            cross_att_layer:
            norm_layer:
            self_att_layer:
        """
        cross_mask = torch.einsum("bm,bn->bmn", main_context_mask,
                                  side_context_mask)  # (N, Lq, Lk)
        cross_out = cross_att_layer(main_context_feat, side_context_feat,
                                    side_context_feat,
                                    cross_mask)  # (N, Lq, D)
        residual_out = norm_layer(cross_out + main_context_feat)
        if self.config.encoder_type in ["cnn", "transformer"]:
            return self_att_layer(residual_out, main_context_mask.unsqueeze(1))
        elif self.config.encoder_type in ["gru", "lstm"]:
            return self_att_layer(residual_out,
                                  main_context_mask.sum(1).long())[0]

    def encode_input(self, feat, mask, input_proj_layer, encoder_layer,
                     pos_embed_layer):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            # add_pe: bool, whether to add positional encoding
            pos_embed_layer
        """
        feat = input_proj_layer(feat)

        if self.config.encoder_type in ["cnn", "transformer"]:
            feat = pos_embed_layer(feat)
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
            return encoder_layer(feat, mask)  # (N, L, D_hidden)
        elif self.config.encoder_type in ["gru", "lstm"]:
            if self.config.add_pe_rnn:
                feat = pos_embed_layer(feat)
            mask = mask.sum(1).long()  # (N, ), torch.LongTensor
            return encoder_layer(feat, mask)[0]  # (N, L, D_hidden)

    def get_modularized_queries(self,
                                encoded_query,
                                query_mask,
                                modular_vector_mapping,
                                return_modular_att=False):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
            return_modular_att: bool
        """
        if self.config.no_modular:
            modular_query = torch.max(mask_logits(encoded_query,
                                                  query_mask.unsqueeze(2)),
                                      dim=1)[0]  # (N, D)
            return modular_query, modular_query  #
        else:
            modular_attention_scores = modular_vector_mapping(
                encoded_query)  # (N, L, 2 or 1)
            modular_attention_scores = F.softmax(mask_logits(
                modular_attention_scores, query_mask.unsqueeze(2)),
                                                 dim=1)
            # TODO check whether it is the same
            modular_queries = torch.einsum("blm,bld->bmd",
                                           modular_attention_scores,
                                           encoded_query)  # (N, 2 or 1, D)
            if return_modular_att:
                assert modular_queries.shape[1] == 2
                return modular_queries[:,
                                       0], modular_queries[:,
                                                           1], modular_attention_scores
            else:
                if modular_queries.shape[1] == 2:
                    return modular_queries[:,
                                           0], modular_queries[:,
                                                               1]  # (N, D) * 2
                else:  # 1
                    return modular_queries[:,
                                           0], modular_queries[:,
                                                               0]  # the same

    def get_modular_weights(self, encoded_query, query_mask):
        """
        Args:
            encoded_query: (N, L, D)
            query_mask: (N, L)
        """
        max_encoded_query, _ = torch.max(mask_logits(encoded_query,
                                                     query_mask.unsqueeze(2)),
                                         dim=1)  # (N, D)
        modular_weights = self.modular_weights_calculator(
            max_encoded_query)  # (N, 2)
        modular_weights = F.softmax(modular_weights, dim=-1)
        return modular_weights[:, 0:1], modular_weights[:, 1:2]  # (N, 1) * 2

    def multi_scale_proposal(self):
        return

    def get_video_level_scores(
        self,
        modularied_query,
        context_feat1,
        context_mask,
    ):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat1: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        #TODO clips as mutliscale candidate

        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat1 = F.normalize(context_feat1, dim=-1)
        query_context_scores = torch.einsum("md,nld->mln", modularied_query,
                                            context_feat1)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        #import pdb
        #pdb.set_trace()
        query_context_scores = mask_logits(query_context_scores,
                                           context_mask)  # (N, L, N)
        query_context_scores, _ = torch.max(
            query_context_scores,
            dim=1)  # (N, N) diagonal positions are positive pairs.
        return query_context_scores

    def generate_phrase_scores(self, modularied_query, context_feat1,
                               context_mask, query_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, S, D)
            context_feat1: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        #TODO clips as mutliscale candidate

        # modularied_query = F.normalize(modularied_query, dim=-1)
        # context_feat1 = F.normalize(context_feat1, dim=-1)
        query_context_scores = torch.einsum(
            "mwd,ncd->mnwc", modularied_query,
            context_feat1)  # (N_query, N_video, L_word, L_clips)
        query_mask_padded = query_mask.unsqueeze(2)  # (N_query, L_clipis, 1)
        query_mask_padded = query_mask_padded.unsqueeze(
            1)  # (N_query, 1, L_word, 1)
        context_mask_padded = context_mask.unsqueeze(
            1)  # (N_video, L_video, 1)
        context_mask_padded = context_mask_padded.unsqueeze(
            0)  # (1, N_video, L_video, 1)
        clip_attn_per_word = query_context_scores.masked_fill(
            query_mask_padded.bool(), 0)  # (N_query, N_video, L_word, L_clips)
        clip_attn_per_word = clip_attn_per_word.masked_fill(
            context_mask_padded.bool(),
            0)  # (N_query, N_video, L_word, L_clips)
        clip_attn_per_word[clip_attn_per_word < 0] = 0
        clip_attn_per_word = self.h2v_l2norm(clip_attn_per_word, dim=3)
        clip_attn_per_word = mask_logits(
            clip_attn_per_word,
            query_mask_padded)  # (N_query, N_video, L_word, L_clips)
        clip_attn_per_word = mask_logits(
            clip_attn_per_word,
            context_mask_padded)  # (N_query, N_video, L_word, L_clips)
        clip_attn_per_word = F.softmax(clip_attn_per_word, dim=3)
        # 不确定
        # clip_attn_per_word = mask_logits(clip_attn_per_word,
        #                                  query_mask_padded)  # 把 mask 的位置置为-1e14
        # clip_attn_per_word = mask_logits(clip_attn_per_word,
        #                                  context_mask_padded)  # 把 mask 的位置置为-1e14

        if True:
            # Infer clip_attn_per_word: [N_query, ALL_video, L_word, L_clip] context_feat1: [ALL_video, L_clips, D]
            if clip_attn_per_word.size()[0] != clip_attn_per_word.size()[1]:
                # context_feat1.unsqueeze(0).repeat(clip_attn_per_word.size()[0], 1, 1, 1): [N_query, N_video, L_clips, d]
                vid_attned_embeds = torch.einsum(
                    'bnwc,bncd->bnwd', clip_attn_per_word,
                    context_feat1.unsqueeze(0).repeat(
                        clip_attn_per_word.size()[0], 1, 1, 1))
                word_attn_sims = torch.einsum(
                    'bnwd,bnwd->bnw', self.h2v_l2norm(vid_attned_embeds),
                    self.h2v_l2norm(
                        modularied_query.unsqueeze(1).repeat(
                            1,
                            vid_attned_embeds.size()[1], 1, 1)))
                # sum: (N_query, N_video)
                phrase_scores = torch.sum(word_attn_sims * query_mask.float().unsqueeze(1), 2) \
                        / torch.sum(query_mask, 1).float().unsqueeze(1).clamp(min=1)
            # Train clip_attn_per_word: [N_query, N_video, L_word, L_clip] context_feat1: [N_video, L_clips, D]
            if clip_attn_per_word.size()[0] == clip_attn_per_word.size()[1]:
                vid_attned_embeds = torch.einsum('bnwc,bcd->bnwd',
                                                 clip_attn_per_word,
                                                 context_feat1)
                # word_attn_sims: [N_query, N_video, L_word]
                word_attn_sims = torch.einsum(
                    'bnwd,nwd->bnw', self.h2v_l2norm(vid_attned_embeds),
                    self.h2v_l2norm(modularied_query))
                # sum: (N_query, N_video)
                phrase_scores = torch.sum(word_attn_sims * query_mask.float().unsqueeze(1), 2) \
                        / torch.sum(query_mask, 1).float().unsqueeze(1).clamp(min=1)

        if False:
            # (batch_vids, batch_phrases, num_phrases)
            word_attn_sims = torch.sum(ground_sims * vid_attn_per_word, dim=2)
        return phrase_scores

    def cosine_sim(self, im, s):
        inner_prod = im.mm(s.t())
        im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
        s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
        sim = inner_prod / (im_norm * s_norm)
        return sim

    def my_get_video_level_scores(self, phrase_embeds, vid_embeds, vid_masks,
                                  phrase_masks):

        batch_vids, num_frames, _ = vid_embeds.size()
        vid_pad_masks = (vid_masks == 0).unsqueeze(1).unsqueeze(3)
        batch_phrases, num_phrases, dim_embed = phrase_embeds.size()

        # compute component-wise similarity
        vid_2d_embeds = vid_embeds.contiguous().view(-1, dim_embed)
        phrase_2d_embeds = phrase_embeds.contiguous().view(-1, dim_embed)

        # size = (batch_vids, batch_phrases, num_frames, num_phrases)
        ground_sims = self.cosine_sim(vid_2d_embeds, phrase_2d_embeds).view(
            batch_vids, num_frames, batch_phrases,
            num_phrases).transpose(1, 2)
        # #* 直接选择 max epoch1 表现不好 pass
        # max_values, _ = torch.max(ground_sims, dim=2)
        # max_values, _ = torch.max(max_values, dim=2)
        # return max_values.permute(1, 0)
        # size = (batch_vids, batch_phrases, num_frames, num_phrases)  (N_video, N_query, L_clips, L_word, )
        # ground_sims = torch.einsum(
        #     "vcd,qwdd->vqcw", vid_embeds, phrase_embeds)
        vid_attn_per_word = ground_sims.masked_fill(vid_pad_masks, 0)
        vid_attn_per_word[vid_attn_per_word < 0] = 0
        vid_attn_per_word = self.h2v_l2norm(vid_attn_per_word, dim=2)
        vid_attn_per_word = vid_attn_per_word.masked_fill(vid_pad_masks, -1e18)
        simattn_sigma = 2
        vid_attn_per_word = torch.softmax(simattn_sigma * vid_attn_per_word,
                                          dim=2)
        #* 一种对齐方式
        vid_attned_embeds = torch.einsum('abcd,ace->abde', vid_attn_per_word,
                                         vid_embeds)
        word_attn_sims = torch.einsum('abde,bde->abd',
                                      self.h2v_l2norm(vid_attned_embeds),
                                      self.h2v_l2norm(phrase_embeds))
        # if self.config.attn_fusion == 'embed':
        # vid_attned_embeds = torch.einsum('abcd,ace->abde',
        #                                  vid_attn_per_word, vid_embeds)
        # word_attn_sims = torch.einsum('abde,bde->abd',
        #                               self.h2v_l2norm(vid_attned_embeds),
        #                               self.h2v_l2norm(phrase_embeds))
        # elif self.config.attn_fusion == 'sim':
        # (batch_vids, batch_phrases, num_phrases)
        # word_attn_sims = torch.sum(ground_sims * vid_attn_per_word, dim=2)

        # sum: (batch_vid, batch_phrases)
        phrase_scores = torch.sum(word_attn_sims * phrase_masks.float().unsqueeze(0), 2) \
                   / torch.sum(phrase_masks, 1).float().unsqueeze(0).clamp(min=1)
        return phrase_scores.permute(1, 0)

    def h2v_l2norm(self, inputs, dim=-1):
        # inputs: (batch, dim_ft)
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        inputs = inputs / norm.clamp(min=1e-10)
        return inputs

    def get_unnormalized_video_level_scores(self, modularied_query,
                                            context_feat, context_mask):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        query_context_scores = torch.einsum("md,nld->mln", modularied_query,
                                            context_feat)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        query_context_scores = mask_logits(query_context_scores,
                                           context_mask)  # (N, L, N)
        return query_context_scores

    def my_get_unnormalized_video_level_scores(
        self,
        modularied_query,
        context_feat,
    ):
        """ Calculate video2query scores for each pair of video and query inside the batch.
        Args:
            modularied_query: (N, D)
            context_feat: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)  score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        query_context_scores = torch.einsum("md,nld->mln", modularied_query,
                                            context_feat)  # (N, L, N)
        # 因为是cluster之后计算的 sim，所有每一个 semantic_feat 都有效，不需要 mask
        # context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        # query_context_scores = mask_logits(query_context_scores,
        #                                    context_mask)  # (N, L, N)
        return query_context_scores

    def get_span_prediction(self, video_query2, video_feat2, sub_query2,
                            sub_feat2, context_mask):
        assert self.use_video and self.use_sub
        vq2 = video_query2.unsqueeze(1).repeat(1, video_feat2.shape[1], 1)
        sq2 = sub_query2.unsqueeze(1).repeat(1, sub_feat2.shape[1], 1)
        combined_vq = torch.cat((video_feat2, vq2), dim=-1)
        combined_sq = torch.cat((sub_feat2, sq2), dim=-1)
        #import pdb; pdb.set_trace()
        #v_span_prob = self.span_conv_vq(combined_vq)
        #s_span_prob = self.span_conv_sq(combined_sq)
        v_span_prob = self.span_conv_vq(combined_vq.transpose(1, 2)).transpose(
            1, 2)
        s_span_prob = self.span_conv_sq(combined_sq.transpose(1, 2)).transpose(
            1, 2)
        v_span_prob = torch.squeeze(v_span_prob, -1)
        s_span_prob = torch.squeeze(s_span_prob, -1)
        span_prob = (v_span_prob + s_span_prob) / 2.
        span_prob = mask_logits(span_prob, context_mask)
        return span_prob

    def get_biaffine_span_prediction(self,
                                     video_query2,
                                     video_feat2,
                                     sub_query2,
                                     sub_feat2,
                                     context_mask,
                                     val=False):
        #if val:
        #import pdb;pdb.set_trace()
        vq2 = video_query2.unsqueeze(1).repeat(1, video_feat2.shape[1], 1)
        sq2 = sub_query2.unsqueeze(1).repeat(1, sub_feat2.shape[1], 1)
        #print(val, video_feat2.shape, vq2.shape, sub_feat2.shape, sq2.shape)
        vq_ctx = torch.mul(video_feat2, vq2)
        sq_ctx = torch.mul(sub_feat2, sq2)
        sim = (vq_ctx + sq_ctx) / 2
        #st_ctx = self.conv_st_ctx(vq_ctx.transpose(1,2)).transpose(1,2)
        #ed_ctx = self.conv_ed_ctx(sq_ctx.transpose(1,2)).transpose(1,2)
        st_emb = self.start_layer(sim)
        ed_emb = self.end_layer(sim)
        out = self.biaffineLayer(st_emb, ed_emb)
        out = out.squeeze()
        #print('biaffine', out[0].sum(), out[0,:5,:5])
        return out

    def get_merged_st_ed_prob(self,
                              video_query,
                              video_feat,
                              sub_query,
                              sub_feat,
                              context_mask,
                              cross=False,
                              return_similaity=False):
        """context_mask could be either video_mask or sub_mask, since they are the same"""
        assert self.use_video and self.use_sub and self.config.span_predictor_type == "conv"
        video_query = self.video_query_linear(video_query)
        sub_query = self.sub_query_linear(sub_query)
        stack_conv = self.config.stack_conv_predictor_conv_kernel_sizes != -1
        num_convs = len(self.config.stack_conv_predictor_conv_kernel_sizes
                        ) if stack_conv else None
        if cross:
            video_similarity = torch.einsum("md,nld->mnl", video_query,
                                            video_feat)
            sub_similarity = torch.einsum("md,nld->mnl", sub_query, sub_feat)
            similarity = (video_similarity + sub_similarity
                          ) / 2  # (Nq, Nv, L)  from query to all videos.
            n_q, n_c, l = similarity.shape
            similarity = similarity.view(n_q * n_c, 1, l)
            if not stack_conv:
                st_prob = self.merged_st_predictor(similarity).view(
                    n_q, n_c, l)  # (Nq, Nv, L) 这里的形状是(1, 100, 100)
                ed_prob = self.merged_ed_predictor(similarity).view(
                    n_q, n_c, l)  # (Nq, Nv, L)
            else:
                st_prob_list = []
                ed_prob_list = []
                for idx in range(num_convs):
                    st_prob_list.append(self.merged_st_predictors[idx](
                        similarity).squeeze().unsqueeze(2))
                    ed_prob_list.append(self.merged_ed_predictors[idx](
                        similarity).squeeze().unsqueeze(2))
                # (Nq*Nv, L, 3) --> (Nq*Nv, L) -> (Nq, Nv, L)
                st_prob = self.combine_st_conv(torch.cat(st_prob_list,
                                                         dim=2)).view(
                                                             n_q, n_c, l)
                ed_prob = self.combine_ed_conv(torch.cat(ed_prob_list,
                                                         dim=2)).view(
                                                             n_q, n_c, l)
        else:
            video_similarity = torch.einsum("bd,bld->bl", video_query,
                                            video_feat)  # (N, L)
            sub_similarity = torch.einsum("bd,bld->bl", sub_query,
                                          sub_feat)  # (N, L)

            # 修改
            # video_similarity = video_similarity[:, ]
            similarity = (video_similarity + sub_similarity) / 2
            if not stack_conv:
                #import pdb
                #pdb.set_trace()
                st_prob = self.merged_st_predictor(
                    similarity.unsqueeze(1)).squeeze()  # (N, L)
                ed_prob = self.merged_ed_predictor(
                    similarity.unsqueeze(1)).squeeze()  # (N, L)
            else:
                st_prob_list = []
                ed_prob_list = []
                for idx in range(num_convs):
                    st_prob_list.append(self.merged_st_predictors[idx](
                        similarity.unsqueeze(1)).squeeze().unsqueeze(2))
                    ed_prob_list.append(self.merged_ed_predictors[idx](
                        similarity.unsqueeze(1)).squeeze().unsqueeze(2))
                st_prob = self.combine_st_conv(torch.cat(
                    st_prob_list, dim=2)).squeeze()  # (N, L, 3) --> (N, L)
                ed_prob = self.combine_ed_conv(torch.cat(
                    ed_prob_list, dim=2)).squeeze()  # (N, L, 3) --> (N, L)
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        if return_similaity:
            assert not cross
            return st_prob, ed_prob, similarity, video_similarity, sub_similarity
        else:
            return st_prob, ed_prob

    def get_st_ed_prob(self,
                       modularied_query,
                       context_feat2,
                       context_mask,
                       module_name="video",
                       cross=False):
        return self._get_st_ed_prob(
            modularied_query,
            context_feat2,
            context_mask,
            module_query_linear=getattr(self, module_name + "_query_linear"),
            st_predictor=getattr(self, module_name + "_st_predictor"),
            ed_predictor=getattr(self, module_name + "_ed_predictor"),
            cross=cross)

    def _get_st_ed_prob(self,
                        modularied_query,
                        context_feat2,
                        context_mask,
                        module_query_linear,
                        st_predictor,
                        ed_predictor,
                        cross=False):
        """
        Args:
            modularied_query: (N, D)
            context_feat2: (N, L, D), output of the first transformer encoder layer
            context_mask: (N, L)
            module_query_linear:
            st_predictor:
            ed_predictor:
            cross: at inference, calculate prob for each possible pairs of query and context.
        """
        query = module_query_linear(
            modularied_query)  # (N, D) no need to normalize here.
        if cross:
            if self.config.span_predictor_type == "conv":
                similarity = torch.einsum(
                    "md,nld->mnl", query,
                    context_feat2)  # (Nq, Nv, L)  from query to all videos.
                n_q, n_c, l = similarity.shape
                similarity = similarity.view(n_q * n_c, 1, l)
                st_prob = st_predictor(similarity).view(n_q, n_c,
                                                        l)  # (Nq, Nv, L)
                ed_prob = ed_predictor(similarity).view(n_q, n_c,
                                                        l)  # (Nq, Nv, L)
            elif self.config.span_predictor_type == "cat_linear":
                st_prob_q = st_predictor[0](query).unsqueeze(1)  # (Nq, 1, 1)
                st_prob_ctx = st_predictor[1](
                    context_feat2).squeeze().unsqueeze(0)  # (1, Nv, L)
                st_prob = st_prob_q + st_prob_ctx  # (Nq, Nv, L)
                ed_prob_q = ed_predictor[0](query).unsqueeze(1)  # (Nq, 1, 1)
                ed_prob_ctx = ed_predictor[1](
                    context_feat2).squeeze().unsqueeze(0)  # (1, Nv, L)
                ed_prob = ed_prob_q + ed_prob_ctx  # (Nq, Nv, L)
            context_mask = context_mask.unsqueeze(0)  # (1, Nv, L)
        else:
            if self.config.span_predictor_type == "conv":
                similarity = torch.einsum("bd,bld->bl", query,
                                          context_feat2)  # (N, L)
                st_prob = st_predictor(
                    similarity.unsqueeze(1)).squeeze()  # (N, L)
                ed_prob = ed_predictor(
                    similarity.unsqueeze(1)).squeeze()  # (N, L)
            elif self.config.span_predictor_type == "cat_linear":
                # avoid concatenation by break into smaller matrix multiplications.
                st_prob = st_predictor[0](query) + st_predictor[1](
                    context_feat2).squeeze()  # (N, L)
                ed_prob = ed_predictor[0](query) + ed_predictor[1](
                    context_feat2).squeeze()  # (N, L)
        st_prob = mask_logits(st_prob, context_mask)  # (N, L)
        ed_prob = mask_logits(ed_prob, context_mask)
        return st_prob, ed_prob

    # 修改
    def get_pred_from_triplet(self,
                              word_feat,
                              word_mask,
                              features,
                              query_face_hidden,
                              span_mask=None,
                              cross=False,
                              is_eval=False,
                              max_triplet_videos=10,
                              gt_idx=None):
        # video_query_word, sub_query_word = self.encode_query_word(word_feat, word_mask)
        # video_query_phrase, sub_query_phrase = self.encode_query_phrase(phrase_feat, phrase_mask)
        # video_query_sentence, sub_query_sentence = self.encode_query_sentence(sentence_feat, sentence_mask)
        divisor = self.use_sub + self.use_video
        video_feat_1 = features["video_feat_1"]
        video_feat_2 = features["video_feat_2"]
        video_feat_3 = features["video_feat_3"]
        video_feat_1_mask = features["video_feat_1_mask"]
        video_feat_2_mask = features["video_feat_2_mask"]
        video_feat_3_mask = features["video_feat_3_mask"]
        sub_feat_1 = features["sub_feat_1"]
        sub_feat_2 = features["sub_feat_2"]
        sub_feat_3 = features["sub_feat_3"]
        sub_feat_1_mask = features["sub_feat_1_mask"]
        sub_feat_2_mask = features["sub_feat_2_mask"]
        sub_feat_3_mask = features["sub_feat_3_mask"]
        VR_video_feat = features["VR_video_feat"]
        VR_video_feat_mask = features["VR_video_feat_mask"]
        VR_sub_feat = features["VR_sub_feat"]
        VR_sub_feat_mask = features["VR_sub_feat_mask"]
        VR_video_sub_feat = features["VR_VLAD"]
        VR_video_sub_feat_mask = features["VR_VLAD_mask"]

        video_query_word, sub_query_word = self.encode_query_word(
            word_feat, word_mask)
        video_query_word2, sub_query_word2 = self.encode_query_word2(
            word_feat, word_mask)  # SVMR
        video_query_word3, sub_query_word3 = self.encode_query_word3(
            word_feat, word_mask)  # loss_vcl
        video_sub_query = self.encode_query_word_v_s(word_feat, word_mask)
        if gt_idx == None:
            # get video-level retrieval scores
            video_q2ctx_scores_word = self.get_video_level_scores(
                video_query_word, VR_video_feat,
                VR_video_feat_mask) if self.use_video else 0
            sub_q2ctx_scores_word = self.get_video_level_scores(
                sub_query_word, VR_sub_feat,
                VR_sub_feat_mask) if self.use_sub else 0
            # 修改
            # video_sub_q2ctx_scores_word = self.get_video_level_scores(
            #     video_sub_query[0], VR_video_sub_feat,
            #     VR_video_sub_feat_mask) if self.use_sub else 0
            # se_v_feats: [bsz, N, qdim] se_v_attw: [bsz, N, L_word]
            fine_grain_video_scores, _, _ = self.cal_video_score(
                word_feat, video_feat_1, word_mask, video_feat_1_mask)
            fine_grain_sub_scores, _, _ = self.cal_sub_score(
                word_feat, sub_feat_1, word_mask, sub_feat_1_mask)

            loss_vid_kl_cross = 0
            # loss_vid_kl_cross = self.kl(fine_grain_video_scores,
            #                             video_sub_q2ctx_scores_word)
            loss_sub_kl_cross = 0
            # loss_sub_kl_cross = self.kl(fine_grain_sub_scores,
            #                             video_sub_q2ctx_scores_word)
            loss_vid_kl = self.kl(video_q2ctx_scores_word,
                                  fine_grain_video_scores)
            loss_sub_kl = self.kl(sub_q2ctx_scores_word,
                                  fine_grain_sub_scores)
            # 自定义系数
            # loss_vid_kl *= 0
            # loss_sub_kl *= 0
            loss_vid_kl *= 1e3
            loss_sub_kl *= 1e3
            loss_vid_kl_cross *= 1e3
            loss_sub_kl_cross *= 1e3
            if not is_eval:
                v_query_diverse_loss, v2q_v_ce_loss = self.share_norm_loss.query_diverse_loss(video_q2ctx_scores_word, video_query_word)
                s_query_diverse_loss, v2q_s_ce_loss = self.share_norm_loss.query_diverse_loss(sub_q2ctx_scores_word, sub_query_word)
                v_s_query_diverse_loss = 0
                v2q_vs_ce_loss = 0
                # v_s_query_diverse_loss, v2q_vs_ce_loss = self.share_norm_loss.query_diverse_loss(video_sub_q2ctx_scores_word, video_sub_query[0])
                query_diverse_loss = v_query_diverse_loss + s_query_diverse_loss + v_s_query_diverse_loss
                v2q_ce_loss = (v2q_v_ce_loss + v2q_s_ce_loss + v2q_vs_ce_loss) / 3
            metric_weight = F.softmax(self.metric_weight, dim=0)
            # q2ctx_scores = \
            # metric_weight[0] * (video_q2ctx_scores_word + sub_q2ctx_scores_word + video_sub_q2ctx_scores_word) + metric_weight[1] * (fine_grain_video_scores) + metric_weight[2] * (fine_grain_sub_scores)  # (N, N) # (100_bsz, 2179) in inference stage
            # q2ctx_scores_eval = (video_q2ctx_scores_word +
            #                      sub_q2ctx_scores_word +
            #                      video_sub_q2ctx_scores_word) / (divisor + 1)
            q2ctx_scores = \
            metric_weight[0] * (video_q2ctx_scores_word + sub_q2ctx_scores_word + 0) + metric_weight[1] * (fine_grain_video_scores) + metric_weight[2] * (fine_grain_sub_scores)  # (N, N) # (100_bsz, 2179) in inference stage
            q2ctx_scores_eval = (video_q2ctx_scores_word +
                                 sub_q2ctx_scores_word +
                                 0) / (divisor + 1)

        else:
            q2ctx_scores = 0.
        if is_eval:
            if gt_idx == None:
                # infer的时候只用底层
                # _sorted_q2c_scores, _sorted_q2c_indices = \
                #         torch.topk(q2ctx_scores, max_triplet_videos, dim=1, largest=True) #(100_bsz, max_triplet_videos)
                _sorted_q2c_scores, _sorted_q2c_indices = \
                        torch.topk(q2ctx_scores_eval, max_triplet_videos, dim=1, largest=True) #(100_bsz, max_triplet_videos)
            else:
                _sorted_q2c_indices = gt_idx
                max_triplet_videos = 1
                q2ctx_scores_eval = 0
            #for one_query_feat in query_feat:
            tmp_st_prob = []
            tmp_ed_prob = []
            tmp_out_2d = []
            tmp_querys = []
            tmp_querys1 = []
            tmp_querys2 = []
            out_2d = []
            query_iter = 0
            for one_query_feat, one_query_mask, topk_q2c_indice, one_video_query2, one_sub_query2 in zip(
                    word_feat, word_mask, _sorted_q2c_indices,
                    video_query_word2, sub_query_word2):
                topk_video_feat = video_feat_1[topk_q2c_indice].repeat(1, 1, 1)
                topk_video_mask = video_feat_1_mask[topk_q2c_indice].repeat(
                    1, 1)
                topk_sub_feat = sub_feat_1[topk_q2c_indice].repeat(1, 1, 1)
                topk_sub_mask = sub_feat_1_mask[topk_q2c_indice].repeat(1, 1)
                topk_query_feat = one_query_feat.repeat(
                    max_triplet_videos, 1, 1)
                topk_query_mask = one_query_mask.repeat(max_triplet_videos, 1)
                new_one_video_query2 = one_video_query2.repeat(1, 1)
                new_one_sub_query2 = one_sub_query2.repeat(1, 1)
                #import pdb;pdb.set_trace()

                if self.triplet:
                    # 用于二阶段的特征
                    conquer_feat, one_st_prob, one_ed_prob = self.bidirect(topk_query_feat, topk_video_feat, topk_sub_feat, topk_video_mask, topk_query_mask)
                    one_st_prob = one_st_prob.unsqueeze(0)
                    one_ed_prob = one_ed_prob.unsqueeze(0)
                    tmp_video_feat2 = conquer_feat
                    tmp_sub_feat2 = conquer_feat
                    # tmp_video_feat2 = self.conquer_cross_visual(conquer_feat, topk_video_mask, topk_video_feat, topk_video_mask)
                    # tmp_sub_feat2 = self.conquer_cross_visual(conquer_feat, topk_video_mask, topk_sub_feat, topk_sub_mask)
                    stage2_score = self.vs_score(conquer_feat)
                    # 如果是SVMR就不需要2stage影响1stage
                    if gt_idx == None:
                        # 获得最大值 stage_max_score: [qbsz, rank_k]
                        stage_max_score = stage2_score.max(dim=0)[0]
                        # 高斯衰减权重
                        sigma = 25
                        # q2ctx_scores_eval[query_iter, topk_q2c_indice] = q2ctx_scores_eval[query_iter, topk_q2c_indice]
                        q2ctx_scores_eval[query_iter, topk_q2c_indice] = q2ctx_scores_eval[query_iter, topk_q2c_indice] * (1 + torch.exp(-(stage2_score - stage_max_score) ** 2 / sigma))
                        # q2ctx_scores_eval[query_iter, topk_q2c_indice] = q2ctx_scores_eval[query_iter, topk_q2c_indice] * stage2_score
                    query_iter = query_iter + 1

                    tmp_query = topk_query_feat
                    tmp_querys.append(tmp_query)
                else:
                    (tmp_video_feat2,
                     tmp_query1), _, _ = self.video_query_Encoder(
                         topk_video_feat, topk_video_mask, topk_query_feat,
                         topk_query_mask, self.ctx_token_pos_embed)

                    (tmp_sub_feat2, tmp_query2), _, _ = self.sub_query_Encoder(
                        topk_sub_feat, topk_sub_mask, topk_query_feat,
                        topk_query_mask, self.ctx_token_pos_embed)
                    tmp_querys1.append(tmp_query1)
                    tmp_querys2.append(tmp_query2)


                if self.config.merge_two_stream and self.use_video and self.use_sub:

                    if self.is_biaffine:
                        one_out_2d = self.get_biaffine_span_prediction(
                            new_one_video_query2,
                            tmp_video_feat2,
                            new_one_sub_query2,
                            tmp_sub_feat2,
                            shot_feat1_mask,
                            val=True)
                        one_out_2d = one_out_2d.unsqueeze(0)
                        tmp_out_2d.append(one_out_2d)
                    tmp_st_prob.append(one_st_prob)
                    tmp_ed_prob.append(one_ed_prob)
            st_prob = torch.cat(tmp_st_prob, dim=0)
            ed_prob = torch.cat(tmp_ed_prob, dim=0)
            if self.is_biaffine:
                out_2d = torch.cat(tmp_out_2d, dim=0)
            if not self.triplet:
                tmp_querys1 = torch.cat(tmp_querys1, dim=0)
                tmp_querys2 = torch.cat(tmp_querys2, dim=0)
            else:
                tmp_querys1 = 0
                tmp_querys2 = 0
            return q2ctx_scores_eval, st_prob, ed_prob, tmp_querys1, tmp_querys2, out_2d

        else:
            query1 = 0.
            query2 = 0.
            if self.triplet:
                # 用于二阶段的特征
                # stage1_score: [qbsz, sample + 1]
                sample_video_feat_1, sample_sub_feat_1, sample_video_feat_1_mask, stage1_score = self.share_norm_loss.sample_hard(q2ctx_scores, video_feat_1, sub_feat_1, video_feat_1_mask)
                sample_num = sample_video_feat_1.shape[1]
                # sample_sub_feat_1, sample_sub_feat_1_mask, stage1_score = self.share_norm_loss.sample_hard(q2ctx_scores, sub_feat_1, sub_feat_1_mask)
                sample = sample_video_feat_1.shape[1]
                _sample_video_feat_1 = []
                _sample_sub_feat_1 = []
                _st_prob = []
                _ed_prob = []
                # _conquery_feat = []
                _conquery_mask = sample_video_feat_1_mask
                _stage2_score = []
                for i in range(sample):
                    tmp_feat, tmp_st_prob, tmp_ed_prob = self.bidirect(word_feat, sample_video_feat_1[:, i], sample_sub_feat_1[:, i], sample_video_feat_1_mask[:, i], word_mask)
                    _st_prob.append(tmp_st_prob)
                    _ed_prob.append(tmp_ed_prob)
                    # _conquery_feat.append(tmp_feat)
                    video_score = self.vs_score(tmp_feat)
                    _stage2_score.append(video_score)
                    if i == 0:
                        # 正样本gt_conquery_feat: [qbsz, lv, d]gt_conquery_feat = tmp_feat
                        gt_conquery_feat = tmp_feat
                    # tmp_video_feat_1 = self.conquer_cross_visual(gt_conquery_feat, sample_video_feat_1_mask[:, i], sample_video_feat_1[:, i], sample_video_feat_1_mask[:, i])
                    # tmp_sub_feat_1 = self.conquer_cross_sub(gt_conquery_feat, sample_video_feat_1_mask[:, i], sample_sub_feat_1[:, i], sample_video_feat_1_mask[:, i])
                    tmp_video_feat_1 = tmp_feat
                    tmp_sub_feat_1 = tmp_feat
                    _sample_video_feat_1.append(tmp_video_feat_1)
                    _sample_sub_feat_1.append(tmp_sub_feat_1)
                    # _stage2_score_sub.append(self.vs_score_sub(tmp_sub_feat_1))
                qbsz = word_feat.shape[0]
                st_prob = torch.stack(_st_prob)
                ed_prob = torch.stack(_ed_prob)
                st_prob = st_prob.permute(1, 0, 2).contiguous().view(qbsz,-1)
                ed_prob = ed_prob.permute(1, 0, 2).contiguous().view(qbsz,-1)

                sample_video_feat_1 = torch.stack(_sample_video_feat_1)
                sample_sub_feat_1 = torch.stack(_sample_sub_feat_1)
                sample_video_feat_1 = sample_video_feat_1.permute(1, 0, 2, 3)
                sample_sub_feat_1 = sample_sub_feat_1.permute(1, 0, 2, 3)
                # stage2_score_visual: [qbsz, sample + 1]
                stage2_score = torch.stack(_stage2_score)
                stage2_score = stage2_score.permute(1, 0)
                # consistency_loss
                consistency_loss = self.consistency_loss(stage1_score, stage2_score)
                # vs_loss
                vs_loss, triplet_loss = self.vs_score.video_score_loss(stage2_score, measure="cross_entropy")

                # 修改: 这部分为了代码完整性，暂时不变
                video_feat2 = video_feat_1
                sub_feat2 = sub_feat_1
                # hard Frame-CL
                # loss_fcl_vq = hard_video_query_loss(sample_video_feat2, video_query_word2, span_mask, sample_video_feat_1_mask, measure='JSD')
                # loss_fcl_sq = hard_video_query_loss(sample_sub_feat2, sub_query_word2, span_mask, sample_sub_feat_1_mask, measure='JSD')
                # loss_fcl = ((loss_fcl_vq + loss_fcl_sq) / 2) * 0.01
            if self.config.merge_two_stream and self.use_video and self.use_sub:
                fine_grain_st_prob, fine_grain_ed_prob = self.fine_grain_svmr_train(word_feat, video_feat_1, sub_feat_1, word_mask, video_feat_1_mask, sub_feat_1_mask)

                if self.is_biaffine:
                    out_2d = self.get_biaffine_span_prediction(
                        video_query_word2,
                        video_feat2,
                        sub_query_word2,
                        video_feat_1_mask,
                        video_feat_1_mask,
                        val=False)
                else:
                    out_2d = None
            #import pdb;pdb.set_trace()
            if not cross:
                span_prob = self.get_span_prediction(video_query_word2,
                                                     video_feat2,
                                                     sub_query_word2,
                                                     sub_feat2,
                                                     video_feat_1_mask)
                # 问题！！
                span_prob2 = self.get_span_prediction(video_query_word2,
                                                      video_feat2,
                                                      sub_query_word2,
                                                      sub_feat2,
                                                      video_feat_1_mask)
            return q2ctx_scores, st_prob, ed_prob, query1, query2, span_prob, span_prob2, video_query_word3, sub_query_word3, video_sub_query[
                1], out_2d, loss_vid_kl, loss_sub_kl, loss_vid_kl_cross, loss_sub_kl_cross, sample_video_feat_1_mask, vs_loss, triplet_loss, consistency_loss, query_diverse_loss, v2q_ce_loss  # un-normalized masked probabilities!!!!!

    def get_pred_from_raw_query(self,
                                query_feat,
                                query_mask,
                                video_feat1,
                                video_feat2,
                                video_mask,
                                sub_feat1,
                                sub_feat2,
                                sub_mask,
                                cross=False):
        """
        Args:
            query_feat: (N, Lq, Dq)
            query_mask: (N, Lq)
            video_feat1: (N, Lv, D) or None
            video_feat2:
            video_mask: (N, Lv)
            sub_feat1: (N, Lv, D) or None
            sub_feat2:
            sub_mask: (N, Lv)
            cross:
        """
        video_query, sub_query = self.encode_query(query_feat, query_mask)
        divisor = self.use_sub + self.use_video

        # get video-level retrieval scores
        video_q2ctx_scores = self.get_video_level_scores(
            video_query, video_feat1, video_mask) if self.use_video else 0
        sub_q2ctx_scores = self.get_video_level_scores(
            sub_query, sub_feat1, sub_mask) if self.use_sub else 0
        q2ctx_scores = (video_q2ctx_scores +
                        sub_q2ctx_scores) / divisor  # (N, N)

        if self.config.merge_two_stream and self.use_video and self.use_sub:
            st_prob, ed_prob = self.get_merged_st_ed_prob(video_query,
                                                          video_feat2,
                                                          sub_query,
                                                          sub_feat2,
                                                          video_mask,
                                                          cross=cross)
        else:
            video_st_prob, video_ed_prob = self.get_st_ed_prob(
                video_query,
                video_feat2,
                video_mask,
                module_name="video",
                cross=cross) if self.use_video else (0, 0)
            sub_st_prob, sub_ed_prob = self.get_st_ed_prob(
                sub_query, sub_feat2, sub_mask, module_name="sub",
                cross=cross) if self.use_sub else (0, 0)
            st_prob = (video_st_prob + sub_st_prob) / divisor  # (N, Lv)
            ed_prob = (video_ed_prob + sub_ed_prob) / divisor  # (N, Lv)
        return q2ctx_scores, st_prob, ed_prob  # un-normalized masked probabilities!!!!!

    def get_video_level_loss(self, query_context_scores):
        """ ranking loss between (pos. query + pos. video) and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (N, N), cosine similarity [-1, 1],
                Each row contains the scores between the query to each of the videos inside the batch.
        """
        bsz = len(query_context_scores)
        diagonal_indices = torch.arange(bsz).to(query_context_scores.device)
        # 取对角线元素等价torch.einsum("ii->i", query_context_scores)
        pos_scores = query_context_scores[diagonal_indices,
                                          diagonal_indices]  # (N, )
        query_context_scores_masked = copy.deepcopy(query_context_scores.data)
        # impossibly large for cosine similarity, the copy is created as modifying the original will cause error
        query_context_scores_masked[diagonal_indices, diagonal_indices] = 999
        pos_query_neg_context_scores = self.get_neg_scores(
            query_context_scores, query_context_scores_masked)
        neg_query_pos_context_scores = self.get_neg_scores(
            query_context_scores.transpose(0, 1),
            query_context_scores_masked.transpose(0, 1))
        loss_neg_ctx = self.get_ranking_loss(pos_scores,
                                             pos_query_neg_context_scores)
        loss_neg_q = self.get_ranking_loss(pos_scores,
                                           neg_query_pos_context_scores)
        return loss_neg_ctx, loss_neg_q

    def get_neg_scores(self, scores, scores_masked):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos. Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores, except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz = len(scores)
        batch_indices = torch.arange(bsz).to(scores.device)
        _, sorted_scores_indices = torch.sort(scores_masked,
                                              descending=True,
                                              dim=1)
        sample_min_idx = 1  # skip the masked positive
        sample_max_idx = min(sample_min_idx + self.config.hard_pool_size, bsz) \
            if self.config.use_hard_negative else bsz
        sampled_neg_score_indices = sorted_scores_indices[
            batch_indices,
            torch.randint(sample_min_idx, sample_max_idx,
                          size=(bsz, )).to(scores.device)]  # (N, )
        sampled_neg_scores = scores[batch_indices,
                                    sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.config.ranking_loss_type == "hinge":  # max(0, m + S_neg - S_pos)
            return torch.clamp(self.config.margin + neg_score - pos_score,
                               min=0).sum() / len(pos_score)
        elif self.config.ranking_loss_type == "lse":  # log[1 + exp(S_neg - S_pos)]
            return torch.log1p(
                torch.exp(neg_score - pos_score)).sum() / len(pos_score)
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")


def mask_logits(target, mask):
    #import pdb;pdb.set_trace()
    return target * mask + (1 - mask) * (-1e10)


"""
class VideoQueryEncoder(nn.Module):
    def __init__(self, config, video_modality, visual_dim=4352, text_dim= 768, query_dim=768, hidden_dim = 768,split_num=100,):
        super(VideoQueryEncoder, self).__init__()
    def forward(self, query_feat, query_mask, video_feat, video_mask, face_feat, face_mask, sub_feat, sub_mask, tef_feat, tef_mask, st_ed_indices):
        video_feat1, sub_feat1 = self.videoEncoder(video_feat, video_position_ids, video_token_type_ids, video_mask,
                          sub_feat, sub_position_ids, sub_token_type_ids, sub_mask)

        query_feat1 = self.queryEncoder(query_feat, query_position_ids, query_token_type_ids, query_mask)

        query_context_scores, st_prob, ed_prob = self.get_pred_from_raw_query(query_feat, query_mask, video_feat1, video_feat2, video_mask, sub_feat1, sub_feat2, sub_mask, cross=False)
        loss_st_ed = 0
        if self.config.lw_st_ed != 0:
            loss_st = self.temporal_criterion(st_prob, st_ed_indices[:, 0])
            loss_ed = self.temporal_criterion(ed_prob, st_ed_indices[:, 1])
            loss_st_ed = loss_st + loss_ed

        loss_neg_ctx, loss_neg_q = 0, 0
        if self.config.lw_neg_ctx != 0 or self.config.lw_neg_q != 0:
            loss_neg_ctx, loss_neg_q = self.get_video_level_loss(query_context_scores)

        loss_st_ed = self.config.lw_st_ed * loss_st_ed
        loss_neg_ctx = self.config.lw_neg_ctx * loss_neg_ctx
        loss_neg_q = self.config.lw_neg_q * loss_neg_q
        loss = loss_st_ed + loss_neg_ctx + loss_neg_q
        return loss, {"loss_st_ed": float(loss_st_ed),
                      "loss_neg_ctx": float(loss_neg_ctx),
                      "loss_neg_q": float(loss_neg_q),
                      "loss_overall": float(loss)}
"""


class TransformerBaseModel(nn.Module):
    """
    Base Transformer model
    """

    def __init__(self, config, with_emb=True):
        super(TransformerBaseModel, self).__init__()
        self.with_emb = with_emb
        if self.with_emb:
            self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)

    def forward(self,
                features,
                position_ids,
                token_type_ids,
                attention_mask,
                my_mask=None):
        # embedding layer
        # 修改
        if position_ids == None and token_type_ids == None:
            self.with_emb = False
        if self.with_emb:
            embedding_output = self.embeddings(token_type_ids=token_type_ids,
                                               inputs_embeds=features,
                                               position_ids=position_ids)
        else:
            embedding_output = features

        encoder_outputs = self.encoder(embedding_output,
                                       my_mask=my_mask,
                                       attention_mask=attention_mask)

        sequence_output = encoder_outputs[0]

        return sequence_output


class ThreeModalEncoder(nn.Module):
    """
        Three modality Transformer Encoder model
    """

    def __init__(self,
                 config,
                 img_dim,
                 text_dim,
                 query_dim,
                 hidden_dim,
                 split_num,
                 output_split=True):
        super(ThreeModalEncoder, self).__init__()
        self.img_linear = LinearLayer(in_hsz=img_dim, out_hsz=hidden_dim)
        self.text_linear = LinearLayer(in_hsz=text_dim, out_hsz=hidden_dim)
        self.query_linear = LinearLayer(in_hsz=query_dim, out_hsz=hidden_dim)

        self.transformer = TransformerBaseModel(config, with_emb=False)
        self.output_split = output_split
        if self.output_split:
            self.split_num = split_num

    #def forward(self, visual_features, visual_position_ids, visual_token_type_ids, visual_attention_mask,
    #            text_features,text_position_ids,text_token_type_ids,text_attention_mask, ctx_token_pos_embed):
    def forward(self,
                visual_features,
                visual_attention_mask,
                text_features,
                text_attention_mask,
                query_features,
                query_attention_mask,
                ctx_token_pos_embed,
                query_token_pos_embed,
                is_eval=False):

        transformed_im = self.img_linear(visual_features)
        transformed_text = self.text_linear(text_features)
        transformed_query = self.query_linear(query_features)
        #transformed_im = visual_features
        #transformed_text = text_features
        #transformed_query = query_features

        visual_token_type_ids, visual_position_ids = ctx_token_pos_embed(
            transformed_im, 0)
        text_token_type_ids, text_position_ids = ctx_token_pos_embed(
            transformed_text, 1)
        query_token_type_ids, query_position_ids = query_token_pos_embed(
            transformed_query, 1)
        #import pdb; pdb.set_trace()
        transformer_input_feat = torch.cat(
            (transformed_im, transformed_text, transformed_query), dim=1)
        transformer_input_feat_pos_id = torch.cat(
            (visual_position_ids, text_position_ids, query_position_ids),
            dim=1)
        transformer_input_feat_token_id = torch.cat(
            (visual_token_type_ids, text_token_type_ids, query_token_type_ids),
            dim=1)
        transformer_input_feat_mask = torch.cat(
            (visual_attention_mask, text_attention_mask, query_attention_mask),
            dim=1)

        output = self.transformer(
            features=transformer_input_feat,
            position_ids=transformer_input_feat_pos_id,
            token_type_ids=transformer_input_feat_token_id,
            attention_mask=transformer_input_feat_mask)
        #import pdb
        #pdb.set_trace()
        if self.output_split:
            #return torch.split(output,self.split_num,dim=1)
            return torch.split(output, [
                transformed_im.shape[1], transformed_text.shape[1],
                transformed_query.shape[1]
            ],
                               dim=1), transformed_im, transformed_text
        else:
            return output, transformed_im, transformed_text


class TwoModalEncoder(nn.Module):
    """
        Two modality Transformer Encoder model
    """

    def __init__(self,
                 config,
                 img_dim,
                 text_dim,
                 hidden_dim,
                 split_num,
                 output_split=True,
                 with_emb=True):
        super(TwoModalEncoder, self).__init__()
        self.img_linear = LinearLayer(in_hsz=img_dim, out_hsz=hidden_dim)
        self.text_linear = LinearLayer(in_hsz=text_dim, out_hsz=hidden_dim)

        self.transformer = TransformerBaseModel(config, with_emb=with_emb)
        self.output_split = output_split
        if self.output_split:
            self.split_num = split_num

    #def forward(self, visual_features, visual_position_ids, visual_token_type_ids, visual_attention_mask,
    #            text_features,text_position_ids,text_token_type_ids,text_attention_mask, ctx_token_pos_embed):
    def forward(self, visual_features, visual_attention_mask, text_features,
                text_attention_mask, ctx_token_pos_embed):
        #import pdb; pdb.set_trace()
        transformed_im = self.img_linear(visual_features)
        transformed_text = self.text_linear(text_features)

        visual_token_type_ids, visual_position_ids = ctx_token_pos_embed(
            transformed_im, 0)
        text_token_type_ids, text_position_ids = ctx_token_pos_embed(
            transformed_text, 1)
        transformer_input_feat = torch.cat((transformed_im, transformed_text),
                                           dim=1)
        transformer_input_feat_pos_id = torch.cat(
            (visual_position_ids, text_position_ids), dim=1)
        transformer_input_feat_token_id = torch.cat(
            (visual_token_type_ids, text_token_type_ids), dim=1)
        transformer_input_feat_mask = torch.cat(
            (visual_attention_mask, text_attention_mask), dim=1)

        output = self.transformer(
            features=transformer_input_feat,
            position_ids=transformer_input_feat_pos_id,
            token_type_ids=transformer_input_feat_token_id,
            attention_mask=transformer_input_feat_mask)

        #import pdb
        #pdb.set_trace()
        if self.output_split:
            #return torch.split(output,self.split_num,dim=1)
            return torch.split(output, transformed_im.shape[1],
                               dim=1), transformed_im, transformed_text
        else:
            return output, transformed_im, transformed_text


class OneModalEncoder(nn.Module):
    """
        One modality  Transformer Encoder model
    """

    def __init__(self, config, input_dim, hidden_dim):
        super(OneModalEncoder, self).__init__()
        self.linear = LinearLayer(in_hsz=input_dim, out_hsz=hidden_dim)
        self.transformer = TransformerBaseModel(config)

    #def forward(self, features, position_ids, token_type_ids, attention_mask, query_token_pos_embed):
    def forward(self, features, attention_mask, query_token_pos_embed):

        transformed_features = self.linear(features)

        #import pdb
        #pdb.set_trace()
        token_type_ids, position_ids = query_token_pos_embed(
            transformed_features, 1)

        output = self.transformer(features=transformed_features,
                                  position_ids=position_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        return output, transformed_features


class Biaffine(nn.Module):
    r"""
    Biaffine layer for first-order scoring.
    This function has a tensor of weights :math:`W` and bias terms if needed.
    The score :math:`s(x, y)` of the vector pair :math:`(x, y)` is computed as :math:`x^T W y`,
    in which :math:`x` and :math:`y` can be concatenated with bias terms.
    References:
        - Timothy Dozat and Christopher D. Manning. 2017.
          `Deep Biaffine Attention for Neural Dependency Parsing`_.
    Args:
        n_in (int):
            The size of the input feature.
        n_out (int):
            The number of output channels.
        bias_x (bool):
            If ``True``, adds a bias term for tensor :math:`x`. Default: ``True``.
        bias_y (bool):
            If ``True``, adds a bias term for tensor :math:`y`. Default: ``True``.
    .. _Deep Biaffine Attention for Neural Dependency Parsing:
        https://openreview.net/forum?id=Hk95PK9le
    """

    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(
            torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))

        self.reset_parameters()

    def __repr__(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return f"{self.__class__.__name__}({s})"

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        r"""
        Args:
            x (torch.Tensor): ``[batch_size, seq_len, n_in]``.
            y (torch.Tensor): ``[batch_size, seq_len, n_in]``.
        Returns:
            ~torch.Tensor:
                A scoring tensor of shape ``[batch_size, n_out, seq_len, seq_len]``.
                If ``n_out=1``, the dimension for ``n_out`` will be squeezed automatically.
        """

        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        # 16, 69, 501  |  1, 501, 500  | 16, 69, 500  -> 16, 1, 69, 69
        s = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s


class BiaffineLayer(nn.Module):

    def __init__(self, in_size1, in_size2, class_size, dropout=0.3):
        """
        :param inSize1: hidden_size
        :param inSize2: hidden_size
        :param classSize: label num

        下面的hidden_size+1中的1就对应着原文中的第二项：W * concat（v1，v2）add bias
        """
        super(BiaffineLayer, self).__init__()
        self.bilinearMap = nn.Parameter(
            torch.FloatTensor(in_size1, class_size, in_size2))
        self.classSize = class_size

    def forward(self, x1, x2):
        # [b, n, v1] -> [b*n, v1]
        # print("BIAFFINEPARA:", self.bilinearMap)
        #import pdb; pdb.set_trace()

        batch_size = x1.shape[0]
        bucket_size = x1.shape[1]  # seq_len

        x1 = torch.cat(
            (x1, torch.ones([batch_size, bucket_size, 1]).to(x1.device)),
            axis=2)
        x2 = torch.cat(
            (x2, torch.ones([batch_size, bucket_size, 1]).to(x2.device)),
            axis=2)
        # Static shape info
        vector_set_1_size = x1.shape[-1]
        vector_set_2_size = x2.shape[-1]

        # [b, seq_len, v1] -> [b * seq_len, v1]
        vector_set_1 = x1.reshape((-1, vector_set_1_size))

        # [v1, class_size, v2] -> [v1, class_size * v2]
        bilinear_map = self.bilinearMap.reshape((vector_set_1_size, -1))

        # [b * seq_len, v1] x [v1, class_size * v2] -> [b * seq_len, class_size * v2]
        bilinear_mapping = torch.matmul(vector_set_1, bilinear_map)

        # [b * seq_len, class_size * v2] -> [b, seq_len * class_size, v2]
        bilinear_mapping = bilinear_mapping.reshape(
            (batch_size, bucket_size * self.classSize, vector_set_2_size))

        # [b, seq_len * class_size, v2] x [b, seq_len, v2]T -> [b, seq_len*class_size, seq_len]
        bilinear_mapping = torch.matmul(bilinear_mapping, x2.transpose(1, -1))

        # [b, seq_len*class_size, seq_len] -> [b, seq_len, class_size, seq_len]
        bilinear_mapping = bilinear_mapping.reshape(
            (batch_size, bucket_size, self.classSize, bucket_size))

        #bilinear_mapping = torch.einsum('bxi,ioj,byj->bxyo', x1, self.bilinearMap, x2)
        return bilinear_mapping.transpose(-2, -1)


class MILNCELoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(MILNCELoss, self).__init__()
        self.reduction = reduction

    def forward(self, q2ctx_scores, contexts=None, queries=None):
        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        x = x.view(bsz, bsz, -1)
        nominator = x * torch.eye(
            x.shape[0], dtype=torch.float32, device=device)[:, :, None]
        nominator = nominator.sum(dim=1)
        nominator = torch.logsumexp(nominator, dim=1)
        denominator = torch.cat((x, x.permute(1, 0, 2)),
                                dim=1).view(x.shape[0], -1)
        denominator = torch.logsumexp(denominator, dim=1)
        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator

    def my_forward(self, q2ctx_scores=None, contexts=None, queries=None):
        if q2ctx_scores is None:
            assert contexts is not None and queries is not None
            x = torch.matmul(contexts, queries.t())
            device = contexts.device
            bsz = contexts.shape[0]
        else:
            x = q2ctx_scores
            device = q2ctx_scores.device
            bsz = q2ctx_scores.shape[0]

        # 找到每行的 top4 hard samples（排除对角线元素）
        indices = torch.arange(bsz, device=device)
        pos_score = x[indices, indices]
        x[indices, indices] = -1e10
        last_bsz = x.shape[0]
        if last_bsz < 10:
            hard_sample = last_bsz - 1
        if last_bsz > 10:
            hard_sample = 50
        max_indices = torch.topk(x, hard_sample, dim=1)[1]
        hard_samples = torch.gather(x, 1, max_indices)
        result = torch.cat([pos_score.unsqueeze(1), hard_samples], dim=1)

        nominator = torch.logsumexp(result[:, 0].unsqueeze(1), dim=1)
        denominator = torch.logsumexp(result, dim=1)

        if self.reduction:
            return torch.mean(denominator - nominator)
        else:
            return denominator - nominator


class VideoLocalAttention(nn.Module):

    def __init__(
        self,
        action_config,
    ):
        super(VideoLocalAttention, self).__init__()

        # self.convert_visual = LinearLayer(in_hsz=3584, out_hsz=256)
        # drop face feat
        # self.convert_visual = LinearLayer(in_hsz=3072, out_hsz=256)
        self.convert_visual = LinearLayer(in_hsz=4352, out_hsz=256)

        self.layernorm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        # self.face_to_linear = LinearLayer(
        #     in_hsz=512, out_hsz=256)

        # 128, 64, 32, 16, 8, 4
        self.visual_action_former = backbones.ConvTransformerBackbone(
            **action_config)
        self.config = action_config
        # self.visual_action_formers = nn.ModuleList([self.visual_action_former for _ in range(4)])

    def forward(self, video_feat, video_mask, ctx_pro):
        # assert video_feat.size()[2] == 3584, "video_feat 没有在最后一个维度拼接人脸！"
        video_feat = self.convert_visual(video_feat)  # 3584 ~> 256
        visual_token_type_ids, visual_position_ids = ctx_pro(video_feat, 0)
        # sub_token_type_ids, sub_position_ids = ctx_pro(sub_feat, 1)

        # embedding layer
        video_feat = video_feat + visual_token_type_ids + visual_position_ids
        # sub_feat = sub_feat + sub_token_type_ids + sub_position_ids

        video_feat = self.dropout(self.layernorm(video_feat))
        video_feat = expend_mask.fit_model_input(video_feat)  # bsz, 256, 128
        video_mask = expend_mask.fit_model_mask(video_mask)  # bsz, 1, 128
        # sub_feat = expend_mask.fit_model_input(sub_feat)  # bsz, 256, 128
        # sub_mask = expend_mask.fit_model_mask(sub_mask)  # bsz, 1, 128
        video_FPN, video_mask, = self.visual_action_former(
            video_feat, video_mask)
        # concate loca attention 的结果
        # video_feat = torch.cat(video_FPN, dim=-1)
        # video_mask = torch.cat(video_mask, dim=-1)
        video_feat = [feat.permute(0, 2, 1) for feat in video_FPN]
        video_mask = [mask.int().squeeze(1) for mask in video_mask]
        # video_feat = video_FPN[3]
        # video_mask = video_mask[3]

        #* video_mask = video_mask.int().squeeze(1)
        # video_mask = expend_mask.reverse_mask(video_mask.int()).squeeze(1) # mask 反了

        # 本来有linear层
        #* trans_img = video_feat.permute(0, 2, 1)
        # trans_sub = sub_feat.permute(0, 2, 1)
        # trans_img = self.img_linear(video_feat.permute(0, 2, 1))
        # trans_sub = self.sub_linear(sub_feat.permute(0, 2, 1))
        return video_feat, video_mask


class SubLocalAttention(nn.Module):

    def __init__(
        self,
        action_config,
    ):
        super(SubLocalAttention, self).__init__()
        self.convert_sub = LinearLayer(in_hsz=768, out_hsz=256)
        self.layernorm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)

        # 128, 64, 32, 16, 8, 4
        self.sub_action_former = backbones.ConvTransformerBackbone(
            **action_config)
        self.config = action_config
        # self.visual_action_formers = nn.ModuleList([self.visual_action_former for _ in range(4)])

    def forward(self, sub_feat, sub_mask, ctx_pro):
        sub_feat = self.convert_sub(sub_feat)  # 768 ~> 256
        sub_token_type_ids, sub_position_ids = ctx_pro(sub_feat, 1)
        # sub_token_type_ids, sub_position_ids = ctx_pro(sub_feat, 1)

        # embedding layer
        sub_feat = sub_feat + sub_token_type_ids + sub_position_ids
        # sub_feat = sub_feat + sub_token_type_ids + sub_position_ids

        sub_feat = self.dropout(self.layernorm(sub_feat))
        sub_feat = expend_mask.fit_model_input(sub_feat)  # bsz, 256, 128
        sub_mask = expend_mask.fit_model_mask(sub_mask)  # bsz, 1, 128
        sub_FPN, sub_mask, = self.sub_action_former(sub_feat, sub_mask)
        # concate local attention 的结果
        # sub_feat = torch.cat(sub_FPN, dim=-1)
        # sub_mask = torch.cat(sub_mask, dim=-1)
        # subtitle 是 global attention
        sub_feat = [feat.permute(0, 2, 1) for feat in sub_FPN]
        sub_mask = [mask.int().squeeze(1) for mask in sub_mask]
        # sub_feat = sub_FPN[3]
        # sub_mask = sub_mask[3]

        # sub_mask = sub_mask.int().squeeze(1)
        # video_mask = expend_mask.reverse_mask(video_mask.int()).squeeze(1) # mask 反了

        # 本来有linear层
        # trans_sub = sub_feat.permute(0, 2, 1)
        return sub_feat, sub_mask


class LocalAttention(nn.Module):

    def __init__(self, action_config, config):
        super(LocalAttention, self).__init__()
        # self.convert_visual = LinearLayer(
        #     in_hsz=256, out_hsz=256)
        self.img_linear = LinearLayer(in_hsz=256, out_hsz=256)
        self.sub_linear = LinearLayer(in_hsz=256, out_hsz=256)
        self.convert_visual = LinearLayer(in_hsz=3584, out_hsz=256)
        # self.convert_visual = LinearLayer(
        #     in_hsz=3072, out_hsz=256)
        self.convert_sub = LinearLayer(in_hsz=768, out_hsz=256)

        self.layernorm = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.1)
        # self.face_to_linear = LinearLayer(
        #     in_hsz=512, out_hsz=256)

        # 128, 64, 32, 16, 8, 4
        self.visual_action_former = backbones.ConvTransformerBackbone(
            **action_config)
        self.config = action_config
        # self.visual_action_formers = nn.ModuleList([self.visual_action_former for _ in range(4)])

    def forward(self, video_feat, video_mask, sub_feat, sub_mask, ctx_pro):
        assert video_feat.size()[2] == 3584, "video_feat 没有在最后一个维度拼接人脸！"
        video_feat = self.convert_visual(video_feat)  # 3584 ~> 256
        sub_feat = self.convert_sub(sub_feat)  # 768 ~> 256
        visual_token_type_ids, visual_position_ids = ctx_pro(video_feat, 0)
        sub_token_type_ids, sub_position_ids = ctx_pro(sub_feat, 1)

        # embedding layer
        video_feat = video_feat + visual_token_type_ids + visual_position_ids
        sub_feat = sub_feat + sub_token_type_ids + sub_position_ids

        video_feat, sub_feat = torch.split(
            (self.dropout(
                self.layernorm(torch.cat([video_feat, sub_feat], dim=1)))),
            [video_feat.size()[1], video_feat.size()[1]],
            dim=1)
        # self.dropout
        video_feat = expend_mask.fit_model_input(video_feat)  # bsz, 256, 128
        video_mask = expend_mask.fit_model_mask(video_mask)  # bsz, 1, 128
        sub_feat = expend_mask.fit_model_input(sub_feat)  # bsz, 256, 128
        sub_mask = expend_mask.fit_model_mask(sub_mask)  # bsz, 1, 128
        video_FPN, sub_FPN, video_mask, = self.visual_action_former(
            video_feat, video_mask, sub_feat, sub_mask)
        video_feat = video_FPN[2]
        sub_feat = sub_FPN[2]
        video_mask = video_mask[2]

        video_mask = video_mask.int().squeeze(1)
        # video_mask = expend_mask.reverse_mask(video_mask.int()).squeeze(1) # mask 反了

        # 本来有linear层
        trans_img = video_feat.permute(0, 2, 1)
        trans_sub = sub_feat.permute(0, 2, 1)
        # trans_img = self.img_linear(video_feat.permute(0, 2, 1))
        # trans_sub = self.sub_linear(sub_feat.permute(0, 2, 1))
        return trans_img, trans_sub, video_mask


class MyTwoModalEncoder(nn.Module):
    """
        Two modality Transformer Encoder model
    """

    def __init__(self, config, split_num, output_split=True, with_emb=True):
        super(MyTwoModalEncoder, self).__init__()

        self.transformer = TransformerBaseModel(config, with_emb=with_emb)
        self.output_split = output_split
        if self.output_split:
            self.split_num = split_num

    def forward(self, visual_features, visual_attention_mask, text_features,
                text_attention_mask, ctx_token_pos_embed):

        visual_token_type_ids, visual_position_ids = ctx_token_pos_embed(
            visual_features, 0)
        text_token_type_ids, text_position_ids = ctx_token_pos_embed(
            text_features, 1)
        transformer_input_feat = torch.cat((visual_features, text_features),
                                           dim=1)
        transformer_input_feat_pos_id = torch.cat(
            (visual_position_ids, text_position_ids), dim=1)
        transformer_input_feat_token_id = torch.cat(
            (visual_token_type_ids, text_token_type_ids), dim=1)
        transformer_input_feat_mask = torch.cat(
            (visual_attention_mask, text_attention_mask), dim=1)

        output = self.transformer(
            features=transformer_input_feat,
            position_ids=transformer_input_feat_pos_id,
            token_type_ids=transformer_input_feat_token_id,
            attention_mask=transformer_input_feat_mask)

        if self.output_split:
            #return torch.split(output,self.split_num,dim=1)
            return torch.split(output, visual_features.shape[1],
                               dim=1), visual_features, text_features


class MyOneModalEncoder(nn.Module):
    """
        One modality  Transformer Encoder model
    """

    def __init__(self, config, input_dim, hidden_dim):
        super(MyOneModalEncoder, self).__init__()
        self.linear = LinearLayer(in_hsz=256, out_hsz=hidden_dim)
        self.transformer = TransformerBaseModel(config)

    #def forward(self, features, position_ids, token_type_ids, attention_mask, query_token_pos_embed):
    def forward(self, features, attention_mask, query_token_pos_embed):

        transformed_features = self.linear(features)

        #import pdb
        #pdb.set_trace()
        token_type_ids, position_ids = query_token_pos_embed(
            transformed_features, 1)

        output = self.transformer(features=transformed_features,
                                  position_ids=position_ids,
                                  token_type_ids=token_type_ids,
                                  attention_mask=attention_mask)
        return output, transformed_features


class AdaptFace(nn.Module):

    def __init__(self, input_dim, output_dim, n_heads):
        super(AdaptFace, self).__init__()
        self.n_heads = n_heads
        self.query = nn.Linear(input_dim, output_dim)
        self.key = nn.Linear(input_dim, output_dim)
        self.value = nn.Linear(input_dim, output_dim)

    def transpose_for_scores(self, x):

        # 检查正确的多头数量
        assert x.size()[-1] % self.n_heads == 0, "多头数量不能被 hidd_dim 整除"
        new_shape = x.size()[:-1] + (self.n_heads,
                                     x.size()[-1] // self.n_heads)

        # [bsz, seq, n_heads, ds]
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def fit_chunk_model(self, x):

        # 适应模型的输入
        assert len(x.size()) == 4, "face_qkv不满足[bsz, n_heads, seq, ds]形状"
        bsz, n_heads, seq, ds = x.size()
        x = x.contiguous().view(bsz * n_heads, seq, ds)
        return x

    def forward(self, face_feat, face_mask):

        mixed_query_layer = self.query(face_feat)
        mixed_key_layer = self.key(face_feat)
        mixed_value_layer = self.value(face_feat)

        # [bsz, n_heads, seq, ds]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        face_query = self.fit_chunk_model(query_layer)
        face_key = self.fit_chunk_model(key_layer)
        face_value = self.fit_chunk_model(value_layer)

        return face_query, face_key, face_value, face_mask


class NetVLAD(nn.Module):

    def __init__(self, cluster_size, feature_size, add_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1 / math.sqrt(feature_size)) *
                                     torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter(
            (1 / math.sqrt(feature_size)) *
            torch.randn(1, feature_size, cluster_size))

        self.add_norm = add_norm
        self.LayerNorm = nn.LayerNorm(cluster_size)
        self.out_dim = cluster_size * feature_size

    def forward(self, x, mask):
        max_sample = x.size()[1]
        x = x.contiguous().view(-1, self.feature_size)
        # assignment形状: (batch_size * max_sample, cluster_size)
        assignment = torch.matmul(x, self.clusters)

        if self.add_norm:
            assignment = self.LayerNorm(assignment)

        assignment = assignment.view(-1, max_sample, self.cluster_size)
        # 加入的mask 之后的 assignment
        mask = mask.unsqueeze(-1)  # mask: [bsz, seq, 1]
        assignment.masked_fill_(~mask.bool(), -1e14)
        assignment = F.softmax(assignment, dim=1)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)
        # assignment形状: (batch_size, self.cluster_size, max_sample)
        # x形状: (batch_size, max_sample, self.feature_size)
        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)
        # 修改:
        vlad = vlad.contiguous().view(-1,
                                      self.feature_size * self.cluster_size)
        vlad = F.normalize(vlad)
        return vlad, vlad.contiguous().view(-1, self.feature_size,
                                            self.cluster_size).transpose(1, 2)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad, vlad.view(-1, self.cluster_size, self.feature_size)


class NVLDModel(nn.Module):

    def __init__(self, config):
        super(NVLDModel, self).__init__()
        # self.convert_feat = nn.Linear(config.init_size,
        #                               config.hidden_size)
        self.text_pooling = NetVLAD(feature_size=config.hidden_size,
                                    cluster_size=config.text_cluster)
        self.dropout = nn.Dropout(config.moe_dropout_prob)
        self.fc_lyr = nn.Linear(in_features=config.hidden_size,
                                out_features=2,
                                bias=False)

    def forward(self, query_feat, query_mask):
        # 修改: pooled_text: [bsz, clusters, d]
        # pooled_text = self.text_pooling(query_feat)
        # query_feat = self.convert_feat(query_feat)
        _, pooled_text = self.text_pooling(query_feat, query_mask)
        # pooled_text = self.transformer(features=pooled_text, position_ids=None, token_type_ids=None, attention_mask=torch.ones(pooled_text.size()[:-1]).to(device=pooled_text.device))
        pooled_text = self.dropout(pooled_text)
        return pooled_text

class TecentAggregation(nn.Module):

    def __init__(self, low_dim, high_seq):
        super(TecentAggregation, self).__init__()
        self.high_level_att = HighlevelAttention(low_dim, high_seq)
        self.two_layer = TwoLayer(low_dim)


    def forward(self, x, x_mask):
        # x: [bsz, low_feat_seq, d_dim]
        # low_feat: [bsz, high_feat_seq, low_feat_seq]
        high_level_att = self.high_level_att(x, x_mask)

        # low_feat: [bsz, low_feat_seq, d_dim]
        mlp_feat = self.two_layer(x)
        high_feat = torch.einsum("bcw,bwd->bcd", high_level_att, mlp_feat)
        return high_feat

class HighlevelAttention(nn.Module):

    def __init__(self, low_dim, high_seq):
        super(HighlevelAttention, self).__init__()
        self.fc = nn.Linear(low_dim, high_seq)

    def forward(self, low_feat, low_feat_mask):
        # low_feat: [bsz, low_feat_seq, low_feat_dim]
        # x: [bsz, low_feat_seq, high_feat_seq]
        x = self.fc(low_feat)
        # x: [bsz, high_feat_seq, low_feat_seq]
        x = x.masked_fill_(~low_feat_mask.bool().unsqueeze(2), -1e14)
        x = F.softmax(x, dim=1).permute(0, 2, 1)
        return x

class TwoLayer(nn.Module):

    def __init__(self, d_dim):
        super(TwoLayer, self).__init__()
        self.linear1 = nn.Linear(d_dim, 2 * d_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2 * d_dim, d_dim)
        self.ln = nn.LayerNorm(d_dim)

    def forward(self, low_feat):
        x = self.linear1(low_feat)
        x = self.relu1(x)
        # x: [bsz, low_feat_seq, d_dim]
        x = self.linear2(x)
        x = self.ln(x)
        return x

""" Computation helpers """
def apply_on_sequence(layer, inp):
    " For nn.Linear, this fn is DEPRECATED "
    # inp = to_contiguous(inp)
    inp = inp.contiguous()
    inp_size = list(inp.size())
    output = layer(inp.view(-1, inp_size[-1]))
    output = output.view(*inp_size[:-1], -1)
    return output
class Attention(nn.Module):
    def __init__(self, config, prefix=""):
        super(Attention, self).__init__()
        name = prefix if prefix == "" else prefix+"_"
        print("Attention - ", name)

        # parameters
        kdim = config.get(name+"att_key_dim", 256)
        cdim = config.get(name+"att_cand_dim", 256)
        att_hdim = config.get(name+"att_hdim", 128)
        # kdim = config.get(name+"att_key_dim", -1)
        # cdim = config.get(name+"att_cand_dim", -1)
        # att_hdim = config.get(name+"att_hdim", -1)
        drop_p = config.get(name+"att_drop_prob", 0.0)

        # layers
        self.key2att = nn.Linear(kdim, att_hdim)
        self.feat2att = nn.Linear(cdim, att_hdim)
        self.to_alpha = nn.Linear(att_hdim, 1)
        self.drop = nn.Dropout(drop_p)

    def forward(self, key, feats, feat_masks=None, return_weight=True):
        """ Compute attention weights and attended feature (weighted sum)
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2, "{} != 2".format(len(key.size()))
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)

        # compute attention weights
        logits = self.compute_att_logits(key, feats, feat_masks) # [B,A]
        #* 修改
        logits = logits.masked_fill(torch.relu(logits).float().eq(0), -1e14)
        weight = self.drop(F.softmax(logits, dim=1))             # [B,A]

        # compute weighted sum: bmm working on (B,1,A) * (B,A,D) -> (B,1,D)
        att_feats = torch.bmm(weight.unsqueeze(1), feats).squeeze(1) # B * D
        if return_weight:
            return att_feats, weight
        return att_feats

    def compute_att_logits(self, key, feats, feat_masks=None):
        """ Compute attention weights
        Args:
            key: key vector to compute attention weights; [B, K]
            feats: features where attention weights are computed; [B, A, D]
            feat_masks: mask for effective features; [B, A]
        """
        # check inputs
        assert len(key.size()) == 2
        assert len(feats.size()) == 3 or len(feats.size()) == 4
        assert feat_masks is None or len(feat_masks.size()) == 2

        # dealing with dimension 4
        if len(feats.size()) == 4:
            B, W, H, D = feats.size()
            feats = feats.view(B, W*H, D)
        A = feats.size(1)

        # embedding key and feature vectors
        att_f = apply_on_sequence(self.feat2att, feats)   # B * A * att_hdim
        att_k = self.key2att(key)                                   # B * att_hdim
        att_k = att_k.unsqueeze(1).expand_as(att_f)                 # B * A * att_hdim

        # compute attention weights
        dot = torch.tanh(att_f + att_k)                             # B * A * att_hdim
        # dot = torch.tanh(att_f * att_k)                             # B * A * att_hdim
        alpha = apply_on_sequence(self.to_alpha, dot)     # B * A * 1
        alpha = alpha.view(-1, A)                                   # B * A
        if feat_masks is not None:
            alpha = alpha.masked_fill(feat_masks.float().eq(0), -1e9)

        return alpha
class SequentialQueryAttention(nn.Module):
    def __init__(self, config):
        super(SequentialQueryAttention, self).__init__()

        self.nse = config.get("num_semantic_entity", 8)
        self.qdim = config.get("sqan_qdim", 256) # 512
        # self.nse = config.get("num_semantic_entity", -1)
        # self.qdim = config.get("sqan_qdim", -1) # 512
        self.global_emb_fn = nn.ModuleList( # W_q^(n) in Eq. (4)
                [nn.Linear(self.qdim, self.qdim) for i in range(self.nse)])
        self.guide_emb_fn = nn.Sequential(*[
            nn.Linear(2*self.qdim, self.qdim), # W_g in Eq. (4)
            nn.ReLU()
        ])
        self.att_fn = Attention(config, "sqan")

    def forward(self, q_feats, w_feats, w_mask=None):
        """ extract N (=nse) semantic entity features from query
        Args:
            q_feats: sentence-level feature; [B,qdim]
            w_feats: word-level features; [B,L,qdim]
            w_mask: mask for effective words; [B,L]
        Returns:
            se_feats: semantic entity features; [B,N,qdim]
            se_attw: attention weight over words; [B,N,L]
        """

        B = w_feats.size(0)
        # 修改
        # prev_se = w_feats.new_zeros(B, self.qdim)
        prev_se = w_feats.new_empty(B, self.qdim).normal_()
        se_feats, se_attw = [], []
        # compute semantic entity features sequentially
        for n in range(self.nse):
            # perform Eq. (4)
            q_n = self.global_emb_fn[n](q_feats) # [B,qdim] -> [B,qdim]
            # 修改
            g_n = self.guide_emb_fn(torch.cat([q_n, prev_se], dim=1)) # [B,2*qdim] -> [B,qdim]
            # perform Eq. (5), (6), (7)
            att_f, att_w = self.att_fn(g_n, w_feats, w_mask)

            prev_se = att_f
            se_feats.append(att_f)
            se_attw.append(att_w)

        return torch.stack(se_feats, dim=1), torch.stack(se_attw, dim=1)

class DQALoss(nn.Module):
    def __init__(self, config, prefix=""):
        super(DQALoss, self).__init__()
        self.name = prefix if prefix == "" else prefix+"_"
        print("Distinct Query Attention Loss - ", self.name)

        self.w = config.get("dqa_weight", 1.0)
        self.r = config.get("dqa_lambda", 0.2)

    def forward(self, att_matrix, gts):
        """ loss function to diversify attention weights
        Args:
            att_matrix: words和 sentic 注意力矩阵
            gts: dictionary of ground-truth
        Returns:
            loss: loss value; [1], float tensor
        """
        attw = att_matrix # [B,num_att,N]
        NA = attw.size(1)

        attw_T = torch.transpose(attw, 1, 2).contiguous()

        I = torch.eye(NA).unsqueeze(0).type_as(attw) * self.r
        #pdb.set_trace()
        P = torch.norm(torch.bmm(attw, attw_T) - I, p="fro", dim=[1,2], keepdim=True)
        #P = torch.norm(torch.bmm(attw, attw_T) - I, p=2, dim=[1,2], keepdim=True)
        #P = torch.bmm(attw, attw_T) - I
        #P = torch.norm(P.cpu(), p="fro", dim=[1,2], keepdim=True).cuda()

        if torch.isnan(P).sum() > 0:
            print("attw: ", attw)
            pdb.set_trace()

        da_loss = self.w * (P**2).mean()

        return da_loss
    # def forward(self, net_outs, gts):
    #     """ loss function to diversify attention weights
    #     Args:
    #         net_outs: dictionary of network outputs
    #         gts: dictionary of ground-truth
    #     Returns:
    #         loss: loss value; [1], float tensor
    #     """
    #     attw = net_outs[self.name+"dqa_attw"] # [B,num_att,N]
    #     NA = attw.size(1)

    #     attw_T = torch.transpose(attw, 1, 2).contiguous()

    #     I = torch.eye(NA).unsqueeze(0).type_as(attw) * self.r
    #     #pdb.set_trace()
    #     P = torch.norm(torch.bmm(attw, attw_T) - I, p="fro", dim=[1,2], keepdim=True)
    #     #P = torch.norm(torch.bmm(attw, attw_T) - I, p=2, dim=[1,2], keepdim=True)
    #     #P = torch.bmm(attw, attw_T) - I
    #     #P = torch.norm(P.cpu(), p="fro", dim=[1,2], keepdim=True).cuda()

    #     if torch.isnan(P).sum() > 0:
    #         print("attw: ", attw)
    #         pdb.set_trace()

    #     da_loss = self.w * (P**2).mean()

    #     return da_loss
class HBIPooling(nn.Module):

    def __init__(self, config):
        super(HBIPooling, self).__init__()
        self.k = config.k
        self.ctm = CTM(sample_ratio=config.sample_ratio,
        embed_dim=config.embed_dim, dim_out=config.dim_out, k=config.k)
        self.block =TCBlock(dim=config.dim_out, num_heads=config.num_heads)

    def forward(self, video_feat, video_mask):
        bsz, maxl, d = video_feat.size()
        # idx_token: [bsz, maxl,]
        idx_token = torch.arange(maxl).unsqueeze(0).repeat(bsz, 1, 1).to(video_feat.device)
        agg_weight = video_feat.new_ones(bsz, maxl, 1)
        token_dict = {
            "x": video_feat,
            "token_num": maxl,
            "idx_token": idx_token,
            "agg_weight": agg_weight,
            "mask": video_mask,
        }
        down_dict = self.ctm(token_dict)
        q_dict = self.block(down_dict)
        return q_dict["x"], q_dict["x"].new_ones(bsz, q_dict["x"].size()[1])

class KL(nn.Module):
    def __init__(self, ):
        super(KL, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        kl = F.kl_div(logpt0, logpt1, reduction='mean')
        return kl
class CalEventLevel(nn.Module):

    def __init__(self, hidden_size=256):
        super(CalEventLevel, self).__init__()
        self.text_weight_fc1 = nn.Linear(hidden_size, 1)
        self.video_weight_fc1 = nn.Linear(hidden_size, 1)

    def forward(self, text_feat, video_feat, text_mask, video_mask, is_eval=False):
        text_weight = self.text_weight_fc1(text_feat).squeeze(2)  # B x N_t x D -> B x N_t
        text_weight = torch.softmax(text_weight, dim=-1)  # B x N_t

        video_weight = self.video_weight_fc1(video_feat).squeeze(2)  # B x N_v x D -> B x N_v
        video_weight = torch.softmax(video_weight, dim=-1)  # B x N_v

        text_feat = F.normalize(text_feat, dim=-1)
        video_feat = F.normalize(video_feat, dim=-1)
        # text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        # video_feat = video_feat / video_feat.norm(dim=-1, keepdim=True)
        retrieve_logits = torch.einsum('atd,bvd->abtv', [text_feat, video_feat])
        retrieve_logits = torch.einsum('abtv,at->abtv', [retrieve_logits, text_mask])
        retrieve_logits = torch.einsum('abtv,bv->abtv', [retrieve_logits, video_mask])
        if is_eval:
            return retrieve_logits
        t2v_logits, max_idx1 = retrieve_logits.max(dim=-1)  # abtv -> abt
        t2v_logits = torch.einsum('abt,at->ab', [t2v_logits, text_weight])

        v2t_logits, max_idx2 = retrieve_logits.max(dim=-2)  # abtv -> abv
        v2t_logits = torch.einsum('abv,bv->ab', [v2t_logits, video_weight])

        _retrieve_logits = (t2v_logits + v2t_logits) / 2.0

        return _retrieve_logits, _retrieve_logits.T, retrieve_logits

class FineGrainGround(nn.Module):

    def __init__(self, config):
        super(FineGrainGround, self).__init__()
        self.config = config
        # attention every word for each clips in tne video
        self.agg_words_conv1D_st = nn.Conv1d(in_channels=config.max_desc_l, out_channels=1, kernel_size=5, padding=2)
        self.agg_words_conv1D_ed = nn.Conv1d(in_channels=config.max_desc_l, out_channels=1, kernel_size=5, padding=2)
    def forward(self, fine_grain_word_ctx, ctx_mask):
        # padding
        if fine_grain_word_ctx.size()[2] != self.config.max_desc_l:
            fine_grain_word_ctx = F.pad(fine_grain_word_ctx, [0, 0, 0, self.config.max_desc_l-fine_grain_word_ctx.size()[2], 0, 0, 0, 0])
        # fine_grain_word_ctx: [qbsz, vbsz, n_word, n_clips]
        q_bsz, v_bsz, n_word, n_clips = fine_grain_word_ctx.size()
        fine_grain_word_ctx = fine_grain_word_ctx.contiguous().view(q_bsz * v_bsz, n_word, n_clips)
        # fine_grain_word_ctx ~> [qbsz * vbsz, out_channels(1), n_clips]
        fine_grain_word_ctx_conv_st = self.agg_words_conv1D_st(fine_grain_word_ctx)
        fine_grain_word_ctx_conv_ed = self.agg_words_conv1D_ed(fine_grain_word_ctx)
        # fine_grain_word_ctx_conv ~> [q_bsz, v_bsz, n_clips]
        fine_grain_word_ctx_conv_st = fine_grain_word_ctx_conv_st.squeeze(1).view(q_bsz, v_bsz, n_clips)
        fine_grain_word_ctx_conv_ed = fine_grain_word_ctx_conv_ed.squeeze(1).view(q_bsz, v_bsz, n_clips)

        fine_grain_word_ctx_conv_st = mask_logits(fine_grain_word_ctx_conv_st, ctx_mask)
        fine_grain_word_ctx_conv_ed = mask_logits(fine_grain_word_ctx_conv_ed, ctx_mask)
        assert fine_grain_word_ctx_conv_st.size()[0] == fine_grain_word_ctx_conv_st.size()[1]
        # 使用 torch.arange 来生成对角线的索引
        indices = torch.arange(fine_grain_word_ctx_conv_st.size()[0])
        return fine_grain_word_ctx_conv_st[indices, indices], fine_grain_word_ctx_conv_ed[indices, indices]

class SVMR_Train(nn.Module):

    def __init__(self, config, model_config):
        super(SVMR_Train, self).__init__()
        self.my_triple_Encoder = ThreeModalEncoder(
            config=model_config.triplet_config,
            img_dim=config.hidden_size,
            text_dim=config.hidden_size,
            query_dim=config.hidden_size,
            hidden_dim=config.hidden_size,
            split_num=config.max_ctx_l,
        )
    def forward(self, video_feat, video_mask, sub_feat, sub_mask, word_feat, word_mask, ctx_token_pos_embed, query_token_pos_embed, is_eval=False):
        # word_feat: [qbsz, lw, ward_dim]
        # context_feat: [qbsz, sample_num, lc, context_dim]
        qbsz, sample_num, lv, video_dim = video_feat.shape
        _video_feat = []
        _sub_feat = []
        for i in range(sample_num):
            (tmp_video_feat, tmp_sub_feat, tmp_query), _, _ = self.my_triple_Encoder(video_feat[:, i, :, :], video_mask[:, i, :], \
                sub_feat[:, i, :, :], sub_mask[:, i, :], word_feat, word_mask, \
                ctx_token_pos_embed, query_token_pos_embed, is_eval)
            _video_feat.append(tmp_video_feat)
            _sub_feat.append(tmp_sub_feat)
        # _video_feat: [qbsz, sample_num, lv, hidden_dim]
        _video_feat = torch.stack(_video_feat, dim=1)
        _sub_feat = torch.stack(_sub_feat, dim=1)
        return _video_feat, _sub_feat

    def infer(self, video_feat, video_mask, sub_feat, sub_mask, word_feat, word_mask, ctx_token_pos_embed, query_token_pos_embed, is_eval):
        return self.my_triple_Encoder(video_feat, video_mask, \
                sub_feat, sub_mask, word_feat, word_mask, \
                ctx_token_pos_embed, query_token_pos_embed, is_eval)
