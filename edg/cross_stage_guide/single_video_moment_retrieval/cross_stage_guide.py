from pathlib import Path

import numpy as np

from .start_end_dataset_with_face import \
    start_end_collate, prepare_batch_inputs
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from edg.modules.model_components import \
    LinearLayer, TrainablePositionalEncoding
from edg.utils.model_utils import MLP
from edg.utils.basic_utils import load_config
from edg.modules.transformer.bert import BertEncoder
from edg.modules.transformer.bert_embed import BertEmbeddings

from third_party.actionformer.libs.modeling import backbones, expend_mask
from edg.modules.cluster_merge import *
from edg.modules.cross_stage_losses import *

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
    "mha_win_size": [4, 8, 16, 4],  # size of local window for mha
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
    "mha_win_size": [4, 8, 16, 4],  # size of local window for mha
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
    sample_ratio = 0.25,
    embed_dim = 256,
    dim_out = 256,
    k = 3,
    num_heads = 8
)


class CrossStageGuide(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.query_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_desc_l,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop,
            is_query=True)

        # 修改
        self.sub_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)

        # self.ctx_token_pos_embed = TrainablePositionalEncoding(
        #     # max_position_embeddings=config.max_ctx_l,
        #     max_position_embeddings=212,  # fit 拼接过后的 video
        #     type_vocab_size=2,
        #     hidden_size=config.hidden_size,
        #     dropout=config.input_drop)
        self.ctx_token_pos_embed = TrainablePositionalEncoding(
            max_position_embeddings=config.max_ctx_l,
            type_vocab_size=2,
            hidden_size=config.hidden_size,
            dropout=config.input_drop)
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
        my_model_config = load_config(str(config_dir / "my_model_config.json"))
        self.t2vVLAD = NVLDModel(my_model_config.netvlad_config)
        SQAN_config = config_dir / "LG_config.yaml"
        # 加载配置文件
        with open(SQAN_config, 'r') as f:
            SQAN_config = yaml.safe_load(f)
        self.bidirect = BidirectionalAttention(video_dim=256)
        # 初始化 QuerySequenceEncoder 实例
        self.l2g_query_video_encoder = SequentialQueryAttention(SQAN_config)
        self.l2g_query_sub_encoder = SequentialQueryAttention(SQAN_config)
        self.kl = KL()
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
        self.visual_pro = nn.Linear(config.hidden_size, config.hidden_size)
        self.sub_pro = nn.Linear(config.hidden_size, config.hidden_size)
        self.fusion_pro = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.self = OneModalEncoder(
            config=model_config.bert_config,
            input_dim=256,
            hidden_dim=config.hidden_size,
            with_emb=False,
        )
        self.cross = CrossAttention(dim=256, context_dim=256)
        self.st_begin = ConvSE()
        self.ed_end = ConvSE()
        # self.nce_criterion = MILNCELoss(reduction='mean')
        self.discriminator = nn.Sequential(
            nn.Linear(config.hidden_size * 5, 1),
            nn.Softplus(),
            )
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

    def get_svmr_loss(self, st_ed_indices, word_feat, word_mask, \
                      pos_video_feat, pos_sub_feat, neg_video_feat, neg_sub_feat, \
                      pos_video_mask, pos_sub_mask, neg_video_mask, neg_sub_mask, span_mask, weight_hard):
        """
        params:
            word_feat: (qbsz, lw, d)
            word_mask: (qbsz, lw)
            pos_video_feat: (bsz, lv, d)
            pos_video_mask: (bsz, lv,)
            neg_sub_feat: (bsz, sample, lv, d)
            neg_sub_mask: (bsz, sample, lv)
        """
        qbsz = word_feat.size(0)
        sample = neg_video_feat.size(1)
        # visual_query, sub_query = self.encode_query_word(word_feat, word_mask)
        # # visual_query_emb: [bsz, 1, d]
        # visual_query_emb = visual_query.unsqueeze(1)
        # sub_query_emb = sub_query.unsqueeze(1)
        # pos_video_feat_pro = self.visual_pro(pos_video_feat)
        # pos_sub_feat_pro = self.sub_pro(pos_sub_feat)
        # pos_video_feat_tmp = torch.mul(F.normalize(torch.mul(pos_video_feat_pro, visual_query_emb), p=2, dim=-1), pos_video_feat)
        # pos_sub_feat_tmp = torch.mul(F.normalize(torch.mul(pos_sub_feat_pro, sub_query_emb), p=2, dim=-1), pos_sub_feat)
        # pos_clip_feat = torch.cat([pos_video_feat_tmp, pos_sub_feat_tmp], dim=2)
        # pos_clip_feat = self.fusion_pro(pos_clip_feat)
        # # self.self没有PE 和 ME 编码
        # pos_clip_feat, _ = self.self(pos_clip_feat, pos_video_mask, self.V2ctx_token_pos_embed)
        # pos_clip_feat = self.cross(pos_clip_feat, pos_video_mask, word_feat, word_mask)
        # tmp_st_prob = self.st_begin(pos_clip_feat, pos_video_mask)
        # tmp_ed_prob = self.ed_end(pos_clip_feat, pos_video_mask)
        _, tmp_st_prob, tmp_ed_prob = self.bidirect(word_feat, pos_video_feat, pos_sub_feat, pos_video_mask, word_mask)
        _st_prob = []
        _ed_prob = []
        _st_prob.append(tmp_st_prob)
        _ed_prob.append(tmp_ed_prob)
        for i in range(sample):
            # neg_video_feat_pro = self.visual_pro(neg_video_feat[:, i])
            # neg_sub_feat_pro = self.sub_pro(neg_sub_feat[:, i])
            # neg_video_feat_tmp = torch.mul(F.normalize(torch.mul(neg_video_feat_pro, visual_query_emb), p=2, dim=-1), neg_video_feat[:, i])
            # neg_sub_feat_tmp = torch.mul(F.normalize(torch.mul(neg_sub_feat_pro, sub_query_emb), p=2, dim=-1), neg_sub_feat[:, i])
            # neg_clip_feat = torch.cat([neg_video_feat_tmp, neg_sub_feat_tmp], dim=2)
            # neg_clip_feat = self.fusion_pro(neg_clip_feat)
            # neg_clip_feat, _ = self.self(neg_clip_feat, neg_video_mask[:, i], self.V2ctx_token_pos_embed)
            # neg_clip_feat = self.cross(neg_clip_feat, neg_video_mask[:, i], word_feat, word_mask)
            _, tmp_st_prob, tmp_ed_prob = self.bidirect(word_feat, neg_video_feat[:, i], neg_sub_feat[:, i], neg_video_mask[:, i], word_mask)
            # tmp_st_prob = self.st_begin(tmp_st_prob, neg_video_mask[:, i])
            # tmp_ed_prob = self.ed_end(tmp_ed_prob, neg_video_mask[:, i])
            _st_prob.append(tmp_st_prob)
            _ed_prob.append(tmp_ed_prob)
        st_prob = torch.stack(_st_prob)
        ed_prob = torch.stack(_ed_prob)
        st_prob = st_prob.permute(1, 0, 2).contiguous().view(qbsz, -1)
        ed_prob = ed_prob.permute(1, 0, 2).contiguous().view(qbsz, -1)

        _loss_st, _loss_ed = self.share_norm_loss.moment_share_loss(
            st_prob, ed_prob, st_ed_indices, pos_video_mask, weight_hard)
        # intra_loss = self.share_norm_loss.moment_intra_cl_loss(span_mask, st_prob, ed_prob, pos_video_mask)
        intra_loss = 0
        # vv_loss_st = self.share_norm_loss.share_vv_contrastive_loss(
        #     discriminator_R, sapn_mask, pos_video_mask, neg_video_mask)
        # kl_loss_st, kl_loss_ed = self.share_norm_loss.moment_gaussian_loss(st_prob, ed_prob, st_ed_indices, pos_video_mask)
        loss_st = _loss_st * 0.01
        loss_ed = _loss_ed * 0.01
        # 排除 vv_loss_st 影响
        # vv_loss_st = 0
        loss_st_ed = loss_st + loss_ed
        # loss_moment_kl = kl_loss_st + kl_loss_ed
        return loss_st_ed, intra_loss

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
        neg_video_feat,
        neg_video_mask,
        neg_sub_feat,
        neg_sub_mask,
        weight_hard,
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
        (video_feat, sub_feat
         ), self.video_input_proj, self.sub_input_proj = self.videoEncoder(
             video_feat, video_mask, sub_feat, sub_mask,
             self.ctx_token_pos_embed)
        # _, ml = video_mask.size()

        # # video_feat: ([bsz, ml, d], [bsz, 64, d], [bsz, 32, d])
        # # sub_feat: ([bsz, ml, d], [bsz, 64, d], [bsz, 32, d])
        # # positive_video_feat
        # video_feat, video_mask = self.videolocal(video_feat, video_mask,
        #                                          self.V1ctx_token_pos_embed)
        # sub_feat, sub_mask = self.subglobal(sub_feat, sub_mask,
        #                                     self.V2ctx_token_pos_embed)
        neg_sample = 5
        assert neg_video_feat[0].shape[0] == neg_sample, "负样本数量不等于 5 "
        local_neg_video_feat = []
        local_neg_sub_feat = []
        local_neg_video_mask = []
        local_neg_sub_mask = []
        for i in range(neg_sample):
            (_neg_video_feat, _neg_sub_feat
             ), self.video_input_proj, self.sub_input_proj = self.videoEncoder(
                 neg_video_feat[:, i], neg_video_mask[:, i], neg_sub_feat[:, i], neg_sub_mask[:, i],
                 self.ctx_token_pos_embed)
            _neg_video_mask = neg_video_mask[:, i]
            _neg_sub_mask = neg_sub_mask[:, i]
            local_neg_video_feat.append(_neg_video_feat)
            local_neg_sub_feat.append(_neg_sub_feat)
            local_neg_video_mask.append(_neg_video_mask)
            local_neg_sub_mask.append(_neg_sub_mask)
        # torch.stack(local_neg_video_feat): [neg_sample, bsz, lv, d]
        # local_neg_video_feat: [bsz, neg_sample, lv, d]
        local_neg_video_feat = torch.stack(local_neg_video_feat).permute(
            1, 0, 2, 3)
        local_neg_sub_feat = torch.stack(local_neg_sub_feat).permute(
            1, 0, 2, 3)
        local_neg_video_mask = torch.stack(local_neg_video_mask).permute(
            1, 0, 2)
        local_neg_sub_mask = torch.stack(local_neg_sub_mask).permute(1, 0, 2)
        features = {
            "neg_video_feat":
            local_neg_video_feat,
            "neg_sub_feat":
            local_neg_sub_feat,
            "neg_video_mask":
            local_neg_video_mask,
            "neg_sub_mask":
            local_neg_sub_mask,
            "video_feat_1":video_feat,
            "video_feat_1_mask":video_mask,
            "sub_feat_1":sub_feat,
            "sub_feat_1_mask":sub_mask,
        }

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
        loss_st_ed, intra_loss = self.get_svmr_loss(st_ed_indices, word_feat, word_mask, features["video_feat_1"], features["sub_feat_1"], \
                                        features["neg_video_feat"], features["neg_sub_feat"], \
                                        features["video_feat_1_mask"], features["sub_feat_1_mask"], \
                                        features["neg_video_mask"], features["neg_sub_mask"], span_mask, weight_hard)
        # intra_loss = 0
        loss = loss_st_ed + intra_loss
        return loss, {
            "loss_st_ed": float(loss_st_ed),
            "intra_loss": float(intra_loss),
            "loss_overall": float(loss),
        }

    def svmr_infer(self, gt_idx, word_feat, word_mask, \
                   video_feat_1, video_feat_1_mask, \
                    sub_feat_1, sub_feat_1_mask):
        _sorted_q2c_indices = gt_idx
        tmp_st_prob = []
        tmp_ed_prob = []
        max_triplet_videos = 1  # SVMR only 1
        #for one_query_feat in query_feat:
        for one_query_feat, one_query_mask, topk_q2c_indice, in zip(
                word_feat,
                word_mask,
                _sorted_q2c_indices,
        ):
            topk_video_feat = video_feat_1[topk_q2c_indice].repeat(1, 1, 1)
            topk_video_mask = video_feat_1_mask[topk_q2c_indice].repeat(1, 1)
            topk_sub_feat = sub_feat_1[topk_q2c_indice].repeat(1, 1, 1)
            topk_sub_mask = sub_feat_1_mask[topk_q2c_indice].repeat(1, 1)
            topk_query_feat = one_query_feat.repeat(max_triplet_videos, 1, 1)
            topk_query_mask = one_query_mask.repeat(max_triplet_videos, 1)

            # visual_emb, sub_emb = self.encode_query_word(topk_query_feat, topk_query_mask)
            # visual_emb = visual_emb.unsqueeze(1)
            # sub_emb = sub_emb.unsqueeze(1)
            # pos_video_feat_pro = self.visual_pro(topk_video_feat)
            # pos_sub_feat_pro = self.sub_pro(topk_sub_feat)
            # pos_video_feat_tmp = torch.mul(F.normalize(torch.mul(pos_video_feat_pro, visual_emb), p=2, dim=-1), topk_video_feat)
            # pos_sub_feat_tmp = torch.mul(F.normalize(torch.mul(pos_sub_feat_pro, sub_emb), p=2, dim=-1), topk_sub_feat)
            # pos_clip_feat = torch.cat([pos_video_feat_tmp, pos_sub_feat_tmp], dim=2)
            # pos_clip_feat = self.fusion_pro(pos_clip_feat)
            # pos_clip_feat, _ = self.self(pos_clip_feat, topk_video_mask, self.V2ctx_token_pos_embed)
            # pos_clip_feat = self.cross(pos_clip_feat, topk_video_mask, topk_query_feat, topk_query_mask)
            # one_st_prob = self.st_begin(pos_clip_feat, topk_video_mask)
            # one_ed_prob = self.ed_end(pos_clip_feat, topk_video_mask)
            _, one_st_prob, one_ed_prob = self.bidirect(
                topk_query_feat, topk_video_feat, topk_sub_feat,
                topk_video_mask, topk_query_mask)
            # one_st_prob = one_st_prob.unsqueeze(0)
            # one_ed_prob = one_ed_prob.unsqueeze(0)
            tmp_st_prob.append(one_st_prob)
            tmp_ed_prob.append(one_ed_prob)
        st_prob = torch.cat(tmp_st_prob, dim=0)
        ed_prob = torch.cat(tmp_ed_prob, dim=0)

        return st_prob, ed_prob

    def compute_context_info(self, model, eval_dataset, opt):
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
        video_feat_1_mask = []
        sub_feat1 = []
        sub_feat_1_mask = []

        for idx, batch in tqdm(enumerate(context_dataloader),
                               desc="Computing query2video scores",
                               total=len(context_dataloader)):
            metas.extend(batch[0])
            model_inputs = prepare_batch_inputs(batch[1],
                                                device=opt.device,
                                                non_blocking=opt.pin_memory)
            device = model_inputs["video_feat"].device

            feature = model.encode_context(
                model_inputs["video_feat"],
                model_inputs["video_mask"],
                model_inputs["sub_feat"],
                model_inputs["sub_mask"],
            )
            _video_feat1 = feature["video_feat_1"]
            _video_feat1_mask = feature["video_feat_1_mask"]
            _sub_feat1 = feature["sub_feat_1"]
            _sub_feat1_mask = feature["sub_feat_1_mask"]
            video_feat1.append(_video_feat1)
            video_feat_1_mask.append(_video_feat1_mask)
            sub_feat1.append(_sub_feat1)
            sub_feat_1_mask.append(_sub_feat1_mask)

        def cat_tensor(tensor_list):
            if len(tensor_list) == 0:
                return None
            else:
                seq_l = [e.shape[1] for e in tensor_list]
                b_sizes = [e.shape[0] for e in tensor_list]
                b_sizes_cumsum = np.cumsum([0] + b_sizes)
                if len(tensor_list[0].shape) == 3:
                    hsz = tensor_list[0].shape[2]
                    res_tensor = tensor_list[0].new_zeros(
                        sum(b_sizes), max(seq_l), hsz)
                elif len(tensor_list[0].shape) == 2:
                    res_tensor = tensor_list[0].new_zeros(
                        sum(b_sizes), max(seq_l))
                else:
                    raise ValueError("Only support 2/3 dimensional tensors")
                for i, e in enumerate(tensor_list):
                    res_tensor[
                        b_sizes_cumsum[i]:b_sizes_cumsum[i + 1], :seq_l[i]] = e
                return res_tensor

        return dict(
            video_metas=metas,
            video_feat_1=cat_tensor(video_feat1),
            video_feat_1_mask=cat_tensor(video_feat_1_mask),
            sub_feat_1=cat_tensor(sub_feat1),
            sub_feat_1_mask=cat_tensor(sub_feat_1_mask),
        )

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
        (video_feat, sub_feat), self.video_input_proj, self.sub_input_proj = self.videoEncoder(
                 video_feat, video_mask, sub_feat, sub_mask,
                 self.ctx_token_pos_embed)

        _, ml = video_mask.size()
        # video_feat: ([bsz, ml, d], [bsz, 64, d], [bsz, 32, d])
        # sub_feat: ([bsz, ml, d], [bsz, 64, d], [bsz, 32, d])
        # video_feat, video_mask = self.videolocal(video_feat, video_mask,
        #                                          self.V1ctx_token_pos_embed)
        # sub_feat, sub_mask = self.subglobal(sub_feat, sub_mask,
        #                                     self.V2ctx_token_pos_embed)

        features = {
            "video_feat_1":video_feat,
            "video_feat_1_mask":video_mask,
            "sub_feat_1":sub_feat,
            "sub_feat_1_mask":sub_mask,
        }
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
            video_sub_q2ctx_scores_word = self.get_video_level_scores(
                video_sub_query[0], VR_video_sub_feat,
                VR_video_sub_feat_mask) if self.use_sub else 0
            # se_v_feats: [bsz, N, qdim] se_v_attw: [bsz, N, L_word]
            fine_grain_video_scores, _, _ = self.cal_video_score(
                word_feat, video_feat_1, word_mask, video_feat_1_mask)
            fine_grain_sub_scores, _, _ = self.cal_sub_score(
                word_feat, sub_feat_1, word_mask, sub_feat_1_mask)

            loss_vid_kl_cross = self.kl(fine_grain_video_scores,
                                        video_sub_q2ctx_scores_word)
            loss_sub_kl_cross = self.kl(fine_grain_sub_scores,
                                        video_sub_q2ctx_scores_word)
            loss_vid_kl = self.kl(video_q2ctx_scores_word,
                                  fine_grain_video_scores)
            loss_sub_kl = self.kl(sub_q2ctx_scores_word, fine_grain_sub_scores)
            # 自定义系数
            loss_vid_kl *= 1e3
            loss_sub_kl *= 1e3
            loss_vid_kl_cross *= 1e3
            loss_sub_kl_cross *= 1e3
            if not is_eval:
                v_query_diverse_loss, v2q_v_ce_loss = self.share_norm_loss.query_diverse_loss(
                    video_q2ctx_scores_word, video_query_word)
                s_query_diverse_loss, v2q_s_ce_loss = self.share_norm_loss.query_diverse_loss(
                    sub_q2ctx_scores_word, sub_query_word)
                v_s_query_diverse_loss, v2q_vs_ce_loss = self.share_norm_loss.query_diverse_loss(
                    video_sub_q2ctx_scores_word, video_sub_query[0])
                query_diverse_loss = v_query_diverse_loss + s_query_diverse_loss + v_s_query_diverse_loss
                v2q_ce_loss = (v2q_v_ce_loss + v2q_s_ce_loss +
                               v2q_vs_ce_loss) / 3
            metric_weight = F.softmax(self.metric_weight, dim=0)
            q2ctx_scores = \
            metric_weight[0] * (video_q2ctx_scores_word + sub_q2ctx_scores_word + video_sub_q2ctx_scores_word) + metric_weight[1] * (fine_grain_video_scores) + metric_weight[2] * (fine_grain_sub_scores)  # (N, N) # (100_bsz, 2179) in inference stage
            q2ctx_scores_eval = (video_q2ctx_scores_word +
                                 sub_q2ctx_scores_word +
                                 video_sub_q2ctx_scores_word) / (divisor + 1)

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
                    conquer_feat, one_st_prob, one_ed_prob = self.bidirect(
                        topk_query_feat, topk_video_feat, topk_sub_feat,
                        topk_video_mask, topk_query_mask)
                    one_st_prob = one_st_prob.unsqueeze(0)
                    one_ed_prob = one_ed_prob.unsqueeze(0)
                    tmp_video_feat2 = conquer_feat
                    tmp_sub_feat2 = conquer_feat
                    stage2_score = self.vs_score(conquer_feat)
                    # 如果是SVMR就不需要2stage影响1stage
                    if gt_idx == None:
                        tmp = q2ctx_scores_eval.new_ones(
                            q2ctx_scores_eval.shape[1])
                        stage2_score_indices = torch.nonzero(
                            stage2_score > 0).squeeze()
                        # 使用索引找到对应的值
                        stage2_score_max_k = stage2_score[stage2_score_indices]
                        # stage2_score_max_k, stage2_score_indices = torch.topk(stage2_score, k=10, largest=True)
                        tmp[topk_q2c_indice[
                            stage2_score_indices]] = stage2_score_max_k
                        max_neg_score = torch.max(stage2_score)
                        neg_tmp = q2ctx_scores_eval.new_ones(
                            q2ctx_scores_eval.shape[1])
                        # stage2_score_clone = stage2_score.clone()
                        # neg_tmp[topk_q2c_indice] = stage2_score_clone
                        sigma = 45
                        neg_tmp[topk_q2c_indice] = neg_tmp[
                            topk_q2c_indice] + torch.exp(
                                -(stage2_score - max_neg_score)**2 / sigma)
                        neg_tmp[topk_q2c_indice[stage2_score_indices]] = 1
                        # k = 10
                        # topk_indice = topk_q2c_indice[:k]
                        # tmp[topk_indice] = stage2_score[:k]
                        # 只筛选得分 > 0 的 stage2_score
                        stage2_score = tmp
                        q2ctx_scores_eval[query_iter, :] = q2ctx_scores_eval[
                            query_iter, :] * stage2_score * neg_tmp
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
                sample_video_feat_1, sample_sub_feat_1, sample_video_feat_1_mask, stage1_score = self.share_norm_loss.sample_hard(
                    q2ctx_scores, video_feat_1, sub_feat_1, video_feat_1_mask)
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
                    tmp_feat, tmp_st_prob, tmp_ed_prob = self.bidirect(
                        word_feat, sample_video_feat_1[:, i],
                        sample_sub_feat_1[:, i],
                        sample_video_feat_1_mask[:, i], word_mask)
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
                st_prob = st_prob.permute(1, 0, 2).contiguous().view(qbsz, -1)
                ed_prob = ed_prob.permute(1, 0, 2).contiguous().view(qbsz, -1)

                sample_video_feat_1 = torch.stack(_sample_video_feat_1)
                sample_sub_feat_1 = torch.stack(_sample_sub_feat_1)
                sample_video_feat_1 = sample_video_feat_1.permute(1, 0, 2, 3)
                sample_sub_feat_1 = sample_sub_feat_1.permute(1, 0, 2, 3)
                # stage2_score_visual: [qbsz, sample + 1]
                stage2_score = torch.stack(_stage2_score)
                stage2_score = stage2_score.permute(1, 0)
                # consistency_loss
                consistency_loss = self.consistency_loss(
                    stage1_score, stage2_score)
                # vs_loss
                vs_loss, triplet_loss = self.vs_score.video_score_loss(
                    stage2_score, measure="cross_entropy")

                # 修改: 这部分为了代码完整性，暂时不变
                video_feat2 = video_feat_1
                sub_feat2 = sub_feat_1
                # hard Frame-CL
                # loss_fcl_vq = hard_video_query_loss(sample_video_feat2, video_query_word2, span_mask, sample_video_feat_1_mask, measure='JSD')
                # loss_fcl_sq = hard_video_query_loss(sample_sub_feat2, sub_query_word2, span_mask, sample_sub_feat_1_mask, measure='JSD')
                # loss_fcl = ((loss_fcl_vq + loss_fcl_sq) / 2) * 0.01
            if self.config.merge_two_stream and self.use_video and self.use_sub:
                fine_grain_st_prob, fine_grain_ed_prob = self.fine_grain_svmr_train(
                    word_feat, video_feat_1, sub_feat_1, word_mask,
                    video_feat_1_mask, sub_feat_1_mask)

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

    def __init__(self, config, input_dim, hidden_dim, with_emb=True):
        super(OneModalEncoder, self).__init__()
        self.linear = LinearLayer(in_hsz=input_dim, out_hsz=hidden_dim)
        self.with_emb = with_emb
        self.transformer = TransformerBaseModel(config, self.with_emb)
    def forward(self, features, attention_mask, query_token_pos_embed):

        transformed_features = self.linear(features)

        #import pdb
        #pdb.set_trace()
        # 被我修改成 0 (visual)
        token_type_ids, position_ids = query_token_pos_embed(
            transformed_features, 0)

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
