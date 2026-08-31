import math
import torch
from torch.nn import functional as F
from torch import nn
from scipy.stats import norm
import easydict
from edg.modules.transformer.bert import BertEncoder
try:
    import apex.normalization.fused_layer_norm.FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    BertLayerNorm = torch.nn.LayerNorm
from edg.modules.attention import *
def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e10)





class KL(nn.Module):
    def __init__(self, ):
        super(KL, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        kl = F.kl_div(logpt0, logpt1, reduction='mean')
        return kl





class ShareNormLoss(nn.Module):

    def __init__(self, ):
        super(ShareNormLoss, self).__init__()
        self.nllloss = nn.NLLLoss(reduction="mean")
        self.v2q_ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.moment_ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, query_context_scores, st_prob, ed_prob, gt_indices,
                mask):
        """Negative log-likelihood loss.

        Args:
            st/ed_prob (tensor): probabilities produced by the 1D convolution [vbsz, lv]
            gt_indices (tensor): ground truth
            mask (tensor): validity mask
        """
        hard_negative_indices = self.hard_sample(query_context_scores)
        hard_st, hard_ed = self.hard_similarity_get(hard_negative_indices,
                                                    st_prob, ed_prob)
        indices = torch.arange(0, st_prob.shape[0]).to(st_prob.device)
        gt_st = gt_indices[:, 0]
        gt_st_conv = st_prob[indices, gt_st]
        gt_ed = gt_indices[:, 1]
        gt_ed_conv = ed_prob[indices, gt_ed]
        loss_st, loss_ed = self.share_normalization_loss(
            gt_st_conv, gt_ed_conv, hard_st, hard_ed)
        return loss_st, loss_ed

    def moment_share_loss(self, st_prob, ed_prob, gt_indices, mask, weight_hard):
        """Compute shared normalization.

        Args:
            st_prob (tensor): [qbsz, sample_num * lv] the first sample is the ground truth
            ed_prob (tensor): [qbsz, sample_num * lv]
            gt_indices (tensor): [qbsz, 2]
            mask (tensor): [qbsz, sample_num, lv]
        Return:
            loss_st
            loss_ed
        """


        qbsz= st_prob.shape[0]
        gt_st = gt_indices[:, 0]
        gt_ed = gt_indices[:, 1]
        indices = torch.arange(qbsz).to(device=st_prob.device)
        gt_st_conv_prob = st_prob[indices, gt_st]
        gt_ed_conv_prob = ed_prob[indices, gt_ed]
        _st_prob = torch.where(st_prob < 0, torch.exp(st_prob), st_prob)
        _ed_prob = torch.where(ed_prob < 0, torch.exp(ed_prob), ed_prob)
        st_denominator = _st_prob.sum(dim=1)
        ed_denominator = _ed_prob.sum(dim=1)
        st_molecule = torch.exp(gt_st_conv_prob)
        ed_molecule = torch.exp(gt_ed_conv_prob)
        weight_hard = 1
        loss_st = torch.sum((-torch.log(weight_hard * (st_molecule / (st_denominator + st_molecule))))) / qbsz
        loss_ed = torch.sum((-torch.log(weight_hard * (ed_molecule / (ed_denominator + ed_molecule))))) / qbsz

        return loss_st, loss_ed

    def moment_intra_cl_loss(self, span_mask, st_prob, ed_prob, pos_mask, ):
        """solve intra clip loss

        Args:
            span_mask (tensor): [bsz, less than lv]
            st_prob (tensor): [bsz, (sample_num+ 1) * lv]
            ed_prob (tensor): [bsz, (sample_num+ 1) * lv]
            gt_indices (tensor): [bsz, 2]
            pos_mask (tensor): [bsz, (sample_num+ 1) * lv]]
        """
        span_mask = F.pad(span_mask, (0, pos_mask.shape[1] - span_mask.shape[1]))
        bsz = st_prob.shape[0]
        pos_st_prob = torch.exp(st_prob[:, :pos_mask.shape[1]])
        pos_ed_prob = torch.exp(ed_prob[:, :pos_mask.shape[1]])
        pos_st_prob_moment = (pos_st_prob * span_mask * pos_mask).sum(dim=1)
        pos_ed_prob_moment = (pos_ed_prob * span_mask * pos_mask).sum(dim=1)
        neg_st_prob_moment = (pos_st_prob * (1 - span_mask) * pos_mask).sum(dim=1)
        neg_ed_prob_moment = (pos_ed_prob * (1 - span_mask) * pos_mask).sum(dim=1)
        neg_st_prob_share = torch.exp(st_prob[:, pos_mask.shape[1]:]).sum(dim=1)
        neg_ed_prob_share = torch.exp(ed_prob[:, pos_mask.shape[1]:]).sum(dim=1)
        margin = 0
        intra_loss_st = torch.sum(torch.clamp(margin + neg_st_prob_share + neg_st_prob_moment - pos_st_prob_moment, min=0))
        intra_loss_ed = torch.sum(torch.clamp(margin + neg_ed_prob_share + neg_ed_prob_moment - pos_ed_prob_moment, min=0))
        intra_loss = (intra_loss_st + intra_loss_ed) / (span_mask.shape[1] * 5)
        intra_lr = 1
        intra_loss = intra_lr * intra_loss
        return intra_loss



    def generate_gaussion(self, gt_indices, mask):
        """Generate a Gaussian target distribution.

        Args:
            gt_indices (tensor): [vbsz, 2]
            mask (tensor): [vbsz, lv]
        Return:
            gaussian_st, gaussian_ed
        """
        gt_st = gt_indices[:, 0]
        gt_ed = gt_indices[:, 1]
        gt_ed = torch.where(gt_st == gt_ed, gt_ed + 1, gt_ed)
        valid_len = mask.sum(dim=1)
        sigma = ((gt_ed - gt_st) / valid_len) * 5
        mu1 = gt_st
        mu2 = gt_ed
        gaussian_st = torch.from_numpy(
            norm.pdf(
                torch.arange(mask.shape[1]).unsqueeze(0).repeat(
                    mu1.shape[0], 1).to("cpu"),
                mu1.unsqueeze(1).to("cpu"),
                sigma.unsqueeze(1).to("cpu"))).to(device=mask.device)
        gaussian_ed = torch.from_numpy(
            norm.pdf(
                torch.arange(mask.shape[1]).unsqueeze(0).repeat(
                    mu2.shape[0], 1).to("cpu"),
                mu2.unsqueeze(1).to("cpu"),
                sigma.unsqueeze(1).to("cpu"))).to(device=mask.device)
        return gaussian_st, gaussian_ed

    def moment_gaussian_loss(self, st_prob, ed_prob, gt_indices, mask):
        gaussian_st, gaussian_ed = self.generate_gaussion(gt_indices, mask)
        pad_num = st_prob.min()
        qbsz = st_prob.shape[0]
        indices = torch.arange(qbsz).to(device=st_prob.device)
        lv = mask.shape[1]
        gt_st_conv_prob = st_prob[:, :lv]
        gt_ed_conv_prob = ed_prob[:, :lv]
        gt_st_conv_prob = gt_st_conv_prob * gaussian_st
        gt_ed_conv_prob = gt_ed_conv_prob * gaussian_ed
        gt_st_conv_prob = gt_st_conv_prob.masked_fill(~mask.bool(), pad_num)
        gt_ed_conv_prob = gt_ed_conv_prob.masked_fill(~mask.bool(), pad_num)

        return gt_st_conv_prob, gt_ed_conv_prob

    def loss(self, st_prob, ed_prob, gt_indices, mask):
        """Compute shared normalization.

        Args:
            st_prob (tensor): [qbsz, sample_num * lv] the first sample is the ground truth
            ed_prob (tensor): [qbsz, sample_num * lv]
            gt_indices (tensor): [qbsz, 2]
            mask (tensor): [qbsz, sample_num, lv]
        Return:
            loss_st
            loss_ed
        """
        qbsz, sample_num, lv = mask.shape
        gt_st = gt_indices[:, 0]
        gt_ed = gt_indices[:, 1]
        indices = torch.arange(qbsz).to(device=mask.device)
        gt_st_conv_prob = st_prob[indices, gt_st]
        gt_ed_conv_prob = ed_prob[indices, gt_ed]
        st_denominator = torch.exp(st_prob).sum(dim=1)
        ed_denominator = torch.exp(ed_prob).sum(dim=1)
        st_molecule = torch.exp(gt_st_conv_prob)
        ed_molecule = torch.exp(gt_ed_conv_prob)
        loss_st = torch.sum(-torch.log(st_molecule / (st_denominator + 1e-8)))
        loss_ed = torch.sum(-torch.log(ed_molecule / (ed_denominator + 1e-8)))
        return loss_st, loss_ed

    def batch_prob(self, prob, mask):
        """
        Start/end probabilities within a batch.
        paramerter:
            prob: [vbsz, lv]
            mask: [vbsz, lv]
        """
        batch_prob = prob.sum()
        return batch_prob

    def hard_sample(self, similarity, sample_num=4):
        """Hard Negative Mining

        Args:
            similarity (tensor): retrieval matrix produced during training [qbsz, vbsz]
        Return:
            hard_similarity: hard-negative samples [sample_num, max_ctx]
        """
        hard_negative_indices = similarity.topk(sample_num,
                                                dim=1,
                                                largest=True)[1]
        return hard_negative_indices

    def query_diverse_loss(self,
                           modality_similarity,
                           modality_query,
                           sample_num=4,
                           delta=0.15,
                           alpha=10):
        diag_score = torch.diag(modality_similarity).unsqueeze(1)
        modality_similarity = modality_similarity - torch.eye(
            modality_similarity.shape[0]).to(modality_similarity.device) * 9999
        hard_query_score, _ = modality_similarity.topk(sample_num,
                                                       dim=0,
                                                       largest=True)
        _, hard_query_indices = modality_similarity.topk(sample_num,
                                                         dim=0,
                                                         largest=False)
        hard_query_indices = hard_query_indices.permute(1, 0)
        hard_query_score = hard_query_score.permute(1, 0)
        indices = torch.arange(modality_similarity.shape[0]).to(
            device=modality_similarity.device).unsqueeze(1)
        pos_hard_query_indices = torch.cat([indices, hard_query_indices],
                                           dim=1)
        pos_hard_query_score = torch.cat([diag_score, hard_query_score], dim=1)
        gt_idx = modality_similarity.new_zeros(
            modality_similarity.shape[0]).long()
        v2q_ce_loss = self.v2q_ce_loss(pos_hard_query_score, gt_idx)
        v2q_ce_loss = 0.1 * v2q_ce_loss
        bsz = modality_similarity.shape[0]
        hard_modality_query = []
        for i in range(bsz):
            _hard_modality_query = modality_query[pos_hard_query_indices[i]]
            hard_modality_query.append(_hard_modality_query)
        hard_modality_query = torch.stack(hard_modality_query)
        norm = torch.norm(hard_modality_query, dim=-1, keepdim=True)
        normalized_query = hard_modality_query / norm
        query_sim = torch.einsum("vsd, vld -> vsl", normalized_query,
                                 normalized_query)
        query_sim = query_sim - torch.eye(
            query_sim.shape[1]).unsqueeze(0).repeat(query_sim.shape[0], 1,
                                                    1).to(query_sim.device)
        diverse_loss = torch.log(1 + torch.exp(alpha * (query_sim + delta))
                                 ).mean()  # .mean()/ vbsz * sample
        diverse_loss = 0.05 * diverse_loss
        return diverse_loss, v2q_ce_loss

    def sample_hard(self,
                    similarity,
                    context1_feat,
                    context2_feat,
                    mask,
                    sample_num=4):
        """Hard Negative Mining

        Args:
            similarity (tensor): retrieval matrix produced during training [qbsz, vbsz]
        Return:
            hard_similarity: hard-negative samples [sample_num, max_ctx]
        """
        diag_score = torch.diag(similarity).unsqueeze(1)
        similarity = similarity - torch.eye(similarity.shape[0]).to(
            similarity.device) * 9999
        hard_negative_score, hard_negative_indices = similarity.topk(
            sample_num, dim=1, largest=True)

        pos_indices = torch.arange(
            similarity.shape[0]).unsqueeze(1).to(device=similarity.device)
        hard_negative_score = torch.cat((diag_score, hard_negative_score),
                                        dim=1)
        hard_negative_indices = torch.cat((pos_indices, hard_negative_indices),
                                          dim=1)
        _context1_feat = []
        _context2_feat = []
        _mask = []
        for i in range(hard_negative_indices.shape[0]):
            tmp_feat1 = context1_feat[hard_negative_indices[i]]
            tmp_feat2 = context2_feat[hard_negative_indices[i]]
            tmp_mask = mask[hard_negative_indices[i]]
            _context1_feat.append(tmp_feat1)
            _context2_feat.append(tmp_feat2)
            _mask.append(tmp_mask)

        _context1_feat = torch.stack(_context1_feat)
        _context2_feat = torch.stack(_context2_feat)
        _mask = torch.stack(_mask)
        qbsz, sample_num, lv, video_dim = _context1_feat.shape

        return _context1_feat, _context2_feat, _mask, hard_negative_score

    def hard_similarity_get(self, hard_negative_indices, st_prob, ed_prob):
        st = []
        ed = []
        for i in range(hard_negative_indices.shape[0]):
            tmp_st = st_prob[hard_negative_indices[i]]
            tmp_ed = ed_prob[hard_negative_indices[i]]
            st.append(tmp_st)
            ed.append(tmp_ed)
        st = torch.stack(st)
        ed = torch.stack(ed)
        st = st.contiguous().view(hard_negative_indices.shape[0], -1)
        ed = ed.contiguous().view(hard_negative_indices.shape[0], -1)
        return st, ed

    def share_normalization_loss(self, gt_st, gt_ed, hard_st, hard_ed):
        st_denominator = torch.exp(hard_st).sum(dim=1)
        ed_denominator = torch.exp(hard_ed).sum(dim=1)
        st_molecule = torch.exp(gt_st)
        ed_molecule = torch.exp(gt_ed)
        loss_st = torch.sum(-torch.log(st_molecule / (st_denominator + 1e-8)))
        loss_ed = torch.sum(-torch.log(ed_molecule / (ed_denominator + 1e-8)))
        return loss_st, loss_ed


class VS_score(nn.Module):
    def __init__(self,):
        super(VS_score, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.video_score_predictor = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1, bias=False)
        )
    def forward(self, video_feat,):
        video_score = self.video_score_predictor(video_feat)
        video_score = video_score.squeeze(-1)
        video_score = video_score.max(dim=1)[0]
        return video_score

    def video_score_loss(self, video_score, measure="nce"):
        bsz, sample = video_score.shape
        vs_loss = 0
        if measure == "nce":
            exp_video_score = torch.exp(video_score)
            fen_zi = exp_video_score[:, 0]
            fen_mu = exp_video_score
            vs_loss = torch.sum(-torch.log(fen_zi / fen_mu.sum(dim=1))) / bsz

        if measure == "cross_entropy":
            ce_label = video_score.new_zeros(bsz, sample)
            ce_label[:, 0] = 1
            vs_loss = self.ce_loss(video_score, ce_label)
            vs_loss = vs_loss * 0.1
        triplet_loss = 0
        pos_score = video_score[:, 0]
        random_tensor = torch.randint(1, 5, (bsz, 1)).to(video_score.device)
        neg_score = video_score.gather(1, random_tensor).squeeze(-1)
        triplet_loss = torch.clamp(neg_score - pos_score + 0.05, min=0)
        triplet_loss = triplet_loss.sum() / bsz
        triplet_loss = triplet_loss * 5
        triplet_loss = 0
        return vs_loss, triplet_loss

class ConvSE(nn.Module):
    """
        ConvSE module
    """
    def __init__(self,):
        super(ConvSE, self).__init__()
        conv_cfg_1 = {
        "in_channels": 256,
        "out_channels": 128,
        "kernel_size": 5,
        "stride": 1,
        "padding": 2,
        "bias": False
        }
        conv_cfg_2 = {
        "in_channels": 128,
        "out_channels": 1,
        "kernel_size": 1,
        "stride": 1,
        "padding": 0,
        "bias": False
        }
        self.conv_cfg_1 = easydict.EasyDict(conv_cfg_1)
        self.conv_cfg_2 = easydict.EasyDict(conv_cfg_2)
        self.clip_score_predictor = nn.Sequential(
            nn.Conv1d(**self.conv_cfg_1),
            nn.ReLU(),
            nn.Conv1d(**self.conv_cfg_2),
        )


    def forward(self, contextual_qal_features, video_mask):
        """
        Inputs:
            :param contextual_qal_features: (batch, feat_size, L_v)
            :param video_mask: (batch, L_v)
        Return:
             score: (begin or end) score distribution
        """
        score = self.clip_score_predictor(contextual_qal_features.permute(0, 2, 1)).squeeze(1) #(batch, L_v)

        score = mask_logits(score, video_mask)  #(batch, L_v)

        return score

class TwoStageConpare(nn.Module):
    def __init__(self,):
        super(TwoStageConpare, self).__init__()
        self.margin = 0.05

    def forward(self, stage1_score, stage2_score):
        consistency_loss = 0
        qbsz, sample = stage1_score.shape
        stage1_score_pos = stage1_score[:, 0]
        stage2_score_pos = stage2_score[:, 0]
        pos_score = torch.min(stage1_score_pos, stage2_score_pos)
        stage1_score_neg = stage1_score[:, 1:].sum(dim=1)
        stage2_score_neg = stage2_score[:, 1:].sum(dim=1)
        neg_score = (stage1_score_neg + stage2_score_neg)
        consistency_loss = torch.clamp(neg_score - pos_score + self.margin, min=0)
        consistency_loss = consistency_loss.sum() / qbsz
        return consistency_loss
    def rrf_convert(self, score_eval, k=9):
        score_eval_rank = torch.argsort(-score_eval, dim=1)  # Rank indices for stage1
        fused_scores = 1 / (k + score_eval_rank.float())
        return fused_scores
    def rrf_fusion(self, stage1_score, stage2_score, k=9, is_eval=False):
        """
        Rank-based score fusion.
        Perform Reciprocal Rank Fusion (RRF) on two stages of scores using GPU-efficient operations.

        Args:
            stage1_score (torch.Tensor): Scores from stage 1 with shape (bsz, 5).
            stage2_score (torch.Tensor): Scores from stage 2 with shape (bsz, 5).
            k (int): The RRF constant controlling the impact of rank (default=60).

        Returns:
            torch.Tensor: Fused scores with shape (bsz, 5).
        """
        stage1_rank = torch.argsort(-stage1_score, dim=1)  # Rank indices for stage1
        stage2_rank = torch.argsort(-stage2_score, dim=1)  # Rank indices for stage2

        rank1 = torch.argsort(stage1_rank, dim=1)  # Convert indices to ranks
        rank2 = torch.argsort(stage2_rank, dim=1)

        fused_scores = 1 / (k + rank1.float()) + 1 / (k + rank2.float())
        if is_eval:
            fused_scores = fused_scores.squeeze(0)
        return fused_scores

    def tmp(self, stage1_score, stage2_score):
        qbsz, sample = stage1_score.shape
        stage1_score = stage1_score / torch.sqrt((stage1_score * stage1_score).sum(dim=1, keepdim=True))
        stage2_score = stage2_score / torch.sqrt((stage2_score * stage2_score).sum(dim=1, keepdim=True))
        pos_visual_discrepancy = torch.abs(stage1_score - stage2_score)
        fix_indices = [1, 0, 4, 2, 3]
        tmp_stage2_score = stage2_score[:, fix_indices]
        neg_visual_discrepancy = torch.abs(stage1_score - tmp_stage2_score)
        triplet_loss = torch.clamp(pos_visual_discrepancy - neg_visual_discrepancy + self.margin, min = 0)
        triplet_loss = triplet_loss.sum() / qbsz
        return triplet_loss

class BidirectionalAttention(nn.Module):

    def __init__(self, video_dim):
        super(BidirectionalAttention, self).__init__()
        self.visual_similarity_weight = nn.Linear(video_dim * 3, 1, bias=False)
        self.query_weight = QueryWeightEncoder()
        self.visual_fc = LinearLayer(video_dim * 5, video_dim)
        bert_config_dict = {
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 256,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "max_position_embeddings": 100,
            "num_attention_heads": 8,
            "num_hidden_layers": 1,
            "type_vocab_size": 2,
            "layer_norm_eps": 1e-05,
            "output_attentions": False,
            "output_hidden_states": False
        }

        bert_config = easydict.EasyDict(bert_config_dict)
        self.contextual_QAL_feature_learning = FCPlusTransformer(config=bert_config, input_dim=video_dim * 4)
        self.visual_encoder = BertEncoder(bert_config)
        self.st_encoder = FCPlusTransformer(config=bert_config, input_dim=video_dim * 5)
        self.ed_encoder = FCPlusTransformer(config=bert_config, input_dim=video_dim * 2)
        self.begin_score_modeling = ConvSE()
        self.end_score_modeling = ConvSE()

    def forward(self, query_emb, video_feat, sub_feat, video_mask, query_mask):
        """
        Inputs:
        :param QDF_emb: (batch, L_v, feat_size)
        :param query_emb: (batch, L_q, feat_size)
        :param video_mask: (batch, L_v)
        :param query_mask: (batch, L_q)
        Return:
        QAL: (batch, L_v, feat_size*4)
        """
        QDF_visual_emb = self.query_weight(query_emb, video_feat, sub_feat)
        video_len = QDF_visual_emb.size()[1]
        query_len = query_emb.size()[1]

        _QDF_visual_emb = QDF_visual_emb.unsqueeze(2).repeat(1, 1, query_len, 1)

        _query_emb = query_emb.unsqueeze(1).repeat(1, video_len, 1, 1)

        elementwise_visual_prod = torch.mul(_QDF_visual_emb, _query_emb)
        if _query_emb.shape[0] != _QDF_visual_emb.shape[0]:
            query_emb = query_emb.repeat(_QDF_visual_emb.shape[0], 1, 1)
            _query_emb = _query_emb.repeat(_QDF_visual_emb.shape[0], 1, 1, 1)
        visual_alpha = torch.cat([_QDF_visual_emb, _query_emb, elementwise_visual_prod], dim=3)

        similarity_visual_matrix = self.visual_similarity_weight(visual_alpha).view(-1, video_len, query_len)

        similarity_matrix_mask = torch.einsum("bn,bm->bnm", video_mask, query_mask)


        a = F.softmax(mask_logits(similarity_visual_matrix,
                                  similarity_matrix_mask), dim=-1)

        visual_V2Q = torch.bmm(a, query_emb)


        b = F.softmax(torch.max(mask_logits(similarity_visual_matrix, similarity_matrix_mask), 2)[0], dim=-1)

        b = b.unsqueeze(1)

        visual_Q2V = torch.bmm(b, QDF_visual_emb)

        visual_Q2V = visual_Q2V.repeat(1, video_len, 1)


        visual_QAL = torch.cat([QDF_visual_emb, visual_V2Q,
                         torch.mul(QDF_visual_emb, visual_V2Q),
                         torch.mul(QDF_visual_emb, visual_Q2V)], dim=2)
        Contextual_QAL  = self.contextual_QAL_feature_learning(
            features=visual_QAL,
            feat_mask=video_mask)
        G = torch.cat([visual_QAL,Contextual_QAL], dim=2)
        st_feat = self.st_encoder(G, video_mask)
        ed_feat = self.ed_encoder(torch.cat([Contextual_QAL, st_feat], dim=2), video_mask)
        st_prob = self.begin_score_modeling(st_feat, video_mask)
        ed_prob = self.end_score_modeling(ed_feat, video_mask)
        visual_QAL = self.visual_fc(G)
        visual_QAL = self.visual_encoder(visual_QAL, attention_mask=video_mask)[0]
        return visual_QAL, st_prob, ed_prob


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True,tanh=False):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.tanh = tanh
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = BertLayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        if self.tanh:
            x = torch.tanh(x)
        return x  # (N, L, D)

class FCPlusTransformer(nn.Module):
    """
        FC + Transformer
        FC layer reduces input feature size into hidden size
        Transformer contextualizes QAL feature
    """

    def __init__(self, config,input_dim):
        super(FCPlusTransformer, self).__init__()
        self.trans_linear = LinearLayer(
            in_hsz=input_dim, out_hsz=config.hidden_size)
        self.encoder = BertEncoder(config)

    def forward(self,features, feat_mask):
        """
        Inputs:
            :param contextual_qal_features: (batch, L_v, input_dim)
            :param feat_mask: (batch, L_v)
        Return:
            sequence_output: (batch, L_v, hidden_size)
        """
        transformed_features = self.trans_linear(features)

        encoder_outputs = self.encoder(hidden_states=transformed_features, attention_mask=feat_mask)

        sequence_output = encoder_outputs[0]

        return sequence_output

class QueryWeightEncoder(nn.Module):
    """
        Query Weight Encoder
        Using NetVLAD to aggreate contextual query features
        Using FC + Softmax to get fusion weights for each modality
    """
    def __init__(self, video_modality=["video", "sub"]):
        super(QueryWeightEncoder, self).__init__()
        config_dict = {
                "hidden_size": 256,
                "text_cluster": 32,
                "moe_dropout_prob": 0.1
        }
        config = easydict.EasyDict(config_dict)
        self.text_pooling = NetVLAD(feature_size=config.hidden_size,cluster_size=config.text_cluster)
        self.moe_txt_dropout = nn.Dropout(config.moe_dropout_prob)

        self.moe_fc_txt = nn.Linear(
            in_features=self.text_pooling.out_dim,
            out_features=len(video_modality),
            bias=False)

        self.video_modality = video_modality

    def forward(self, query_feat, video_feat, sub_feat):
        pooled_text = self.text_pooling(query_feat)
        pooled_text = self.moe_txt_dropout(pooled_text)

        moe_weights = self.moe_fc_txt(pooled_text)
        softmax_moe_weights = F.softmax(moe_weights, dim=1)

        query_video_weight, query_sub_weight = softmax_moe_weights.split(1, dim = 1)
        query_video_weight = query_video_weight.squeeze(1)
        query_sub_weight = query_sub_weight.squeeze(1)
        final_query_context_scores = self.compute_final_score(video_feat, sub_feat, query_video_weight, query_sub_weight)
        return final_query_context_scores
    def compute_final_score(self, video_feat, sub_feat, query_video_weight, query_sub_weight):
        video_query_context_scores = torch.einsum("bld, b -> bld", video_feat, query_video_weight)
        sub_query_context_scores = torch.einsum("bld, b -> bld", sub_feat, query_sub_weight)
        final_query_context_scores = video_query_context_scores + sub_query_context_scores
        return final_query_context_scores
class NetVLAD(nn.Module):
    def __init__(self, cluster_size, feature_size, add_norm=True):
        super(NetVLAD, self).__init__()
        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.clusters = nn.Parameter((1 / math.sqrt(feature_size))
                                     * torch.randn(feature_size, cluster_size))
        self.clusters2 = nn.Parameter((1 / math.sqrt(feature_size))
                                      * torch.randn(1, feature_size, cluster_size))

        self.add_norm = add_norm
        self.LayerNorm = BertLayerNorm(cluster_size)
        self.out_dim = cluster_size * feature_size

    def forward(self, x):
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)
        assignment = torch.matmul(x, self.clusters)

        if self.add_norm:
            assignment = self.LayerNorm(assignment)

        assignment = F.softmax(assignment, dim=1)
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = torch.sum(assignment, -2, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)

        x = x.view(-1, max_sample, self.feature_size)
        vlad = torch.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        vlad = F.normalize(vlad)

        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad
