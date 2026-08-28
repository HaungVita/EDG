import math
import random
import torch
from torch.nn import functional as F
from torch import nn
from scipy.stats import norm
import numpy as np
import easydict
from .transformer.bert import BertEncoder
try:
    import apex.normalization.fused_layer_norm.FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    BertLayerNorm = torch.nn.LayerNorm
from .attention import *
def mask_logits(target, mask):
    #import pdb;pdb.set_trace()
    return target * mask + (1 - mask) * (-1e10)



class WSP(nn.Module):

    def __init__(self, hidden_size=256):
        super(WSP, self).__init__()
        self.weight = nn.Linear(hidden_size, 1)
    def forward(self, query, context, context_mask):
        return self.weight_get_video_level_scores(query, context, context_mask)

    def weight_get_video_level_scores(
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
        # weight: [vbsz, semantic, 1]
        weight = self.weight(context_feat1).squeeze(-1)  # (N, L)
        modularied_query = F.normalize(modularied_query, dim=-1)
        context_feat1 = F.normalize(context_feat1, dim=-1)
        query_context_scores = torch.einsum("md,nld->mln", modularied_query,
                                            context_feat1)  # (N, L, N)
        query_context_scores = torch.einsum("mln,nl->mln", query_context_scores,
                                            weight)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        #import pdb
        #pdb.set_trace()
        query_context_scores = mask_logits(query_context_scores,
                                            context_mask)  # (N, L, N)
        query_context_scores, _ = torch.max(
            query_context_scores,
            dim=1)  # (N, N) diagonal positions are positive pairs.
        return query_context_scores

class HGR(nn.Module):

    def __init__(self):
        super().__init__()
        self.word_proj = nn.Linear(256, 256)
        self.context_proj = nn.Conv1d(256, 256, kernel_size=3, padding=1)
    def forward(self, phrase_embeds, vid_embeds, phrase_masks, vid_masks, is_eval=False):
        phrase_embeds = self.word_proj(phrase_embeds)
        vid_embeds = self.context_proj(vid_embeds.permute(0, 2, 1)).permute(0, 2, 1)
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
        # 如果是 infer 阶段，直接返回similarity-per-context-word, 注意这里我们用ground_sims形状修改成(batch_phrases, batch_vids, num_phrases, num_frames)
        if is_eval == True:
            ground_sims = ground_sims.permute(1, 0, 3, 2)
            ground_sims = torch.einsum('abtv,at->abtv', [ground_sims, phrase_masks])
            ground_sims = torch.einsum('abtv,bv->abtv', [ground_sims, vid_masks])
            return ground_sims
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

    def cosine_sim(self, im, s):
        inner_prod = im.mm(s.t())
        im_norm = torch.sqrt((im**2).sum(1).view(-1, 1) + 1e-18)
        s_norm = torch.sqrt((s**2).sum(1).view(1, -1) + 1e-18)
        sim = inner_prod / (im_norm * s_norm)
        return sim

    def h2v_l2norm(self, inputs, dim=-1):
        # inputs: (batch, dim_ft)
        norm = torch.norm(inputs, p=2, dim=dim, keepdim=True)
        inputs = inputs / norm.clamp(min=1e-10)
        return inputs

class KL(nn.Module):
    def __init__(self, ):
        super(KL, self).__init__()

    def forward(self, sim_matrix0, sim_matrix1):
        logpt0 = F.log_softmax(sim_matrix0, dim=-1)
        logpt1 = F.softmax(sim_matrix1, dim=-1)
        kl = F.kl_div(logpt0, logpt1, reduction='mean')
        return kl

class Gaussian_weight(nn.Module):
    def __init__(self, sigma=0.4,):
        super(Gaussian_weight, self).__init__()
        self.sigma = sigma  # sigma是一个超参数, 论文中设为0.4在 ActivityNet Captions 表现最佳
    def forward(self, feat_mask, center_indeices):
        # 下标映射到[-1, 1]区间
        samples, gt_indices = self.index_transform(feat_mask=feat_mask, gt_indices=center_indeices)
        # 计算不同的 sample 的 mu 生成高斯分布进行采样
        gaussian_st, gaussian_ed = self.sample_weight(samples, gt_indices, feat_mask)
        return gaussian_st, gaussian_ed
    def sample_weight(self, samples, center_indeices, feat_mask):
        """采样高斯分布

        Args:
            sample (tensor): 采样点 [nums, max_ctx]
            center_index (tensor): gt采样点 [nums, 2] st, ed
        Returns:
            gaussian_weight: 高斯权重
        """
        device = samples.device
        gaussian_weight_st = []
        gaussian_weight_ed = []
        binary_weight = []
        samples = samples.cpu()
        center_indeices = center_indeices.cpu()
        for sample, center_index, mask in zip(samples, center_indeices, feat_mask):
            st_index = center_index[0]
            ed_index = center_index[1]
            # 计算 mu
            mu_st = st_index
            mu_ed = ed_index
            # mu_st = sample[st_index]
            # mu_ed = sample[ed_index]
            # 计算高斯分布的PDF
            pdf_st = norm.pdf(sample, mu_st, self.sigma)
            pdf_ed = norm.pdf(sample, mu_ed, self.sigma)
            pdf_st = pdf_st * mask.cpu().numpy()
            pdf_ed = pdf_ed * mask.cpu().numpy()
            pdf= pdf_st + pdf_ed
            gaussian_weight_st.append(pdf_st)
            gaussian_weight_ed.append(pdf_ed)
            binary_weight.append(pdf)
            # 以下用于可视化检查采样分布
            # import matplotlib.pyplot as plt
            # 绘制高斯分布的PDF
            # plt.figure(figsize=(10, 6))
            # plt.plot(sample, pdf_st, label=f'Sample mu_st:{mu_st}')
            # plt.plot(sample, pdf_ed, label=f'Sample mu_ed:{mu_ed}')
            # plt.plot(sample, pdf, label=f'Sample mu_st+mu_ed:{mu_st + mu_ed}')
            # plt.xlabel('x')
            # plt.ylabel('Probability Density')
            # plt.title('Gaussian Probability Density Function')
            # plt.legend()
            # plt.grid(True)
            # plt.show()
        gaussian_weight_st = torch.from_numpy(np.stack(gaussian_weight_st)).to(device=device)
        gaussian_weight_ed = torch.from_numpy(np.stack(gaussian_weight_ed)).to(device=device)
        # binary_weight = torch.from_numpy(np.stack(binary_weight)).to(device=device)
        return gaussian_weight_st, gaussian_weight_ed
    def index_transform(self, feat_mask, gt_indices=False):
        """下标映射到[-1, 1]
        Args:
            feat_mask (tensor): 下标 [vbsz, max_ctx]
        Returns:
            index_trans: 映射后的下标
            已测试
        """
        # 形状、序列长度
        nums, max_ctx = feat_mask.shape
        ctx = feat_mask.sum(dim=1)
        # 下标 e.g [1 ,2 ,3 ...]
        index = torch.arange(1, max_ctx+1).to(feat_mask.device).unsqueeze(0).repeat(nums, 1)
        index_trans = (index - 1) * 2 / (ctx - 1).unsqueeze(1) - 1
        index_trans = index_trans * feat_mask
        if isinstance(gt_indices, torch.Tensor):
            gt_indices = (gt_indices - 1) * 2 / (ctx - 1).unsqueeze(1) - 1
        return index_trans, gt_indices

class RankInContext(nn.Module):

    def __init__(self):
        super(RankInContext, self).__init__()
        self.norm_pdf = VGaussian_weight()
        self.cross_entropy = nn.CrossEntropyLoss(reduce=False)

    def forward(self, cluster_idx, gt_idx, prob, feat_mask=None, token_idx=None):
        """获取正样本锚点

        Args:
            cluster_idx (tensor): 指明聚类中心所属 clip 下标 [n_v, num_cluster]
            token_idx (tensor): 每个 clip 下标对应的类别 [n_v, lv]
            gt_idx (tensor):  gt 的 st, ed [vbsz, 2]
            prob (tensor): SVMR 模型输出的概率 [vbsz, lv]
        Returns:
            pos_anchor_indices: 正样本锚点索引
        """
        n_v, num_cluster = cluster_idx.shape
        # pos_mask: [n_v, num_cluster], neg_mask: [n_v, num_cluster]
        pos_mask, neg_mask = self.pos_neg_indices(cluster_idx, gt_idx)
        # pos_cluster_clip = cluster_idx.masked_fill(neg_mask, -1)
        # neg_cluster_clip = cluster_idx.masked_fill(pos_mask, -1)
        # 生成 25 个 mu 为 cluster center 的高斯分布 gaussion_weight: [n_v, num_cluster, max_ctx]
        gaussion_weight = self.norm_pdf(feat_mask, cluster_idx, gt_idx)
        pos_gaussion_weight = gaussion_weight.masked_fill(neg_mask.unsqueeze(2), 0)
        neg_gaussion_weight = gaussion_weight.masked_fill(pos_mask.unsqueeze(2), 0)
        # loss_p 存储每个正样本的 loss [n_v, ]
        loss_p = torch.zeros(pos_mask.shape[0]).to(pos_mask.device)
        # loss_n 存储每个负样本的 loss [n_v, ]
        loss_n = torch.zeros(neg_mask.shape[0]).to(neg_mask.device)
        for cluster_i in range(gaussion_weight.shape[1]):
            # loss_zeros = self.cross_entropy(prob, torch.zeros_like(pos_gaussion_weight)[:, cluster_i])
            # mini-batch 内每个正样本的 loss [n_v, ]
            # TODO: 真实情况下prob有 mask
            _loss_p = self.cross_entropy(prob, pos_gaussion_weight[:, cluster_i])
            # mini-batch 内每个负样本的 loss [n_v, ]
            # TODO: 真实情况下prob有 mask
            _loss_n = self.cross_entropy(prob, neg_gaussion_weight[:, cluster_i])
            loss_p = loss_p + _loss_p
            loss_n = loss_n + _loss_n
        # pos_num: 标量
        pos_num = pos_mask.sum(dim=1)
        neg_num = neg_mask.sum(dim=1)
        eps = 1e-6
        loss_p = loss_p / (pos_num + eps)
        loss_n = loss_n / (neg_num + eps)
        loss_rank = self.get_ranking_loss(loss_p, loss_n)
        return loss_rank

    def get_ranking_loss(self, pos_score, neg_score, method="hinge"):
        """ Note here we encourage positive scores to be larger than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        # 0-1 归一化 pos_score
        # pos_min = pos_score.min()
        # pos_max = pos_score.max()
        # pos_score = (pos_score - pos_min) / (pos_max - pos_min)
        # neg_min = neg_score.min()
        # neg_max = neg_score.max()
        # neg_score = (neg_score - neg_min) / (neg_max - neg_min)
        if method == "hinge":  # max(0, m + S_neg - S_pos)
            return torch.clamp(0.1 + neg_score - pos_score,
                               min=0).sum() / len(pos_score)


    def pos_neg_indices(self, cluster_idx, gt_idx):
        """构建 postive and neglect pair

        Args:
            cluster_idx (tensor): 指明聚类中心所属 clip 下标 [n_v, num_cluster]
            gt_idx (tensor): gt 的 st, ed [vbsz, 2]
        Return:
            pos_mask: 在 [gt+boundary]内的 clutser clip [n_v, num_cluster]
            neg_mask: 不在 [gt+boundary]内的 clutser clip[n_v, num_cluster]
        """
        boundary_bias = 1
        _gt_idx = torch.zeros_like(gt_idx)
        # boundary bias set 1 note: moment length mainly lay in 3s 6s 9s(2clips 4clips 6clips)
        _gt_idx[:, 0] = gt_idx[:, 0] - boundary_bias
        _gt_idx[:, 1] = gt_idx[:, 1] + boundary_bias
        # cluster_clip - st = +/- +: cluster clip 落在 grounding truth 之前, -: cluster clip 落在 grounding truth 之后
        st_dis = cluster_idx - _gt_idx[:, 0].unsqueeze(1)
        st_pos_mask = st_dis > 0
        # cluster_clip - ed = +/- +: cluster clip 落在 grounding truth 之后, -: cluster clip 落在 grounding truth 之前
        ed_dis = cluster_idx - _gt_idx[:, 1].unsqueeze(1)
        ed_pos_mask = ed_dis < 0
        # pos_mask: [vbsz, num_cluster]
        pos_mask = (st_pos_mask & ed_pos_mask).float()
        # exist: [vbsz, ]
        non_pos_index = torch.all(pos_mask == 0, dim=1)
        if non_pos_index.any(dim=0):
            # cluster clip 都没落在 grounding truth 内，pos 按离grounding truth center 最近的 clip 当做 postive
            center = (_gt_idx[non_pos_index, 0] + _gt_idx[non_pos_index, 1]) / 2
            dis = torch.abs(cluster_idx[non_pos_index] - center.unsqueeze(1))
            _, indices = dis.min(dim=1)
            pos_mask[non_pos_index, indices] = 1
        neg_mask = 1 - pos_mask
        return pos_mask.bool(), neg_mask.bool()

class VGaussian_weight(nn.Module):
    def __init__(self, sigma=0.4,):
        super(VGaussian_weight, self).__init__()
        self.sigma = sigma  # sigma是一个超参数, 论文中设为0.4在 ActivityNet Captions 表现最佳

    def forward(self, feat_mask, center_indeices, gt_indices):
        # 下标映射到[-1, 1]区间
        samples, center_indeices_scale = self.index_transform(feat_mask=feat_mask, cluster_indices=center_indeices)
        # 计算不同的 sample 的 mu 生成高斯分布进行采样 gaussian_weight: [bsz, num_cluster, max_ctx]
        gaussian_weight = self.sample_weight(samples, center_indeices_scale, feat_mask, gt_indices)
        return gaussian_weight
    def sample_weight(self, samples, center_indeices_scale, feat_mask, gt_indices):
        """采样高斯分布

        Args:
            sample (tensor): 采样点 [nums, max_ctx]
            center_index (tensor): cluster clip采样点 [nums, cluster_num]
        Returns:
            gaussian_weight: 高斯权重
        """
        device = samples.device
        _gaussian_weight = []
        gaussian_weight = []
        samples = samples.cpu()
        center_indeices_scale = center_indeices_scale.cpu()
        for sample, center_index_scale, mask, gt_index in zip(samples, center_indeices_scale, feat_mask, gt_indices):
            mu_s = center_index_scale
            # 防止除以 0
            relative_rate = 0.01 + (gt_index[1]- gt_index[0]) / (mask.sum())
            # 3 sigma 准则
            sigma = (relative_rate / 4).cpu()
            # 计算高斯分布的PDF
            for mu in mu_s:
                pdf = norm.pdf(sample, mu, sigma)
                pdf = pdf * mask.cpu().numpy()
                _gaussian_weight.append(pdf)
            tmp_weight = np.stack(_gaussian_weight)
            _gaussian_weight = []
            # list: [bsz * [(25, 100)]]
            gaussian_weight.append(tmp_weight)
            # 以下用于可视化检查采样分布
            # import matplotlib.pyplot as plt
            # # 绘制高斯分布的PDF
            # plt.figure(figsize=(10, 6))
            # plt.plot(sample, pdf, label=f'Sample mu:{mu}')
            # # plt.plot(sample, pdf_ed, label=f'Sample mu_ed:{mu_ed}')
            # # plt.plot(sample, pdf, label=f'Sample mu_st+mu_ed:{mu_st + mu_ed}')
            # plt.xlabel('x')
            # plt.ylabel('Probability Density')
            # plt.title('Gaussian Probability Density Function')
            # plt.legend()
            # plt.grid(True)
            # plt.savefig("gausion1.jpg")
            # plt.show()
        gaussian_weight = torch.from_numpy(np.stack(gaussian_weight, )).to(device=device)
        # gaussian_weight_st = torch.from_numpy(np.stack(gaussian_weight_st)).to(device=device)
        # gaussian_weight_ed = torch.from_numpy(np.stack(gaussian_weight_ed)).to(device=device)
        # binary_weight = torch.from_numpy(np.stack(binary_weight)).to(device=device)
        return gaussian_weight
    def index_transform(self, feat_mask, cluster_indices):
        """下标映射到[-1, 1]
        Args:
            feat_mask (tensor): 下标 [vbsz, max_ctx]
            cluster_indices (tensor): [vbsz, cluster_num]
        Returns:
            index_trans: 映射后的下标
            已测试
        """
        # 形状、序列长度
        nums, max_ctx = feat_mask.shape
        ctx = feat_mask.sum(dim=1)
        # 下标 e.g [1 ,2 ,3 ...]
        index = torch.arange(1, max_ctx+1).to(feat_mask.device).unsqueeze(0).repeat(nums, 1)
        index_trans = (index - 1) * 2 / (ctx - 1).unsqueeze(1) - 1
        index_trans = index_trans * feat_mask
        cluster_indices = (cluster_indices - 1) * 2 / (ctx - 1).unsqueeze(1) - 1
        return index_trans, cluster_indices


class ShareNormLoss(nn.Module):

    def __init__(self, ):
        super(ShareNormLoss, self).__init__()
        self.nllloss = nn.NLLLoss(reduction="mean")
        self.v2q_ce_loss = nn.CrossEntropyLoss(reduction="mean")
        self.moment_ce_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, query_context_scores, st_prob, ed_prob, gt_indices,
                mask):
        """负对数似然损失

        Args:
            st/ed_prob (tensor): 1DConv 输出的概率 [vbsz, lv]
            gt_indices (tensor): ground truth
            mask (tensor): 掩码
        """
        # 采样前k个负样本(默认10) hard_negative_indices: [qbsz, 10]
        hard_negative_indices = self.hard_sample(query_context_scores)
        # 构建新的hard_st, hard_ed
        hard_st, hard_ed = self.hard_similarity_get(hard_negative_indices,
                                                    st_prob, ed_prob)
        # 生成0-255的张量下标
        indices = torch.arange(0, st_prob.shape[0]).to(st_prob.device)
        gt_st = gt_indices[:, 0]
        gt_st_conv = st_prob[indices, gt_st]
        gt_ed = gt_indices[:, 1]
        gt_ed_conv = ed_prob[indices, gt_ed]
        loss_st, loss_ed = self.share_normalization_loss(
            gt_st_conv, gt_ed_conv, hard_st, hard_ed)
        return loss_st, loss_ed
        # # exp_prob: [bsz, lv]
        # exp_st_prob = torch.exp(st_prob) * mask
        # exp_ed_prob = torch.exp(ed_prob) * mask
        # # 标量
        # share_st_prob = self.batch_prob(exp_st_prob, mask)
        # share_ed_prob = self.batch_prob(exp_ed_prob, mask)
        # # norm_st_prob: [bsz, lv]
        # norm_st_prob = exp_st_prob / share_st_prob
        # norm_ed_prob = exp_ed_prob / share_ed_prob
        # loss_st = self.nllloss(norm_st_prob, gt_st)
        # loss_ed = self.nllloss(norm_ed_prob, gt_ed)
        # return loss_st, loss_ed

    def moment_share_loss(self, st_prob, ed_prob, gt_indices, mask, weight_hard):
        """计算share_nomalization

        Args:
            st_prob (tensor): [qbsz, sample_num * lv] 第一个 sample 是ground truth
            ed_prob (tensor): [qbsz, sample_num * lv]
            gt_indices (tensor): [qbsz, 2]
            mask (tensor): [qbsz, sample_num, lv]
        Return:
            loss_st
            loss_ed
        """
        # TODO:
        # gasu_pro_st, gasu_pro_ed = self.moment_gaussian_loss(st_prob, ed_prob, gt_indices, mask)
        # pos_lv = mask.shape[1]
        # center_st = gasu_pro_st.max(dim=1)[0]
        # center_ed = gasu_pro_ed.max(dim=1)[0]
        # qbsz, sample_num, lv = mask.shape
        # gasu_pro_st 在 dim=1 维度用 0 填充和st_prob一样的形状


        # loss_st = self.moment_ce_loss(st_prob[:, :pos_lv], gasu_pro_st)
        # loss_ed = self.moment_ce_loss(ed_prob[:, :pos_lv], gasu_pro_ed)
        # loss_st = self.moment_ce_loss(st_prob, gt_indices[:, 0])
        # loss_ed = self.moment_ce_loss(ed_prob, gt_indices[:, 1])
        qbsz= st_prob.shape[0]
        gt_st = gt_indices[:, 0]
        gt_ed = gt_indices[:, 1]
        indices = torch.arange(qbsz).to(device=st_prob.device)
        # gt_st_conv_prob: [qbsz, ]
        gt_st_conv_prob = st_prob[indices, gt_st]
        gt_ed_conv_prob = ed_prob[indices, gt_ed]
        # st_denominator = torch.exp(st_prob).sum(dim=1)
        # ed_denominator = torch.exp(ed_prob).sum(dim=1)
        _st_prob = torch.where(st_prob < 0, torch.exp(st_prob), st_prob)
        _ed_prob = torch.where(ed_prob < 0, torch.exp(ed_prob), ed_prob)
        # _st_prob = torch.where(torch.logical_or(st_prob == st_prob.min(), st_prob < 0), 0, st_prob)
        # _ed_prob = torch.where(torch.logical_or(ed_prob == ed_prob.min(), ed_prob < 0), 0, ed_prob)
        st_denominator = _st_prob.sum(dim=1)
        ed_denominator = _ed_prob.sum(dim=1)
        # st_molecule: [qbsz, ]
        st_molecule = torch.exp(gt_st_conv_prob)
        ed_molecule = torch.exp(gt_ed_conv_prob)
        loss_st = torch.sum((-torch.log(weight_hard * (st_molecule / (st_denominator + st_molecule))))) / qbsz
        loss_ed = torch.sum((-torch.log(weight_hard * (ed_molecule / (ed_denominator + ed_molecule))))) / qbsz
        # loss_ed = torch.sum(weight_hard * (-torch.log(ed_molecule / (ed_denominator + ed_molecule)))) / qbsz

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
        # span 用 0 在 dim=1 补充到和 pos_mask 一样的形状
        span_mask = F.pad(span_mask, (0, pos_mask.shape[1] - span_mask.shape[1]))
        bsz = st_prob.shape[0]
        # gt video
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


        # gt_st = gt_indices[:, 0]
        # gt_ed = gt_indices[:, 1]
        # 保证gt_st和gt_ed至少持续 1.5
        # gt_ed = torch.where(gt_st == gt_ed, gt_ed + 1, gt_ed)
        # indices = torch.arange(gt_st.shape[0]).to(device=gt_st.device)
        # pos_st_prob[indices, gt_st+1:gt_ed-1]

    def generate_gaussion(self, gt_indices, mask):
        """生成高斯分布

        Args:
            gt_indices (tensor): [vbsz, 2]
            mask (tensor): [vbsz, lv]
        Return:
            gaussian_st, gaussian_ed
        """
        # sigma 的值由目标片段相对长度来决定
        # gt_indices: [vbsz, 2]
        gt_st = gt_indices[:, 0]
        gt_ed = gt_indices[:, 1]
        # 修正 gt_ed，使得 gt_st 和 gt_ed 相等时，gt_ed + 1
        gt_ed = torch.where(gt_st == gt_ed, gt_ed + 1, gt_ed)
        valid_len = mask.sum(dim=1)
        sigma = ((gt_ed - gt_st) / valid_len) * 5
        mu1 = gt_st
        mu2 = gt_ed
        # 生成高斯分布
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
        # qbsz, sample_num, lv = mask.shape
        pad_num = st_prob.min()
        qbsz = st_prob.shape[0]
        # gt_st = gt_indices[:, 0]
        # gt_ed = gt_indices[:, 1]
        indices = torch.arange(qbsz).to(device=st_prob.device)
        lv = mask.shape[1]
        # 取 ground truth 的概率
        gt_st_conv_prob = st_prob[:, :lv]
        gt_ed_conv_prob = ed_prob[:, :lv]
        # 按照概率计算损失
        gt_st_conv_prob = gt_st_conv_prob * gaussian_st
        gt_ed_conv_prob = gt_ed_conv_prob * gaussian_ed
        gt_st_conv_prob = gt_st_conv_prob.masked_fill(~mask.bool(), pad_num)
        gt_ed_conv_prob = gt_ed_conv_prob.masked_fill(~mask.bool(), pad_num)

        return gt_st_conv_prob, gt_ed_conv_prob

    def loss(self, st_prob, ed_prob, gt_indices, mask):
        """计算share_nomalization

        Args:
            st_prob (tensor): [qbsz, sample_num * lv] 第一个 sample 是ground truth
            ed_prob (tensor): [qbsz, sample_num * lv]
            gt_indices (tensor): [qbsz, 2]
            mask (tensor): [qbsz, sample_num, lv]
        Return:
            loss_st
            loss_ed
        """
        # TODO:
        qbsz, sample_num, lv = mask.shape
        gt_st = gt_indices[:, 0]
        gt_ed = gt_indices[:, 1]
        indices = torch.arange(qbsz).to(device=mask.device)
        # gt_st_conv_prob: [qbsz, ]
        gt_st_conv_prob = st_prob[indices, gt_st]
        gt_ed_conv_prob = ed_prob[indices, gt_ed]
        st_denominator = torch.exp(st_prob).sum(dim=1)
        ed_denominator = torch.exp(ed_prob).sum(dim=1)
        # st_molecule: [qbsz, ]
        st_molecule = torch.exp(gt_st_conv_prob)
        ed_molecule = torch.exp(gt_ed_conv_prob)
        loss_st = torch.sum(-torch.log(st_molecule / (st_denominator + 1e-8)))
        loss_ed = torch.sum(-torch.log(ed_molecule / (ed_denominator + 1e-8)))
        return loss_st, loss_ed

    def batch_prob(self, prob, mask):
        """
        batch 内 st/ed 的概率
        paramerter:
            prob: [vbsz, lv]
            mask: [vbsz, lv]
        """
        batch_prob = prob.sum()
        return batch_prob

    def hard_sample(self, similarity, sample_num=4):
        """Hard Negative Mining

        Args:
            similarity (tensor): 训练阶段得到的检索矩阵 [qbsz, vbsz]
        Return:
            hard_similarity: 硬负样本 [sample_num, max_ctx]
        """
        # 取负样本索引
        # hard_negative_indices: [qbsz, sample_num]
        hard_negative_indices = similarity.topk(sample_num,
                                                dim=1,
                                                largest=True)[1]
        # 负样本拼接成新的similarity矩阵
        return hard_negative_indices

    def query_diverse_loss(self,
                           modality_similarity,
                           modality_query,
                           sample_num=4,
                           delta=0.15,
                           alpha=10):
        # modality_query: [vbsz, d]
        # 采样同一个 video 中最相似的 query
        diag_score = torch.diag(modality_similarity).unsqueeze(1)
        modality_similarity = modality_similarity - torch.eye(
            modality_similarity.shape[0]).to(modality_similarity.device) * 9999
        # hard_query_indices: [sample_num, vbsz]
        hard_query_score, _ = modality_similarity.topk(sample_num,
                                                       dim=0,
                                                       largest=True)
        _, hard_query_indices = modality_similarity.topk(sample_num,
                                                         dim=0,
                                                         largest=False)
        # hard_query_indices: [vbsz, sample_num]
        hard_query_indices = hard_query_indices.permute(1, 0)
        hard_query_score = hard_query_score.permute(1, 0)
        # indices: [vbsz, 1]
        indices = torch.arange(modality_similarity.shape[0]).to(
            device=modality_similarity.device).unsqueeze(1)
        # pos_hard_query_indices: [vbsz, sample_num+1] 含 pos query for every video
        pos_hard_query_indices = torch.cat([indices, hard_query_indices],
                                           dim=1)
        pos_hard_query_score = torch.cat([diag_score, hard_query_score], dim=1)
        gt_idx = modality_similarity.new_zeros(
            modality_similarity.shape[0]).long()
        v2q_ce_loss = self.v2q_ce_loss(pos_hard_query_score, gt_idx)
        v2q_ce_loss = 0.1 * v2q_ce_loss
        bsz = modality_similarity.shape[0]
        hard_modality_query = []
        # 加入 pos query feat
        for i in range(bsz):
            # _hard_modality_query: [sample_num+1, d]
            _hard_modality_query = modality_query[pos_hard_query_indices[i]]
            hard_modality_query.append(_hard_modality_query)
        # hard_modality_query: [bsz, sample_num+1, d]
        hard_modality_query = torch.stack(hard_modality_query)
        # 计算向量的范数
        norm = torch.norm(hard_modality_query, dim=-1, keepdim=True)
        normalized_query = hard_modality_query / norm
        # cosine for query query_sim: [vbsz, sample_num+1,]
        # 计算余弦相似度
        query_sim = torch.einsum("vsd, vld -> vsl", normalized_query,
                                 normalized_query)
        # 自身相关性的惩罚: 0
        query_sim = query_sim - torch.eye(
            query_sim.shape[1]).unsqueeze(0).repeat(query_sim.shape[0], 1,
                                                    1).to(query_sim.device)
        # 计算多样性损失
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
            similarity (tensor): 训练阶段得到的检索矩阵 [qbsz, vbsz]
        Return:
            hard_similarity: 硬负样本 [sample_num, max_ctx]
        """
        # 取负样本索引
        # hard_negative_indices: [qbsz, sample_num]
        # similarity对角线元素设置成-9999
        diag_score = torch.diag(similarity).unsqueeze(1)
        similarity = similarity - torch.eye(similarity.shape[0]).to(
            similarity.device) * 9999
        hard_negative_score, hard_negative_indices = similarity.topk(
            sample_num, dim=1, largest=True)

        # 确保train的时候有正样本
        pos_indices = torch.arange(
            similarity.shape[0]).unsqueeze(1).to(device=similarity.device)
        # 将 index_tensor 和 original_tensor 在第 1 维度上进行拼接
        # hard_negative_score: [qbsz, sample_num + 1]
        hard_negative_score = torch.cat((diag_score, hard_negative_score),
                                        dim=1)
        hard_negative_indices = torch.cat((pos_indices, hard_negative_indices),
                                          dim=1)
        _context1_feat = []
        _context2_feat = []
        _mask = []
        for i in range(hard_negative_indices.shape[0]):
            # tmp_feat: [sample_num, lv, video_dim]
            tmp_feat1 = context1_feat[hard_negative_indices[i]]
            tmp_feat2 = context2_feat[hard_negative_indices[i]]
            tmp_mask = mask[hard_negative_indices[i]]
            _context1_feat.append(tmp_feat1)
            _context2_feat.append(tmp_feat2)
            _mask.append(tmp_mask)

        # _feat: [qbsz, sample_num, lv, video_dim]
        _context1_feat = torch.stack(_context1_feat)
        _context2_feat = torch.stack(_context2_feat)
        _mask = torch.stack(_mask)
        qbsz, sample_num, lv, video_dim = _context1_feat.shape

        # _feat = _feat.contiguous().view(qbsz * sample_num, lv, video_dim)
        # _mask = _mask.contiguous().view(qbsz * sample_num, lv)
        # 负样本拼接成新的similarity矩阵
        return _context1_feat, _context2_feat, _mask, hard_negative_score

    def hard_similarity_get(self, hard_negative_indices, st_prob, ed_prob):
        # st_prob: [vbsz, lv]
        # ed_prob: [vbsz, 10]
        st = []
        ed = []
        for i in range(hard_negative_indices.shape[0]):
            tmp_st = st_prob[hard_negative_indices[i]]
            tmp_ed = ed_prob[hard_negative_indices[i]]
            st.append(tmp_st)
            ed.append(tmp_ed)
        # st: [qbsz, sample_num, max_ctx]
        st = torch.stack(st)
        ed = torch.stack(ed)
        # 重新调整形状
        st = st.contiguous().view(hard_negative_indices.shape[0], -1)
        ed = ed.contiguous().view(hard_negative_indices.shape[0], -1)
        return st, ed

    def share_normalization_loss(self, gt_st, gt_ed, hard_st, hard_ed):
        # gt_st: [vbsz, ]
        # gt_ed: [vbsz, ]
        # hard_st: [vbsz, sample_num * lv]
        # hard_ed: [vbsz, sample_num * lv]
        # 计算 share normalization loss
        # st_denominator: [vbsz, ]
        st_denominator = torch.exp(hard_st).sum(dim=1)
        ed_denominator = torch.exp(hard_ed).sum(dim=1)
        # st_molecule: [vbsz, ]
        st_molecule = torch.exp(gt_st)
        ed_molecule = torch.exp(gt_ed)
        loss_st = torch.sum(-torch.log(st_molecule / (st_denominator + 1e-8)))
        loss_ed = torch.sum(-torch.log(ed_molecule / (ed_denominator + 1e-8)))
        return loss_st, loss_ed

class GateFusion(nn.Module):

    def __init__(self,):
        super(GateFusion, self).__init__()
        self.vid_weight = nn.Linear(256, 1)
        self.sub_weight = nn.Linear(256, 1)
        self.vid_linear = nn.Linear(256, 256)
        self.sub_linear = nn.Linear(256, 256)
        self.gate = nn.Linear(256 * 2, 256)
        self.video_cross_layer = CrossAttention(256,)
        self.sub_cross_layer = CrossAttention(256,)
        pass

    def query_aware(self, video_feat, sub_feat, video_query, sub_query, is_eval=False):
        # (bsz, lv, 1)
        _video_weight = self.vid_weight(video_feat)
        _sub_weight = self.sub_weight(sub_feat)
        _video_feat = self.vid_linear(video_feat)
        _sub_feat = self.sub_linear(sub_feat)
        if is_eval:
            video_query = video_query.unsqueeze(0).unsqueeze(0)
            sub_query = sub_query.unsqueeze(0).unsqueeze(0)
        if is_eval == False:
            video_query = video_query.unsqueeze(1)
            sub_query = sub_query.unsqueeze(1)
        # query-aware
        _video_feat = _video_feat * video_query
        _sub_feat = _sub_feat * sub_query
        _video_feat = _video_feat * _video_weight
        _sub_feat = _sub_feat * _sub_weight
        # L2 normalization
        _video_feat = F.normalize(_video_feat, dim=-1)
        _sub_feat = F.normalize(_sub_feat, dim=-1)
        video_feat = _video_feat * video_feat
        sub_feat = _sub_feat * sub_feat
        return video_feat, sub_feat

    def forward(self, video_feat, sub_feat, context_mask, video_query, sub_query, is_eval=False):
        video_feat, sub_feat = self.query_aware(video_feat, sub_feat, video_query, sub_query, is_eval)
        fusion_feat = torch.cat([video_feat, sub_feat], dim=-1)
        q_aware_feat = self.gate(fusion_feat)
        v_feat = self.video_cross_layer(q_aware_feat, context_mask, video_feat, context_mask)
        s_feat = self.sub_cross_layer(q_aware_feat, context_mask, sub_feat, context_mask)
        return v_feat, s_feat

class VS_score(nn.Module):
    def __init__(self,):
        super(VS_score, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.video_score_predictor = nn.Sequential(
            nn.Linear(in_features=256, out_features=128, bias=False),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1, bias=False)
        )
        # self.video_score_predictor = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=128, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=1, bias=False)
        # )
        # self.sub_score_predictor = nn.Sequential(
        #     nn.Linear(in_features=256, out_features=128, bias=False),
        #     nn.ReLU(),
        #     nn.Linear(in_features=128, out_features=1, bias=False)
        # )
    def forward(self, video_feat,):
        video_score = self.video_score_predictor(video_feat)
        video_score = video_score.squeeze(-1)
        video_score = video_score.max(dim=1)[0]
        return video_score

    def video_score_loss(self, video_score, measure="nce"):
        # video_score: [bsz, sample]
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
            # video_score = F.softmax(video_score, dim=-1)
            vs_loss = self.ce_loss(video_score, ce_label)
            vs_loss = vs_loss * 0.1
        # 新增triplet loss
        triplet_loss = 0
        # pos_score: [bsz, ]
        pos_score = video_score[:, 0]
        # 生成一个形状为 (256, 1) 的张量，范围在 [1, 4] 之间，包含 1 和 4
        random_tensor = torch.randint(1, 5, (bsz, 1)).to(video_score.device)
        # neg_score: [bsz, ]
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
        # conv_cfg_2 = {
        # "in_channels": 128,
        # "out_channels": 1,
        # "kernel_size": 3,
        # "stride": 1,
        # "padding": 1,
        # "bias": False
        # }
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
        # stage1_score: [qbsz, sample + 1]
        # stage2_score: [qbsz, sample + 1]
        qbsz, sample = stage1_score.shape
        # stage1_score_pos: [qbsz, ]
        stage1_score_pos = stage1_score[:, 0]
        stage2_score_pos = stage2_score[:, 0]
        pos_score = torch.min(stage1_score_pos, stage2_score_pos)
        # stage1_score_neg: [qbsz, ]
        stage1_score_neg = stage1_score[:, 1:].sum(dim=1)
        stage2_score_neg = stage2_score[:, 1:].sum(dim=1)
        neg_score = (stage1_score_neg + stage2_score_neg)
        consistency_loss = torch.clamp(neg_score - pos_score + self.margin, min=0)
        consistency_loss = consistency_loss.sum() / qbsz
        return consistency_loss
    def rrf_convert(self, score_eval, k=9):
        # 基于排名的相对排序范围在1/9 ~ 1/(2179+9)
        # Get the ranks by sorting the scores in descending order
        score_eval_rank = torch.argsort(-score_eval, dim=1)  # Rank indices for stage1
        # Apply RRF formula in a vectorized way
        fused_scores = 1 / (k + score_eval_rank.float())
        # fused_scores = torch.exp(fused_scores)
        # fused_scores = torch.log(fused_scores)
        return fused_scores
    def rrf_fusion(self, stage1_score, stage2_score, k=9, is_eval=False):
        """
        基于排名的得分融合
        Perform Reciprocal Rank Fusion (RRF) on two stages of scores using GPU-efficient operations.

        Args:
            stage1_score (torch.Tensor): Scores from stage 1 with shape (bsz, 5).
            stage2_score (torch.Tensor): Scores from stage 2 with shape (bsz, 5).
            k (int): The RRF constant controlling the impact of rank (default=60).

        Returns:
            torch.Tensor: Fused scores with shape (bsz, 5).
        """
        # Get the ranks by sorting the scores in descending order
        stage1_rank = torch.argsort(-stage1_score, dim=1)  # Rank indices for stage1
        stage2_rank = torch.argsort(-stage2_score, dim=1)  # Rank indices for stage2

        # Generate rank matrices: find where each item ranks in stage1 and stage2
        rank1 = torch.argsort(stage1_rank, dim=1)  # Convert indices to ranks
        rank2 = torch.argsort(stage2_rank, dim=1)

        # Apply RRF formula in a vectorized way
        fused_scores = 1 / (k + rank1.float()) + 1 / (k + rank2.float())
        if is_eval:
            fused_scores = fused_scores.squeeze(0)
        return fused_scores

    def tmp(self, stage1_score, stage2_score):
        # stage1_score: [qbsz, sample + 1]
        # stage2_score: [qbsz, sample + 1]
        qbsz, sample = stage1_score.shape
        # 标准化
        stage1_score = stage1_score / torch.sqrt((stage1_score * stage1_score).sum(dim=1, keepdim=True))
        stage2_score = stage2_score / torch.sqrt((stage2_score * stage2_score).sum(dim=1, keepdim=True))
        pos_visual_discrepancy = torch.abs(stage1_score - stage2_score)
        # 构造负样本, 这个索引可以改变
        fix_indices = [1, 0, 4, 2, 3]
        tmp_stage2_score = stage2_score[:, fix_indices]
        neg_visual_discrepancy = torch.abs(stage1_score - tmp_stage2_score)
        # 对比损失
        triplet_loss = torch.clamp(pos_visual_discrepancy - neg_visual_discrepancy + self.margin, min = 0)
        # triplet_loss = triplet_loss.sum() / sample 太小
        triplet_loss = triplet_loss.sum() / qbsz
        return triplet_loss

class BidirectionalAttention(nn.Module):

    def __init__(self, video_dim):
        super(BidirectionalAttention, self).__init__()
        ## Core Attention for query-aware feature learining
        # self.similarity_weight = nn.Linear(video_dim * 3, 1, bias=False)
        self.visual_similarity_weight = nn.Linear(video_dim * 3, 1, bias=False)
        # self.sub_similarity_weight = nn.Linear(video_dim * 3, 1, bias=False)
        ## Query_aware_feature_learning Module
        self.query_weight = QueryWeightEncoder()
        self.visual_fc = LinearLayer(video_dim * 5, video_dim)
        # self.visual_fc = LinearLayer(video_dim * 4, video_dim)
        # self.sub_fc = LinearLayer(video_dim * 4, video_dim)
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

        # 转换为 EasyDict
        bert_config = easydict.EasyDict(bert_config_dict)
        self.contextual_QAL_feature_learning = FCPlusTransformer(config=bert_config, input_dim=video_dim * 4)
        self.visual_encoder = BertEncoder(bert_config)
        self.st_encoder = FCPlusTransformer(config=bert_config, input_dim=video_dim * 5)
        # 考虑到st的特征
        self.ed_encoder = FCPlusTransformer(config=bert_config, input_dim=video_dim * 2)
        self.begin_score_modeling = ConvSE()
        self.end_score_modeling = ConvSE()
        # self.sub_encoder = BertEncoder(bert_config)

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
        # QDF_visual_emb, QDF_sub_emb = self.query_weight(query_emb, video_feat, sub_feat)
        ## CREATE SIMILARITY MATRIX
        video_len = QDF_visual_emb.size()[1]
        query_len = query_emb.size()[1]

        _QDF_visual_emb = QDF_visual_emb.unsqueeze(2).repeat(1, 1, query_len, 1)
        # _QDF_sub_emb = QDF_sub_emb.unsqueeze(2).repeat(1, 1, query_len, 1)
        # [bs, video_len, 1, feat_size] => [bs, video_len, query_len, feat_size]

        _query_emb = query_emb.unsqueeze(1).repeat(1, video_len, 1, 1)
        # [bs, 1, query_len, feat_size] => [bs, video_len, query_len, feat_size]

        elementwise_visual_prod = torch.mul(_QDF_visual_emb, _query_emb)
        # elementwise_sub_prod = torch.mul(_QDF_sub_emb, _query_emb)
        # [bs, video_len, query_len, feat_size]

        visual_alpha = torch.cat([_QDF_visual_emb, _query_emb, elementwise_visual_prod], dim=3)
        # sub_alpha = torch.cat([_QDF_sub_emb, _query_emb, elementwise_sub_prod], dim=3)
        # [bs, video_len, query_len, feat_size*3]

        similarity_visual_matrix = self.visual_similarity_weight(visual_alpha).view(-1, video_len, query_len)
        # similarity_sub_matrix = self.sub_similarity_weight(sub_alpha).view(-1, video_len, query_len)

        similarity_matrix_mask = torch.einsum("bn,bm->bnm", video_mask, query_mask)
        # [bs, video_len, query_len]

        ## CALCULATE Video2Query ATTENTION

        a = F.softmax(mask_logits(similarity_visual_matrix,
                                  similarity_matrix_mask), dim=-1)
        # aa = F.softmax(mask_logits(similarity_sub_matrix,
        #                           similarity_matrix_mask), dim=-1)
        # [bs, video_len, query_len]

        visual_V2Q = torch.bmm(a, query_emb)
        # sub_V2Q = torch.bmm(aa, query_emb)
        # [bs] ([video_len, query_len] X [query_len, feat_size]) => [bs, video_len, feat_size]

        ## CALCULATE Query2Video ATTENTION

        b = F.softmax(torch.max(mask_logits(similarity_visual_matrix, similarity_matrix_mask), 2)[0], dim=-1)
        # bb = F.softmax(torch.max(mask_logits(similarity_sub_matrix, similarity_matrix_mask), 2)[0], dim=-1)
        # [bs, video_len]

        b = b.unsqueeze(1)
        # bb = bb.unsqueeze(1)
        # [bs, 1, video_len]

        visual_Q2V = torch.bmm(b, QDF_visual_emb)
        # sub_Q2V = torch.bmm(b, QDF_sub_emb)
        # [bs] ([bs, 1, video_len] X [bs, video_len, feat_size]) => [bs, 1, feat_size]

        visual_Q2V = visual_Q2V.repeat(1, video_len, 1)
        # sub_Q2V = sub_Q2V.repeat(1, video_len, 1)
        # [bs, video_len, feat_size]

        ## Append QDF_emb with three query-aware features

        visual_QAL = torch.cat([QDF_visual_emb, visual_V2Q,
                         torch.mul(QDF_visual_emb, visual_V2Q),
                         torch.mul(QDF_visual_emb, visual_Q2V)], dim=2)
        ## Contextualize QAL features
        Contextual_QAL  = self.contextual_QAL_feature_learning(
            features=visual_QAL,
            feat_mask=video_mask)
        G = torch.cat([visual_QAL,Contextual_QAL], dim=2)
        # sub_QAL = torch.cat([QDF_sub_emb, sub_V2Q,
        #                  torch.mul(QDF_sub_emb, sub_V2Q),
        #                  torch.mul(QDF_sub_emb, sub_Q2V)], dim=2)
        st_feat = self.st_encoder(G, video_mask)
        ed_feat = self.ed_encoder(torch.cat([Contextual_QAL, st_feat], dim=2), video_mask)
        st_prob = self.begin_score_modeling(st_feat, video_mask)
        ed_prob = self.end_score_modeling(ed_feat, video_mask)
        # [bs, video_len, feat_size*4]
        visual_QAL = self.visual_fc(G)
        # sub_QAL = self.sub_fc(sub_QAL)
        visual_QAL = self.visual_encoder(visual_QAL, attention_mask=video_mask)[0]
        # sub_QAL = self.sub_encoder(sub_QAL, attention_mask=video_mask)[0]
        return visual_QAL, st_prob, ed_prob
        # return visual_QAL, sub_QAL


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
        # 定义字典
        # hidden_size 从 768 改成 256
        config_dict = {
                "hidden_size": 256,
                "text_cluster": 32,
                "moe_dropout_prob": 0.1
        }
        # 转换为 EasyDict
        config = easydict.EasyDict(config_dict)
        ##NetVLAD
        self.text_pooling = NetVLAD(feature_size=config.hidden_size,cluster_size=config.text_cluster)
        self.moe_txt_dropout = nn.Dropout(config.moe_dropout_prob)

        ##FC
        self.moe_fc_txt = nn.Linear(
            in_features=self.text_pooling.out_dim,
            out_features=len(video_modality),
            bias=False)

        self.video_modality = video_modality

    def forward(self, query_feat, video_feat, sub_feat):
        ##NetVLAD
        pooled_text = self.text_pooling(query_feat)
        pooled_text = self.moe_txt_dropout(pooled_text)

        ##FC + Softmax
        moe_weights = self.moe_fc_txt(pooled_text)
        softmax_moe_weights = F.softmax(moe_weights, dim=1)

        query_video_weight, query_sub_weight = softmax_moe_weights.split(1, dim = 1)
        query_video_weight = query_video_weight.squeeze(1)
        query_sub_weight = query_sub_weight.squeeze(1)
        final_query_context_scores = self.compute_final_score(video_feat, sub_feat, query_video_weight, query_sub_weight)
        return final_query_context_scores
        # return query_video_weight, query_sub_weight
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

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = F.normalize(vlad)

        return vlad



if __name__ == "__main__":

    rank_context = RankInContext()
    bsz, clusters, clips = 3, 25, 100
    # 模拟 mask
    feat_mask = torch.ones(bsz, clips)
    # 构建一个形状为[bsz, clusters]的张量，数值在0到100之间
    cluster_idx = torch.randint(0, clips, (bsz, clusters))
    gt_idx = torch.randint(0, clips, (bsz, 2))
    gt_idx[:, 0], gt_idx[:, 1] = torch.min(gt_idx, dim=1)[0], torch.max(gt_idx, dim=1)[0]
    # 模拟检索结果
    q2ctx_score = torch.rand(bsz, clips)
    # pos_anchor_indices, neg_anchor_indices = rank_context.pos_neg_indices(cluster_idx, gt_idx)
    pos_anchor_indices, neg_anchor_indices = rank_context.get_anchor(cluster_idx, gt_idx, q2ctx_score, feat_mask)
    pos_anchor_indices = rank_context.get_ranking_loss(pos_anchor_indices, neg_anchor_indices)
    pass
