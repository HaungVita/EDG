
import torch
from torch import nn

def mask_logits(target, mask):
    #import pdb;pdb.set_trace()
    return target * mask + (1 - mask) * (-1e10)

class SemanticNCE(nn.Module):

    def __init__(self):
        super(SemanticNCE, self).__init__()
        conv_cfg = dict(in_channels=1,
                        out_channels=1,
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        bias=False)
        self.v_scoreConv1ST = nn.Conv1d(**conv_cfg)
        self.v_scoreConv1ED = nn.Conv1d(**conv_cfg)
        self.s_scoreConv1ST = nn.Conv1d(**conv_cfg)
        self.s_scoreConv1ED = nn.Conv1d(**conv_cfg)


    def forward(self, video_query, video_feat, sub_query, sub_feat, context_mask, video_cluster_idx, sub_cluster_idx, st_ed_indices, cross=False):
        # 如果是 infer
        if cross == True:
            video_similarity = torch.einsum("md,nld->mnl", video_query,
                                            video_feat)
            sub_similarity = torch.einsum("md,nld->mnl", sub_query, sub_feat)
            n_q, n_c, l = video_similarity.shape
            video_similarity = video_similarity.view(n_q * n_c, 1, l)
            sub_similarity = sub_similarity.view(n_q * n_c, 1, l)
            v_st_prob = self.v_scoreConv1ST(video_similarity).view(
                n_q, n_c, l)  # (Nq, Nv, L) 这里的形状是(1, 100, 100)
            v_st_prob = mask_logits(v_st_prob, context_mask)  # (N, L)
            v_ed_prob = self.v_scoreConv1ED(video_similarity).view(
                n_q, n_c, l)  # (Nq, Nv, L) 这里的形状是(1, 100, 100)
            v_ed_prob = mask_logits(v_ed_prob, context_mask)
            s_st_prob = self.s_scoreConv1ST(sub_similarity).view(
                    n_q, n_c, l)  # (Nq, Nv, L)
            s_st_prob = mask_logits(s_st_prob, context_mask)  # (N, L)
            s_ed_prob = self.s_scoreConv1ED(sub_similarity).view(
                    n_q, n_c, l)  # (Nq, Nv, L)
            s_ed_prob = mask_logits(s_ed_prob, context_mask)
            st_prob = v_st_prob + s_st_prob
            ed_prob = v_ed_prob + s_ed_prob
            return st_prob, ed_prob
        if cross == False:
            video_similarity = torch.einsum("bd,bld->bl", video_query,
                                                video_feat)  # (N, L)
            sub_similarity = torch.einsum("bd,bld->bl", sub_query,
                                            sub_feat)  # (N, L)
            v_st_prob = self.v_scoreConv1ST(
                video_similarity.unsqueeze(1)).squeeze()  # (N, L)
            v_st_prob = mask_logits(v_st_prob, context_mask)  # (N, L)
            v_ed_prob = self.v_scoreConv1ED(
                video_similarity.unsqueeze(1)).squeeze()  # (N, L)
            v_ed_prob = mask_logits(v_ed_prob, context_mask)
            s_st_prob = self.s_scoreConv1ST(
                sub_similarity.unsqueeze(1)).squeeze()  # (N, L)
            s_st_prob = mask_logits(s_st_prob, context_mask)  # (N, L)
            s_ed_prob = self.s_scoreConv1ED(
                sub_similarity.unsqueeze(1)).squeeze()  # (N, L)
            s_ed_prob = mask_logits(s_ed_prob, context_mask)

            # ----------Begin of NCE loss----------
            loss_video_nce_base_semantic = self.svmr_cl_loss(semantic_label=video_cluster_idx, st_prob=v_st_prob, ed_prob=v_ed_prob,gt=st_ed_indices)
            loss_sub_nce_base_semantic = self.svmr_cl_loss(semantic_label=sub_cluster_idx, st_prob=s_st_prob, ed_prob=s_ed_prob,gt=st_ed_indices)
            NCE_loss = (loss_video_nce_base_semantic +  loss_sub_nce_base_semantic) / 2
            # 系数
            NCE_loss *= .008
            # ----------End of NCE loss----------
            return NCE_loss
    def get_non_gt_score_exp(self, prob, highest_non_gt_indices_list):
        """获得 non-GT 内所有 clip 与 query 匹配的均值, 表示噪声

        Args:
            prob (tensor): _description_
            highest_non_gt_indices_list (list): 一个 batch 内所有 non-GT 的clip 索引

        Returns:
            non_gt_scores: non-GT 内所有 clip 与 query 匹配的均值, 表示噪声[bsz, ]
        """
        # 获取非 GT clip 和查询的得分
        non_gt_scores_clip = []
        non_gt_scores_score = []
        for i in range(len(highest_non_gt_indices_list)):

            # 获取当前语义对应的所有 clip 下标
            non_gt_index = highest_non_gt_indices_list[i]

            if isinstance(non_gt_index, int):
                non_gt_scores_score.append(torch.tensor(0, device=prob.device))

            else:
                # 获取当前语义对应的所有 clip 和 query 匹配的 scores
                non_gt_score = prob[i, non_gt_index]

                # 计算把 non-GT 的 clip 添加到列表中
                non_gt_scores_clip.append(non_gt_score)

                # 计算负样本的得分
                non_gt_scores_score.append(sum(torch.exp(non_gt_score)))

        # non_gt_scores: [bsz, ]
        non_gt_scores = torch.stack(non_gt_scores_score)
        return non_gt_scores

    def get_gt_score_exp(self, st_prob, ed_prob, gt):
        """用于计算 gt 两端的得分

        Args:
            prob (tensor): 预测 st/ed 概率 [bsz, max_ctx, ]
            gt (tensor): gt的[st, ed] [bsz, 2]
        Return:
            gt_scores: GT 两端 clip 与 query 匹配值 [bsz, 2]
        """
        # st_score: [bsz, ]
        st_score = torch.sum(torch.exp(st_prob[:, gt[:, 0]]), dim=1)
        ed_score = torch.sum(torch.exp(ed_prob[:, gt[:, 1]]), dim=1)

        # gt_scores: [bsz, 2]
        gt_scores = torch.cat([st_score.unsqueeze(1), ed_score.unsqueeze(1)], dim=1)
        return gt_scores

    def svmr_cl_loss(self, semantic_label, st_prob, ed_prob, gt):
        """用于计算语义对比学习

        Args:
            semantic_label (tensor): 每个 clip 对应的语义标签 [bsz, max_ctx]
            prob (tensor): 预测 st/ed 概率 [bsz, max_ctx, ]
            gt (tensor): gt的[st, ed] [bsz, 2]
        """
        # 提取non-GT得分最高的语义
        highest_non_gt_st_indices_list = self.get_hard_semantic(semantic_label, st_prob, gt)
        highest_non_gt_ed_indices_list = self.get_hard_semantic(semantic_label, ed_prob, gt)
        # non_gt_st_scores: [bsz]
        non_gt_st_scores = self.get_non_gt_score_exp(st_prob, highest_non_gt_st_indices_list)
        non_gt_ed_scores = self.get_non_gt_score_exp(ed_prob, highest_non_gt_ed_indices_list)
        # gt_ed_scores: [bsz, 2]
        gt_scores = self.get_gt_score_exp(st_prob, ed_prob, gt)
        loss_nce_st_base_sematic = self.calculate_NCE(non_gt_st_scores, gt_scores[:, 0])
        loss_nce_ed_base_sematic = self.calculate_NCE(non_gt_ed_scores, gt_scores[:, 1])
        loss_nce_base_sematic = (loss_nce_st_base_sematic + loss_nce_ed_base_sematic) / 2
        return loss_nce_base_sematic

    def calculate_NCE(self, non_gt_scores, gt_scores):

        # 分子numerator: [bsz, ]
        # numerator = torch.exp(gt_scores)

        # 分母denominator: [bsz, ]
        # denominator = torch.exp(non_gt_scores) + torch.exp(gt_scores)

        # 前面计算得分已经使用了 exp
        loss_nce_base_sematic = (torch.sum(-torch.log(gt_scores / (gt_scores+non_gt_scores))))
        return loss_nce_base_sematic

    def get_hard_semantic(self, semantic_label, prob, gt):
        """选择 non-gt 与 query 最相似的语义，用于对比

        Args:
            semantic_label (tensor): 每个 clip 对应的语义标签 [bsz, max_ctx]
            prob (tensor): 预测 st/ed 概率 [bsz, max_ctx, ]
            gt (tensor): gt的[st, ed] [bsz, 2]
        Returen:
            highest_non_gt_indices_list (list): GT 以外与 query 最匹配的 clip 索引
        """
        # 批次大小
        bsz, max_ctx = semantic_label.size()
        device = semantic_label.device

        # 创建一个掩码，将 GT 范围内的部分设置为 True
        gt_starts = gt[:, 0].unsqueeze(1).to(device)
        gt_ends = gt[:, 1].unsqueeze(1).to(device)

        # 创建一个 index 张量，表示每个位置的索引
        index = torch.arange(max_ctx).unsqueeze(0).repeat(bsz, 1).to(device)

        # 创建掩码
        mask = (index >= gt_starts) & (index <= gt_ends)

        # 将 GT 范围内的部分设置为 -100
        prob = prob.masked_fill(mask, -1.0000e+10)

        # 得到 GT 范围内的语义标签
        pos_semantic = [semantic_label[i][mask[i]] for i in range(bsz)]
        pos_semantic = [i.unique() for i in pos_semantic]

        highest_non_gt_indices_list = []
        for i in range(bsz):
            # 得到最高得分标签
            highest_non_gt_indices = torch.argmax(prob[i])
            cluster_idx = semantic_label[i, highest_non_gt_indices]
            # 用于防止gt语义过于丰富
            select_semantic_count = 0
            # 如果这个语义在 GT 内，则换一个语义，直到这个语义不在 GT 内
            while cluster_idx in pos_semantic[i]:
                select_semantic_count += 1
                prob[[i], highest_non_gt_indices] = -1.0000e+10
                # 换一个语义
                highest_non_gt_indices = torch.argmax(prob[i])
                cluster_idx = semantic_label[i, highest_non_gt_indices]
                # GT 内超过 10 种语义不计算 NCE
                if select_semantic_count > 10:
                    break
            # 得到 non-GT 范围内的最高得分标签
            if select_semantic_count > 10:
                # GT 内超过 10 种语义不计算 NCE，用-1 下标表示不计算
                highest_non_gt_indices_list.append(-1)
            if select_semantic_count <= 10:
                highest_non_gt_indices = torch.where(semantic_label[i] == cluster_idx)[0]
                highest_non_gt_indices_list.append(highest_non_gt_indices)
        return highest_non_gt_indices_list