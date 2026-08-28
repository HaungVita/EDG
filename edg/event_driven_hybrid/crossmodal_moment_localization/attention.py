import torch
import torch.nn as nn
import torch.nn.functional as nnf
from einops import rearrange

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return nnf.silu(gate) * x

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return nnf.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

class MLP(nn.Module):

    def forward(self, x):
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.ReLU):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class CrossAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim=None,
        dim_head=64,
        heads=12,
        parallel_ff=True,
        ff_mult=4,
        norm_context=True
    ):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, dim_head * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether to have parallel feedforward

        ff_inner_dim = ff_mult * dim

        self.ff = nn.Sequential(
            nn.Linear(dim, ff_inner_dim * 2, bias=False),
            nn.ReLU(),
            nn.Linear(ff_inner_dim * 2, dim, bias=False)
        ) if parallel_ff else None
        # self.ff = nn.Sequential(
        #     nn.Linear(dim, ff_inner_dim * 2, bias=False),
        #     # SwiGLU(),
        #     nn.Linear(ff_inner_dim, dim, bias=False)
        # ) if parallel_ff else None

    def forward(self, x, x_mask, context, context_mask, pre_layernorm=False, local_att=False):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """
        if pre_layernorm == True:
            # pre-layernorm, for queries and context

            x = self.norm(x)
            context = self.context_norm(context)

        # get queries

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads)

        # scale

        q = q * self.scale

        # get key / values

        k, v = self.to_kv(context).chunk(2, dim=-1)

        # query / key similarity

        sim = torch.einsum('b h i d, b j d -> b h i j', q, k)

        # attention

        sim = sim - sim.amax(dim=-1, keepdim=True)

        # apply context mask
        context_mask_expanded = context_mask.unsqueeze(1).unsqueeze(2)  # 扩展掩码到形状 [bsz, 1, 1, len]
        sim = sim.masked_fill(context_mask_expanded == 0, float('-1e14'))  # 掩码无效位置填充 -inf

        # 修改, 这步骤其实没啥用
        x_mask = x_mask.unsqueeze(1).unsqueeze(2)  # 扩展掩码到形状 [bsz, 1, 1, len]
        sim = sim.permute(0, 1, 3, 2).masked_fill(x_mask == 0, float('-1e14')).permute(0, 1, 3, 2)  # 掩码无效位置填充 -inf
        if local_att == True:
            # local_size = 3
            n = sim.size(-1)
            local_matrix_mask = torch.zeros((n, n), dtype=torch.int).to(sim.device)
            local_matrix_mask.fill_diagonal_(1)  # 主对角线为 1
            local_matrix_mask[torch.arange(n-1), torch.arange(1, n)] = 1  # 主对角线下方为 1
            local_matrix_mask[torch.arange(1, n), torch.arange(n-1)] = 1  # 主对角线上方为 1
            local_matrix_mask[torch.arange(n-2), torch.arange(2, n)] = 1  # 主对角线下2方为 1
            local_matrix_mask[torch.arange(2, n), torch.arange(n-2)] = 1  # 主对角线上2方为 1
            sim = sim.masked_fill(local_matrix_mask.unsqueeze(0).unsqueeze(0) == 0, float('-1e14'))  # 掩码无效位置填充 -inf
        attn = sim.softmax(dim=-1)
        # x_mask: [bsz, 1, 1, len]
        x_mask = x_mask.permute(0, 1, 3, 2)
        attn = attn.masked_fill(x_mask == 0, 0)  # 掩码无效位置填充 0
        # aggregate

        out = torch.einsum('b h i j, b j d -> b h i d', attn, v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        # add parallel feedforward (for multimodal layers)

        if exists(self.ff):
            out = out + self.ff(x)

        return out


class GateTransformer(nn.Module):

    def __init__(self, d_model, dropout=0.1, activation="relu"):
        super(GateTransformer, self).__init__()
        # Gated Fusion
        self.gate_cross_attn = CrossAttention(dim=256, context_dim=256)
        self.gate_self_attn = CrossAttention(dim=256, context_dim=256, parallel_ff=False)
        # self.gate_cross_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
        # self.gate_self_attn = nn.MultiheadAttention(d_model, 1, dropout=0.1)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.gate_dropout = nn.Dropout(dropout)
        self.gate_norm = nn.LayerNorm(d_model)
        self.gate_linear = nn.Linear(d_model, d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, clip_feat, cross_clip_feat, clip_mask):
        # ========== Begin of Gated Fusion =============
        # 这里要求 cross_clip_feat 不能res 相加
        gate = (clip_feat * cross_clip_feat).sigmoid()
        # [bsz, lv, d]
        cross_clip_feat_enhance = clip_feat + cross_clip_feat
        cross_clip_feat_enhance = self.gate_self_attn(x=cross_clip_feat_enhance, x_mask=clip_mask, context=cross_clip_feat_enhance, context_mask=clip_mask)
        cross_clip_feat_enhance = self.gate_dropout(self.activation(self.gate_linear(gate*(cross_clip_feat_enhance)))) + clip_feat
        # ========== End of Gated Fusion =============
        cross_clip_feat_enhance = self.gate_norm(cross_clip_feat_enhance)
        return cross_clip_feat_enhance

    def forward_abort(self, semantic_feat, global_query, semantic_feat_mask=None, global_query_mask=None):
        # ========== Begin of Gated Fusion =============
        if semantic_feat_mask != None:  # 此时semantic_feat 是query的word_feat
            hgs2semantic_feat = self.gate_cross_attn(x=semantic_feat, x_mask=semantic_feat_mask, context=global_query, context_mask=global_query_mask)

        # hgs2semantic_feat: [bsz, clus_num, d_model]
        # hgs2semantic_feat = self.gate_cross_attn(query=semantic_feat,
        #                         key=global_query,
        #                         value=global_query)[0]  # \hat{C}
        gate = (semantic_feat*hgs2semantic_feat).sigmoid()
        hgs2semantic_feat = semantic_feat+hgs2semantic_feat
        hgs2semantic_feat = self.gate_self_attn(x=hgs2semantic_feat, x_mask=semantic_feat_mask, context=hgs2semantic_feat, context_mask=semantic_feat_mask)
        # hgs2semantic_feat = self.gate_self_attn(query=hgs2semantic_feat,
        #                            key=hgs2semantic_feat,
        #                            value=hgs2semantic_feat)[0]
        hgs2semantic_feat = self.gate_dropout(self.activation(self.gate_linear(gate*(hgs2semantic_feat)))) + semantic_feat
        hgs2semantic_feat = self.gate_norm(hgs2semantic_feat)
        # ========== End of Gated Fusion =============
        return hgs2semantic_feat

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return nnf.relu
    if activation == "gelu":
        return nnf.gelu
    if activation == "glu":
        return nnf.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")