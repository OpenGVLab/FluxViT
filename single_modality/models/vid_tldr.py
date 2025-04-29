# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
import math
import torch
from einops import rearrange
from typing import Callable, Tuple, List, Union
import torch.nn.functional as F

def get_objective_score(score_attn, eps=1e-17):
    # Mean across the first dimension
    score_attn = score_attn.mean(dim=1)

    # Use torch.clamp with a smaller range
    score_attn = torch.clamp(score_attn, min=eps, max=1.0 - eps)
    score_attn = F.normalize(score_attn, p=1, dim=-1)
    
    # Compute entropy using a numerically stable method
    scores = score_attn * torch.log(score_attn)
    scores = scores.sum(dim=2).unsqueeze(-1)

    # BACKGROUND REMOVING
    B, T_R, _ = scores.shape
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / (scores.amax(dim=1, keepdim=True))
    score_mean = scores.mean(dim=1, keepdim=True)
    score_mask = scores < score_mean

    # FOREGROUND SHARPENING
    scores = scores - score_mean
    scores = scores / (scores.amax(dim=1, keepdim=True))
    scores = scores.masked_fill(score_mask, 0.0)

    return scores

def gini_impurity(probabilities):
    return 1 - torch.sum(probabilities ** 2, dim=-1)


def get_objective_score_gini(score_attn):
    score_attn = score_attn.mean(dim=1)
    
    if torch.isnan(score_attn).any():
        raise ValueError('The Score Value has NAN before impurity operation.')
    
    # score_attn = torch.clamp(score_attn, min=eps, max=1.0 - eps)

    # Normalize to ensure it sums to 1 along the last dimension
    # score_attn = F.normalize(score_attn, p=1, dim=-1)

    # Compute Gini impurity
    scores = gini_impurity(score_attn).unsqueeze(-1)

    if torch.isnan(scores).any():
        raise ValueError('The Score Value has NAN after impurity computation.')

    # BACKGROUND REMOVING
    B, T_R, _ = scores.shape
    scores = scores - scores.amin(dim=1, keepdim=True)
    scores = scores / (scores.amax(dim=1, keepdim=True))
    score_mean = scores.mean(dim=1, keepdim=True)
    score_mask = scores < score_mean

    # FOREGROUND SHARPENING
    scores = scores - score_mean
    scores = scores / (scores.amax(dim=1, keepdim=True))
    scores = scores.masked_fill(score_mask, 0.0)

    return scores


def vidTLDR(x, attn, r, with_cls_token=True, use_gini=False):

    B, T, _ = x.shape
    r_merge = T - r
    r_merge = max(min(r_merge, T // 2, T), 0)
    if not r_merge:
        return x
    
    with torch.no_grad():
        if use_gini:
            score_obj = get_objective_score_gini(attn)
        else:
            score_obj = get_objective_score(attn)

    merge = merging(
        x,
        r_merge        = r_merge,
        score_obj      = score_obj,
        with_cls_token = with_cls_token,
    )

    return merge


def merging(
    metric: torch.Tensor,
    r_merge        : int,
    score_obj      : torch.Tensor,
    with_cls_token = True,
):

    with torch.no_grad():
        metric = metric / metric.norm(dim=-1, keepdim=True) # (1, 2352, 768)

        # SECTION I. TOKEN MERGING
        a, b = metric[..., ::2, :], metric[..., 1::2, :]  # (12, 99, 64), (12, 98, 64)
        n, s, t1, t2 = a.shape[0], a.shape[1], a.shape[-2], b.shape[-2]

        scores = (a @ b.transpose(-1, -2) + 1) / 2 # 0 - 1
        
        if with_cls_token:
            scores[..., 0, :] = -math.inf

        # TOKEN MERGING
        node_max, node_idx = scores.max(dim=-1) # (12, 99), (12, 99)
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None] # (12, 99, 1)
        unm_idx  = edge_idx[..., r_merge:, :]  # Unmerged Tokens (12, 83, 1)
        src_idx  = edge_idx[..., :r_merge, :]  # Merged Tokens   (12, 16, 1)
        dst_idx  = node_idx[..., None].gather(dim=-2, index=src_idx)  # (12, 16, 1)
        unm_idx  = unm_idx.sort(dim=1)[0]

        src_so = None
        if score_obj is not None:
            src_so, dst_so = score_obj[..., ::2, :], score_obj[..., 1::2, :] # (1, 1176, 1)
            src_so = src_so.gather(dim=-2, index=src_idx)  # (12, 91, 197)

    def merge(x: torch.Tensor, mode = "sum", dtype=torch.float32):
        ori_dtype = x.dtype
        x = x.to(dtype=dtype)
        src, dst = x[..., ::2, :], x[..., 1::2, :] # (12, 99, 197), (12, 98, 197)
        n, mid, c = src.shape[0], src.shape[1:-2], src.shape[-1]
        unm = src.gather(dim=-2, index=unm_idx.expand(n, *mid, t1 - r_merge, c)) # (12, 91, 197)
        src = src.gather(dim=-2, index=src_idx.expand(n, *mid, r_merge, c))
        
        if score_obj is not None:
            src = src * src_so
        
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, *mid, r_merge, c), src, reduce=mode)  # (12, 98, 197)
        
        x = torch.cat([unm, dst], dim=-2)  # (12, 1 + 180, 197)
        x = x.to(dtype=ori_dtype)
        return x

    return merge