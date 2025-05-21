import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

import torch.utils.checkpoint as checkpoint
from functools import partial
from einops import rearrange

from .pos_embed import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from .flash_attention_class import FlashAttention
from flash_attn.modules.mlp import FusedMLP
from flash_attn.ops.rms_norm import DropoutAddRMSNorm
from .vid_tldr import vidTLDR
import einops


class DepthwiseConv3D(nn.Module):
    def __init__(self, in_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=0):
        super(DepthwiseConv3D, self).__init__()
        self.depthwise_conv = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, 
                                        stride=stride, padding=padding, groups=in_channels)
        
        # Zero-initialize weights and biases
        nn.init.constant_(self.depthwise_conv.weight, 1.)
        if self.depthwise_conv.bias is not None:
            nn.init.constant_(self.depthwise_conv.bias, 0.)

    def forward(self, x):
        dwconv = self.depthwise_conv(x)
        return dwconv


class CrossAttention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None, out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5
        assert all_head_dim == dim, 'Assertion Failed at line 31'
        
        self.q = nn.Linear(dim, all_head_dim, bias=False)
        self.k = nn.Linear(dim, all_head_dim, bias=False)
        self.v = nn.Linear(dim, all_head_dim, bias=False)
        
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.k_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x, k=None, v=None):
        B, N, C = x.shape
        N_k = k.shape[1]
        N_v = v.shape[1]
        
        q_bias, k_bias, v_bias = None, None, None
        if self.q_bias is not None:
            q_bias = self.q_bias
            k_bias = self.k_bias
            v_bias = self.v_bias
        
        q = F.linear(input=x, weight=self.q.weight, bias=q_bias)
        q = q.reshape(B, N, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)  # (B, N_head, N_q, dim)
        
        k = F.linear(input=k, weight=self.k.weight, bias=k_bias)
        k = k.reshape(B, N_k, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        v = F.linear(input=v, weight=self.v.weight, bias=v_bias)
        v = v.reshape(B, N_v, 1, self.num_heads, -1).permute(2, 0, 3, 1, 4).squeeze(0)
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B, N_head, N_q, N_k)
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class AttentiveBlock(nn.Module):
    
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attn_head_dim=None, out_dim=None):
        super().__init__()
        
        self.norm1_q = norm_layer(dim)
        self.norm1_k = norm_layer(dim)
        self.norm1_v = norm_layer(dim)
        self.cross_attn = CrossAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=drop, attn_head_dim=attn_head_dim, out_dim=out_dim)
        
        if drop_path > 0.:
            print(f"Use DropPath in projector: {drop_path}")
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    
    def forward(self, x_q, x_kv, pos_q, pos_k, bool_masked_pos, rel_pos_bias=None):
        x_q = self.norm1_q(x_q + pos_q)
        x_k = self.norm1_k(x_kv + pos_k)
        x_v = self.norm1_v(x_kv)
        x = self.cross_attn(x_q, k=x_k, v=x_v)
        
        return x


class AttentionPoolingBlock(AttentiveBlock):
    
    def forward(self, x):
        x_q = x.mean(1, keepdim=True)
        x_kv, pos_q, pos_k = x, 0, 0
        x = super().forward(x_q, x_kv, pos_q, pos_k, bool_masked_pos=None, rel_pos_bias=None)
        x = x.squeeze(1)
        return x


class TemporalAttentionPoolingBlock(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.attention_pooling = AttentionPoolingBlock(**kwargs)

    def generate_mask(self, B, T, T_new):
        assert T_new <= T, f"T_new must be less than or equal to T, but we have T_new:{T_new} and T:{T}"
        
        # Randomly choose T_new indices for each batch element
        indices = np.tile(np.random.choice(T, T_new, replace=False), B).reshape(B, T_new)
        
        # Create a boolean mask
        mask = np.zeros((B, T), dtype=bool)
        
        # Use advanced indexing to set the selected indices to True
        np.put_along_axis(mask, indices, True, axis=1)

        return mask
    
    def forward(self, x, T, T_new):
        '''
        @ args:
            x: Tensor shaped [B, T*L + 1, C]
        '''
        if T == T_new:
            return x
        cls_token, x = x[:,0,:].unsqueeze(1), x[:, 1:, :]
        B, TL, C = x.shape
        L = TL // T
        x = x.view(B, T, L, C)
        T_masks = torch.from_numpy(self.generate_mask(B, T, T_new))
        x_origin = x[T_masks].view(B, T_new, L, C)
        x_merging = x[~T_masks].view(B, T - T_new, L, C)
        x_merging = x_merging.view(B * (T - T_new), L, C)
        x_merged = self.attention_pooling(x_merging)
        x_merged = x_merged.view(B, T - T_new, C)
        x_new = torch.cat((cls_token, x_origin.view(B, -1, C), x_merged), dim=1)
        # print(x_new.shape)
        return x_new


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    
    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class AdaptiveRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super(AdaptiveRMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.alpha = nn.Parameter(torch.tensor(-20.0))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        normed_x = x / rms
        alpha = torch.sigmoid(self.alpha)
        
        return alpha * (self.weight * normed_x) + (1 - alpha) * x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False, force_fp32=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        self.force_fp32 = force_fp32
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if self.force_fp32:
            output_type = x.dtype
            out = x.float().mul_(self.gamma.float()) if self.inplace else x.float() * self.gamma.float()
            return out.to(dtype=output_type)
        else:
            out = x.mul_(self.gamma) if self.inplace else x * self.gamma
            return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_flash_attn=False,
                 causal=False, norm_layer=nn.LayerNorm, qk_normalization=False, use_fused_rmsnorm=False, use_lpe=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.use_flash_attn = use_flash_attn
        if use_flash_attn:
            self.causal = causal
            self.inner_attn = FlashAttention(attention_dropout=attn_drop)
        
        self.use_lpe = use_lpe
        
        if use_lpe:
            self.lpe = DepthwiseConv3D(dim)
        
        self.qk_normalization = qk_normalization
        self.q_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.k_norm = norm_layer(dim) if qk_normalization else nn.Identity()
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def _naive_attn(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        
        if self.qk_normalization:
            B_, H_, N_, D_ = q.shape
            q = self.q_norm(q.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
            k = self.k_norm(k.transpose(1, 2).flatten(-2, -1)).view(B_, N_, H_, D_).transpose(1, 2)
        
        if self.use_lpe:
            v = v.permute(0, 2, 1, 3).reshape(B, N, -1)
            lpe = v.contiguous().permute(0, 2, 1)
            B, C, N = lpe.shape
            lpe = lpe.reshape(B, C, 1, N, 1)
            v_lpe = self.lpe(lpe).squeeze().permute(0, 2, 1)
            v = v.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        
        attn = ((q * self.scale) @ k.transpose(-2, -1))
        # attn = attn - attn.max(-1)[0].unsqueeze(-1)  # in case of overflow for fp16
        attn = attn.softmax(dim=-1)
        if return_attn:
            attn_ = attn.detach()
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.use_lpe:
            x = self.proj(x + v_lpe)
        else:
            x = self.proj(x)
        x = self.proj_drop(x)
        if return_attn:
            return x, attn_
        return x
    
    def _flash_attn(self, x, key_padding_mask=None, need_weights=False):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = rearrange(qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads)
        
        if self.qk_normalization:
            q, k, v = qkv.unbind(2)
            if self.use_fused_rmsnorm:
                q = self.q_norm(q.flatten(-2, -1))[0].view(q.shape)
                k = self.k_norm(k.flatten(-2, -1))[0].view(k.shape)
            else:
                q = self.q_norm(q.flatten(-2, -1)).view(q.shape)
                k = self.k_norm(k.flatten(-2, -1)).view(k.shape)
            qkv = torch.stack([q, k, v], dim=2)
        
        if self.use_lpe:
            q, k, v = qkv.unbind(2)
            lpe = v.reshape(B, N, -1).contiguous().permute(0, 2, 1)
            B, C, N = lpe.shape
            lpe = lpe.reshape(B, C, 1, N, 1)
            lpe = self.lpe(lpe).squeeze().permute(0, 2, 1)
        
        context, _ = self.inner_attn(
            qkv, key_padding_mask=key_padding_mask, need_weights=need_weights, causal=self.causal
        )
        
        if self.use_lpe:
            outs = self.proj(rearrange(context, "b s h d -> b s (h d)") + lpe)
        else:
            outs = self.proj(rearrange(context, "b s h d -> b s (h d)"))
        
        outs = self.proj_drop(outs)
        return outs, qkv
    
    def forward(self, x, return_attn=False):
        x = self._naive_attn(x, return_attn=return_attn) if not self.use_flash_attn else self._flash_attn(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU,
                 bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_fused_rmsnorm=False, use_lpe=False):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
                              qk_normalization=qk_normalization,
                              use_fused_rmsnorm=use_fused_rmsnorm,
                              use_lpe=use_lpe
                            )
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def forward(self, x, residual=None, num_merging_to=None):
        
        def _inner_forward(x, residual=None):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x
        
        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual)
        else:
            return _inner_forward(x, residual=residual)


class PatchEmbed(nn.Module):
    """ 3D Image to Patch Embedding
    """
    
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, 
            num_frames=8, tubelet_size=1, norm_layer=None, dual_norm_in_patch_embed=False,
        ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.grid_size = (
            num_frames // tubelet_size, 
            img_size[0] // patch_size[0], 
            img_size[1] // patch_size[1]
        ) # (T, H, W)
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(
            in_channels=in_chans, out_channels=embed_dim, 
            kernel_size=(tubelet_size, patch_size[0], patch_size[1]), 
            stride=(tubelet_size, patch_size[0], patch_size[1])
        )
        self.dual_norm_in_patch_embed = dual_norm_in_patch_embed and norm_layer
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.norm_before = norm_layer(tubelet_size * math.prod(patch_size) * 3) if dual_norm_in_patch_embed and norm_layer else nn.Identity()
    
    def forward(self, x):
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1)
        x = einops.rearrange(x, "b (t1 t2) (ht hp) (wt wp) c -> b (t1 ht wt) (t2 hp wp c)", t2=self.tubelet_size, hp=self.patch_size[0], wp=self.patch_size[1])
        x = self.norm_before(x) # x.shape: [B, T, HW, C]
        x = einops.rearrange(x, "b (t1 ht wt) (t2 hp wp c) -> b (t1 t2) (ht hp) (wt wp) c", t1=T//self.tubelet_size, ht=H//self.patch_size[0], t2=self.tubelet_size, hp=self.patch_size[0], wp=self.patch_size[1])
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)
        x = x.flatten(3).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x


class Linear_Decoder(nn.Module):
    def __init__(self, in_channels=1408, out_channels=3200, 
                 norm_layer=nn.LayerNorm, norm_type='l2'):
        super().__init__()
        self.norm_type = norm_type
        print(f'Normalization Type: {norm_type}')

        self.head = nn.Linear(in_channels, out_channels)
        self.norm = norm_layer(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))

        if self.norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        return x


class MLP_Decoder(nn.Module):
    def __init__(self, in_channels=768, out_channels=768, 
                 norm_layer=nn.LayerNorm, norm_type='l2'):
        super().__init__()
        self.norm_type = norm_type
        print(f'Normalization Type: {norm_type}')

        self.head = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, out_channels)
        )
        self.norm = norm_layer(out_channels)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.norm(self.head(x))

        if self.norm_type == 'l2':
            x = x / x.norm(dim=-1, keepdim=True)
        elif self.norm_type == 'none':
            pass
        else:
            raise NotImplementedError

        return x


class VidTLDRBlock(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_flash_attn=False, use_fused_mlp=False,
            fused_mlp_heuristic=1, with_cp=False, qk_normalization=False, layerscale_no_force_fp32=False,
            use_fused_rmsnorm=False, index=None, use_lpe=False):
        super().__init__()
        
        self.index = index
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                              use_flash_attn=use_flash_attn, causal=False, norm_layer=norm_layer,
                              qk_normalization=qk_normalization,
                              use_fused_rmsnorm=use_fused_rmsnorm, 
                              use_lpe=use_lpe)
        self.ls1 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if use_fused_mlp:
            self.mlp = FusedMLP(in_features=dim, hidden_features=mlp_hidden_dim, heuristic=fused_mlp_heuristic)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values,
                              force_fp32=(not layerscale_no_force_fp32)) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.with_cp = with_cp
        self.use_fused_rmsnorm = use_fused_rmsnorm
    
    def forward(self, x, residual=None, num_merging_to=None, return_attn=False):

        def _inner_forward(x, residual=None, num_merging_to=None, return_attn=False):
            if self.use_fused_rmsnorm:
                x, residual = self.norm1(x, residual)
                x = self.drop_path1(self.ls1(self.attn(x)))
                x, residual = self.norm2(x, residual)
                x = self.drop_path2(self.ls2(self.mlp(x)))
                return x, residual
            else:
                assert residual is None
                x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
                x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
                return x
        
        if self.with_cp:
            return checkpoint.checkpoint(_inner_forward, x, residual, num_merging_to, return_attn)
        else:
            return _inner_forward(x, residual=residual, num_merging_to=num_merging_to, return_attn=return_attn)


class FluxViT(nn.Module):
    def __init__(
            self,
            in_chans: int = 3,
            patch_size: int = 14,
            img_size: int = 224,
            qkv_bias: bool = False,
            drop_path_rate: float = 0.25,
            embed_dim: int = 384,
            head_drop_path_rate: float = 0.,
            num_heads: int = 16,
            mlp_ratio: float = 4,
            init_values: float = 1e-5,
            qk_normalization: bool = True,
            depth: int = 12,
            use_flash_attn: bool = True,
            use_fused_rmsnorm: bool = True,
            use_fused_mlp: bool = True,
            fused_mlp_heuristic: int = 1,
            attn_pool_num_heads: int = 16,
            clip_embed_dim: int = 768,
            layerscale_no_force_fp32: bool = False,
            num_frames: int = 8,
            tubelet_size: int = 1,
            sep_pos_embed: bool = False,
            use_checkpoint: bool = False,
            checkpoint_num: int = 0,
            fc_drop_rate: float = 0., 
            num_classes: int = 1000, 
            mix_tokens: bool = False,
            clip_return_layer: int = 0,
            clip_student_return_interval: int = 1,
            dual_norm_in_patch_embed = False,
            mcm_keep_first = False,
            use_gpe_proj = False,
            use_lpe = False,
        ):
        super().__init__()
        
        assert use_flash_attn == use_fused_rmsnorm == use_fused_mlp, print(
            'use_flash_attn, use_fused_rmsnorm and use_fused_mlp should be consistent')
        print(mlp_ratio)
        self.tubelet_size = tubelet_size
        self.dual_norm_in_patch_embed = dual_norm_in_patch_embed
        self.mix_tokens = mix_tokens
        self.use_flash_attn = use_flash_attn
        self.embed_dim = embed_dim
        self.mcm_keep_first = mcm_keep_first
        self.clip_return_index = []
        for i in range(clip_return_layer):
            self.clip_return_index.append(depth - int(i * clip_student_return_interval) - 1)
        
        flash_norm_layer_for_blocks = partial(DropoutAddRMSNorm, eps=1e-6, prenorm=True)
        norm_layer_for_blocks = partial(RMSNorm, eps=1e-6)
        
        self.gpe_proj = DepthwiseConv3D(embed_dim)    

        self.num_frames = num_frames
        self.norm_layer_for_blocks = norm_layer_for_blocks
        if dual_norm_in_patch_embed:
            self.patch_embed = PatchEmbed(
                img_size, patch_size, in_chans, embed_dim,
                num_frames=num_frames, tubelet_size=tubelet_size, norm_layer=partial(RMSNorm, eps=1e-6), dual_norm_in_patch_embed=dual_norm_in_patch_embed,
            )
        else:
            self.patch_embed = PatchEmbed(
                img_size, patch_size, in_chans, embed_dim,
                num_frames=num_frames, tubelet_size=tubelet_size
            )
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # stolen from https://github.com/facebookresearch/mae_st/blob/dc072aaaf640d06892e23a33b42223a994efe272/models_vit.py#L65-L73C17
        self.sep_pos_embed = sep_pos_embed
        if sep_pos_embed:
            print("Use seperable position embedding")
            grid_size = self.patch_embed.grid_size
            self.grid_size = grid_size
            self.pos_embed_spatial = nn.Parameter(torch.zeros(1, grid_size[1] * grid_size[2], embed_dim))
            self.pos_embed_temporal = nn.Parameter(torch.zeros(1, grid_size[0], embed_dim))
            self.pos_embed_cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            print("Use joint position embedding")
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        # choose which layer to use checkpoint
        with_cp_list = [False] * depth
        if use_checkpoint:
            for idx in range(depth):
                if idx < checkpoint_num:
                    with_cp_list[idx] = True
        print(f"Droppath rate: {dpr}")
        print(f"Checkpoint list: {with_cp_list}")
        
        self.blocks = nn.ModuleList([
            VidTLDRBlock(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias,
                  norm_layer=flash_norm_layer_for_blocks if use_flash_attn else norm_layer_for_blocks,
                  drop_path=dpr[i], init_values=init_values, attn_drop=0.,
                  use_flash_attn=use_flash_attn, 
                  use_fused_mlp=use_fused_mlp,
                  fused_mlp_heuristic=fused_mlp_heuristic,
                  with_cp=with_cp_list[i],
                  qk_normalization=qk_normalization,
                  layerscale_no_force_fp32=layerscale_no_force_fp32,
                  use_fused_rmsnorm=use_fused_rmsnorm,
                  index=i,
                  use_lpe=use_lpe)
            for i in range(depth)]
        )

        # we use multiple projectors and heads for different token counts
        self.clip_projector = nn.ModuleList(
            [AttentionPoolingBlock(
            dim=embed_dim, num_heads=attn_pool_num_heads, qkv_bias=True, qk_scale=None,
            drop=0., attn_drop=0., drop_path=head_drop_path_rate, 
            norm_layer=partial(nn.LayerNorm, eps=1e-5), out_dim=clip_embed_dim
            ) for i in range(3)]
        )
        self.head = nn.ModuleList([nn.Linear(clip_embed_dim, num_classes) for _ in range(3)])
        self.fc_norm = nn.LayerNorm(clip_embed_dim)

        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        
        self.final_clip_decoder_for_projs = nn.ModuleList([
            Linear_Decoder(
                in_channels=clip_embed_dim, 
                out_channels=clip_embed_dim, 
                norm_layer=partial(nn.LayerNorm, eps=1e-5), 
                norm_type='l2'
            )for _ in range(2)]
        )

        self.init_pos_embed()
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def init_pos_embed(self):
        print("Init pos_embed from sincos pos_embed")
        if self.sep_pos_embed:
            pos_embed_spatial = get_2d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
            )
            self.pos_embed_spatial.data.copy_(torch.from_numpy(pos_embed_spatial).float().unsqueeze(0))
            pos_embed_temporal = get_1d_sincos_pos_embed(
                self.pos_embed_spatial.shape[-1], 
                self.patch_embed.grid_size[0], # t_size
            )
            self.pos_embed_temporal.data.copy_(torch.from_numpy(pos_embed_temporal).float().unsqueeze(0))
        else:
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1], 
                self.patch_embed.grid_size[1], # height & weight
                self.patch_embed.grid_size[0], # t_size
                cls_token=True
            )
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
    
    @property
    def dtype(self):
        return self.patch_embed.proj.weight.dtype

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed', 
            'pos_embed_spatial', 
            'pos_embed_temporal', 
            'pos_embed_cls',
            'cls_token'
        }
    
    def sparse_group_token_selection(self, x, B, T, L, C, masking_ratio):
        patch_embed_vectors = x.detach().clone()

        assert T % 2 == 0, 'We here only consider an input frame count divisible by 2'

        # We here use a simplified version for number of groups
        if T % 4 == 0:
            num_groups = 4
        else:
            if T == 2:
                num_groups = 1
            else:
                num_groups = 2

        # divide into sparse groups
        patch_embed_vectors = patch_embed_vectors.reshape(B, num_groups, T // num_groups, L, C).reshape(B * num_groups, T // num_groups, L, C)
        T = T // num_groups
        B = B * num_groups

        # compute L2 - distance
        distance = torch.norm(patch_embed_vectors[:,:T-1,:,:] - patch_embed_vectors[:,1:,:,:], p=2, dim=3)
        importance = torch.cat((distance[:,0,:], distance.flatten(1)), dim=1)
        ids_sorted = torch.argsort(importance, dim=1, descending=True) 
        num_input_tokens = int((1 - masking_ratio) * (T * L))
        ids_restore = torch.argsort(ids_sorted, dim=1)
        input_mask = torch.ones([B, T * L], device=x.device)
        input_mask[:, :num_input_tokens] = 0            
        input_mask = torch.gather(input_mask, dim=1, index=ids_restore)
        
        # restore to previous input shape
        T = T * num_groups
        B = B // num_groups
        input_mask = input_mask.reshape(B, T * L)
        
        return input_mask.to(torch.bool)
    
    
    def expand_pos_embed(self, pos_embed, new_t_size, L):
        '''
        @param: 
            pos_embed: original pos_embed, (1, T*L + 1, embed_dim)
            T: frames
            L: w * h
            method: interpolation method
        '''
        pos_embed_checkpoint = pos_embed
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_extra_tokens = 1
        
        # height (== width) for the checkpoint position embedding
        orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(self.num_frames / self.patch_embed.tubelet_size)) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(L ** 0.5)
        
        # class_token and dist_token are kept unchanged
        if self.num_frames != new_t_size:
            # logger.info(f"Temporal interpolate from {self.num_frames} to {new_t_size} ")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> B， T, HW, C -> BHW, C, T  (B = 1)
            pos_tokens = pos_tokens.view(1, self.num_frames, -1, embedding_size)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, embedding_size, self.num_frames)
            pos_tokens = torch.nn.functional.interpolate(pos_tokens.cpu(), size=new_t_size, mode='linear').cuda()
            pos_tokens = pos_tokens.view(1, -1, embedding_size, new_t_size)
            pos_tokens = pos_tokens.permute(0, 3, 1, 2).reshape(1, -1, embedding_size)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            pos_embed_checkpoint = new_pos_embed
        
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            # logger.info(f"Position interpolate from {orig_size}x{orig_size} to {new_size}x{new_size}")
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            # B, L, C -> BT, H, W, C -> BT, C, H, W
            pos_tokens = pos_tokens.reshape(-1, new_t_size, orig_size, orig_size, embedding_size)
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens.cpu(), size=(new_size, new_size), mode='bicubic', align_corners=False).cuda()
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, new_t_size, new_size, new_size, embedding_size) 
            pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        
        return new_pos_embed
    
    def mix_tokens_fn(self, x):
        sp = x.shape
        idx = torch.randperm(x.shape[0])
        x = x[idx].view(sp)
        return x

    def create_pos_embed(self, x, T, L, C):
        cls_pos, pos_embed = self.pos_embed[:, 0, :].unsqueeze(1), self.pos_embed[:, 1:, :]
        t = self.num_frames
        h_pos_embed = int(math.sqrt(pos_embed.shape[1] / t))
        pos_embed = pos_embed.view(1, t, h_pos_embed, h_pos_embed, C).permute(0, 4, 1, 2, 3)
        pos_embed = self.gpe_proj(pos_embed).permute(0, 2, 3, 4, 1).view(1, -1, C)
        pos_embed = torch.cat((cls_pos, pos_embed), dim=1)
        if pos_embed[0].shape != x[0].shape:
            pos_embed = self.expand_pos_embed(pos_embed, T, L)
        assert pos_embed[0].shape == x[0].shape, f'pos embed shape: {pos_embed.shape} not match x[0].shape {x[0].shape}'
        return pos_embed

    def forward(self, x, num_merging_to=None, masking_ratio=None, output_head=0, return_cls=False, return_projected=False, align_proj=0):
        x = self.patch_embed(x.type(self.dtype))
        B, T, L, C = x.shape  # T: temporal; L: spatial

        # append cls token
        if masking_ratio is not None:
            with torch.no_grad():
                mask = self.sparse_group_token_selection(x, B, T, L, C, masking_ratio=masking_ratio)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = x.view([B, T * L, C])
        x = torch.cat((cls_tokens, x), dim=1)

        # create_pos_embed
        pos_embed = self.create_pos_embed(x, T, L, C)
        x = x + pos_embed

        if masking_ratio is not None:
            cls_tokens, x = x[:,0,:].unsqueeze(1), x[:,1:,:].view(B, T*L, C)
            x = x[~mask].view(B, -1, C)
            x = torch.cat((cls_tokens, x), dim=1)

        residual = None

        for idx, blk in enumerate(self.blocks):
            if isinstance(x, tuple) and len(x) == 2:
                x, residual = x
            x = blk(x, residual=residual, num_merging_to=num_merging_to[idx] if num_merging_to is not None and idx < len(num_merging_to) else None)

        if isinstance(x, tuple) and len(x) == 2:
            x, residual = x
            if residual is not None:
                x = x + residual

        x_final = self.clip_projector[output_head](x)
        x = self.fc_norm(self.fc_dropout(x_final))
        x = self.head[output_head](x)

        if return_cls:
            if return_projected:
                return x, self.final_clip_decoder_for_projs[align_proj](x_final)
            else:
                return x, x_final

        return x


@register_model
def fluxvit_small_patch14(pretrained=False, **kwargs):
    model = FluxViT(
        patch_size=14, embed_dim=384, 
        img_size=252, num_frames=24, # doesn't matter, just for loading
        depth=12, num_heads=6, mlp_ratio=4,
        attn_pool_num_heads=16, clip_embed_dim=768, 
        use_gpe_proj=True, use_lpe=True, dual_norm_in_patch_embed=True,
        **kwargs
    )
    return model


@register_model
def fluxvit_base_patch14(pretrained=False, **kwargs):
    model = FluxViT(
        patch_size=14, embed_dim=768,
        img_size=280, num_frames=32, # doesn't matter, just for loading
        depth=12, num_heads=12, mlp_ratio=4,
        attn_pool_num_heads=16, clip_embed_dim=768, 
        use_gpe_proj=True, use_lpe=True, dual_norm_in_patch_embed=True,
        **kwargs
    )
    return model