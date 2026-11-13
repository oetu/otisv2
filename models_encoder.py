# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:             https://github.com/facebookresearch/mae
# timm:            https://github.com/huggingface/pytorch-image-models
# Register tokens: https://arxiv.org/abs/2309.16588             (Darcet et al.)
# --------------------------------------------------------

from functools import partial

import math 

import torch
import torch.nn as nn

import timm.models.vision_transformer
from timm.layers.weight_init import trunc_normal_tf_
from timm.layers.mlp import SwiGLU

from util.patch_embed import PatchEmbed
from util.transformer import Attention, DyT, AxialRoPEAttention
    

class Transformer(timm.models.vision_transformer.VisionTransformer):
    """ Transformer encoder for time series, adapted from timm's VisionTransformer.
    """
    def __init__(self, input_size, patch_size=(1, 100), global_pool=False, attention_pool=False,
                 masking_blockwise=False, mask_ratio=0.0, mask_c_ratio=0.0, mask_t_ratio=0.0,
                 num_register_tokens=2,
                 rope_theta=10000.0, qk_norm=True, attention_sink=False,
                 **kwargs):

        # SwiGLU activation instead of GELU in MLP layers, to match pre-training
        kwargs['act_layer'] = nn.SiLU
        kwargs['mlp_layer'] = SwiGLU

        super(Transformer, self).__init__(**kwargs)

        # Stash input_size / patch_size on the instance so block-wise masking
        # (which needs the patch grid (V, T)) can recover them.
        # Note: input_size = (channels, variates, time_steps)
        self.input_size = input_size
        self.patch_size = patch_size

        embed_dim = kwargs['embed_dim']
        self.patch_embed = PatchEmbed(input_size[0], patch_size, embed_dim, flatten=False) # set flatten to False

        # Register tokens (à la "Vision Transformers Need Registers", Darcet et al.)
        # Loaded from pre-trained checkpoint; participate in self-attention but ignored by the head.
        # (1, num_register_tokens, embed_dim)
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        self.masking_blockwise = masking_blockwise
        self.mask_ratio = mask_ratio
        self.mask_c_ratio = mask_c_ratio
        self.mask_t_ratio = mask_t_ratio

        if global_pool:
            self.pool = "global_pool" 
        elif attention_pool:
            self.pool = "attention_pool"
            self.attention_pool = nn.MultiheadAttention(embed_dim=kwargs['embed_dim'],
                                                        num_heads=kwargs['num_heads'], batch_first=True)
            self.q = nn.Parameter(torch.zeros(1, 1, embed_dim)) # learnable query for attention pooling
            self.attn_pool_norm = kwargs['norm_layer'](embed_dim)
            self.attn_pool_mlp = SwiGLU(embed_dim, int(embed_dim * kwargs.get('mlp_ratio', 4)), act_layer=nn.SiLU)
        else:
            self.pool = False

        if self.pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        
        # qk_norm must match the pre-trained checkpoint: it is parameter-free
        # (F.rms_norm without weight), so a mismatch is invisible to
        # load_state_dict yet silently changes the softmax scale
        # (1.0 with QK-norm vs head_dim^-0.5 without). All OTISv2 pre-training
        # runs default to qk_norm=False (main_pretrain.py --qk_norm), so
        # downstream entry points must pass it through explicitly.
        for block in self.blocks:
            block.forward = self._block_forward_wrapper(block)
            block.attn = AxialRoPEAttention(kwargs['embed_dim'], kwargs['num_heads'], qkv_bias=kwargs['qkv_bias'],
                                            rope_theta=rope_theta,
                                            num_special_tokens=1 + self.num_register_tokens,
                                            qk_norm=qk_norm,
                                            attention_sink=attention_sink)

        self.initialise_weights()

    def _block_forward_wrapper(self, block_obj):
        """
        Wraps the block forward to accept V, T, v_pos, t_pos for axial RoPE.
        """
        def my_forward(x, V=None, T=None, v_pos=None, t_pos=None, attn_mask=None):
            x = x + block_obj.drop_path1(block_obj.ls1(block_obj.attn(block_obj.norm1(x), attn_mask=attn_mask, V=V, T=T, v_pos=v_pos, t_pos=t_pos)))
            x = x + block_obj.drop_path2(block_obj.ls2(block_obj.mlp(block_obj.norm2(x))))
            return x

        return my_forward

    def initialise_weights(self):
        # initialise cls and register tokens
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.register_tokens, std=.02)

        # initialise attention pool query (aligned with timm AttentionPoolLatent)
        if hasattr(self, 'q'):
            trunc_normal_tf_(self.q, std=self.q.shape[-1] ** -0.5)

        # initialise patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = math.ceil(L * (10 - 10 * mask_ratio)/10) # factor 10 to compensate float precision 
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def random_masking_blockwise(self, x, mask_c_ratio, mask_t_ratio):
        """
        Time series of shape (N, 1, V, T), where V (variates) and T (time) are masked separately.
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        Returns x_masked, v_pos, t_pos where v_pos/t_pos are the original grid
        indices for each kept patch (required for correct axial RoPE positions).
        """
        N, L, D = x.shape  # batch, length, dim
        V, T = int(self.input_size[-2] / self.patch_size[-2]), int(self.input_size[-1] / self.patch_size[-1])

        # mask V (variates)
        x = x.reshape(N, V, T, D)
        len_keep_V = int(V * (1 - mask_c_ratio))
        noise = torch.rand(N, V, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep_V = ids_shuffle[:, :len_keep_V]   # (N, len_keep_V) — original variate indices
        index = ids_keep_V.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, T, D)
        x = torch.gather(x, dim=1, index=index)    # (N, len_keep_V, T, D)

        # mask T (time)
        x = x.permute(0, 2, 1, 3)  # (N, T, len_keep_V, D)
        len_keep_T = int(T * (1 - mask_t_ratio))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep_T = ids_shuffle[:, :len_keep_T]   # (N, len_keep_T) — original temporal indices
        index = ids_keep_T.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_V, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0, 2, 1, 3)    # (N, len_keep_V, len_keep_T, D)
        x_masked = x_masked.reshape(N, len_keep_V * len_keep_T, D)

        # Compute original grid positions for each kept patch.
        # After reshape, patch at index i has variate ids_keep_V[b, i // len_keep_T]
        # and temporal index ids_keep_T[b, i % len_keep_T].
        # (N, len_keep_V, len_keep_T) -> (N, len_keep_V * len_keep_T)
        v_pos = ids_keep_V.unsqueeze(-1).expand(-1, -1, len_keep_T).reshape(N, -1)
        t_pos = ids_keep_T.unsqueeze(1).expand(-1, len_keep_V, -1).reshape(N, -1)

        return x_masked, v_pos, t_pos

    def forward_features(self, x, attn_mask=None):
        """
        x: [B=N, 1, V, T], sequence
        attn_mask: optional [B, V', T'] patch-level padding mask (1 = real),
            e.g. from collate_fn_ft when samples of a batch differ in length.
            Padding patches are then excluded from attention and pooling.
        """
        # embed patches
        # (B, D, V', T')
        x = self.patch_embed(x)

        B, D, V, T = x.shape
        # flatten
        # (B, N, D), with N=V'*T'
        x = x.flatten(2).transpose(1, 2)

        # (B, N) validity of each patch in grid order (None = all real)
        valid = attn_mask.flatten(1).to(x.dtype) if attn_mask is not None else None

        # masking is a training-time augmentation only: eval sees all patches
        if self.training and self.masking_blockwise:
            x, v_pos, t_pos = self.random_masking_blockwise(x, self.mask_c_ratio, self.mask_t_ratio)
            if valid is not None:
                valid = valid.gather(1, v_pos * T + t_pos)
        elif self.training and self.mask_ratio > 0:
            x, _, ids_restore = self.random_masking(x, self.mask_ratio)
            # Recover the original grid positions for each kept patch so that
            # axial RoPE assigns the correct (variate, time) coordinates.
            N_vis = x.shape[1]
            ids_keep = torch.argsort(ids_restore, dim=1)[:, :N_vis]  # (B, N_vis)
            v_pos = ids_keep // T   # (B, N_vis)
            t_pos = ids_keep % T    # (B, N_vis)
            if valid is not None:
                valid = valid.gather(1, ids_keep)
        else:
            v_pos = None
            t_pos = None

        # attention mask over [CLS, registers, patches]; None keeps the unmasked path
        attn_mask_tokens = None
        if valid is not None and not bool(valid.all()):
            S = 1 + self.num_register_tokens
            attn_mask_tokens = torch.cat((torch.ones(B, S, device=x.device, dtype=valid.dtype), valid), dim=1)

        # append cls token
        # (B, 1, D)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        # append register tokens
        # (B, R, D)
        reg_tokens = self.register_tokens.expand(x.shape[0], -1, -1)

        # (B, 1+R+N, D) -> [CLS, Register, Patches]
        x = torch.cat((cls_token, reg_tokens, x), dim=1)

        x = self.pos_drop(x)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, V=V, T=T, v_pos=v_pos, t_pos=t_pos, attn_mask=attn_mask_tokens)

        if self.pool == "attention_pool":
            # (B, 1, D)
            q = self.q.expand(x.shape[0], -1, -1)
            k = x[:, :, :] # includes cls, register, and all patches
            v = x[:, :, :] # includes cls, register, and all patches

            # (B, 1, D); padding keys are ignored (True = ignore)
            key_padding_mask = (attn_mask_tokens == 0) if attn_mask_tokens is not None else None
            x, _ = self.attention_pool(q, k, v, key_padding_mask=key_padding_mask)

            # post-attention residual MLP (aligned with timm AttentionPoolLatent)
            x = x + self.attn_pool_mlp(self.attn_pool_norm(x))

            # (B, D)
            outcome = x.squeeze(dim=1)
        elif self.pool == "global_pool":
            # (B, D)
            patches = x[:, 1 + self.num_register_tokens:, :]  # exclude cls & register tokens
            if attn_mask_tokens is None:
                outcome = patches.mean(dim=1)
            else:
                # mean over real patches only
                w = attn_mask_tokens[:, 1 + self.num_register_tokens:].unsqueeze(-1).to(patches.dtype)
                outcome = (patches * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)
        else: # cls token
            x = self.norm(x)

            # (B, D)
            outcome = x[:, 0].contiguous()

        return outcome

    @torch.no_grad()
    def forward_features_collect_patch_norms(self, x):
        """
        Forward pass through the encoder with **all patches visible** (no
        masking), recording the per-token L2 norm of the **patch tokens** at
        every layer boundary. CLS and register tokens are intentionally
        excluded — only output / patch tokens are tracked.

        Diagnostic helper analogous to OTIS's
        ``forward_encoder_collect_patch_norms`` for tracking how patch-token
        norms evolve across layers during fine-tuning (à la "Vision
        Transformers Need Registers", Fig. 4a). Unlike ``forward_features``,
        this intentionally bypasses any random masking so the per-layer norms
        are computed on the same set of tokens for every sample / step.

        input:
            x: (B, C, V, T)

        output:
            patch_norms: list of (B, N) float tensors, length = depth + 1
                         (or depth + 2 if a final ``self.norm`` exists, i.e.
                          when no pooling is used).
            valid_mask:  (B, N) all-ones mask (the FT collate has no padding,
                         so every patch token is valid).
        """
        S = 1 + self.num_register_tokens

        # embed patches → (B, D, V', T') → (B, N, D)
        x = self.patch_embed(x)
        B, D, V, T = x.shape
        x = x.flatten(2).transpose(1, 2)
        N = x.shape[1]

        # axial positions of all patches (none are dropped)
        all_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        v_pos = all_ids // T
        t_pos = all_ids % T

        # prepend CLS + register tokens
        cls_token = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_token, reg_tokens, x), dim=1)

        x = self.pos_drop(x)

        patch_norms = [x[:, S:, :].float().norm(dim=-1)]  # pre-block-0
        for blk in self.blocks:
            x = blk(x, V=V, T=T, v_pos=v_pos, t_pos=t_pos)
            patch_norms.append(x[:, S:, :].float().norm(dim=-1))

        # Pooled variants (global / attention pool) delete ``self.norm`` and
        # apply ``fc_norm`` only to the pooled scalar — there is no per-token
        # final LayerNorm in those cases, so we stop after the last block.
        if hasattr(self, "norm"):
            x = self.norm(x)
            patch_norms.append(x[:, S:, :].float().norm(dim=-1))

        valid_mask = torch.ones((B, N), device=x.device)
        return patch_norms, valid_mask

    def forward_head(self, x, pre_logits: bool = False):
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)
    
    def forward(self, x, attn_mask=None):
        x = self.forward_features(x, attn_mask=attn_mask)
        x = self.forward_head(x)
        return x


def otis_base(**kwargs):
    model = Transformer(
        embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_small(**kwargs):
    model = Transformer(
        embed_dim=256, depth=14, num_heads=4, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_large(**kwargs):
    model = Transformer(
        embed_dim=320, depth=24, num_heads=5, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_huge(**kwargs):
    model = Transformer(
        embed_dim=448, depth=36, num_heads=7, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model