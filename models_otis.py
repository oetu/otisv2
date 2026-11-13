# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:             https://github.com/facebookresearch/mae
# timm:            https://github.com/huggingface/pytorch-image-models
# iBOT:            https://github.com/bytedance/ibot            (masked patch prediction;
#                                                                we use a subset-mask teacher
#                                                                variant — teacher sees a
#                                                                superset of the student's
#                                                                visible tokens. The L_MIM
#                                                                loss is applied at VISIBLE
#                                                                student positions, not at
#                                                                the masked ones, since the
#                                                                teacher has matched features
#                                                                there.)
# DINOv2:          https://github.com/facebookresearch/dinov2   (KoLeo)
# DINOv3:          https://github.com/facebookresearch/dinov3   (Gram anchoring)
# Register tokens: https://arxiv.org/abs/2309.16588             (Darcet et al.)
# MUON:            https://github.com/KellerJordan/Muon
# --------------------------------------------------------

from functools import partial

import copy
import random
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

import numpy as np

from timm.models.vision_transformer import Block
from timm.layers.weight_init import trunc_normal_
from timm.layers.mlp import Mlp, SwiGLU

from util.patch_embed import PatchEmbed
import util.statistics as statistics
from util.transformer import AxialRoPEAttention, DyT


class DINOHead(nn.Module):
    """
    iBOT / DINO projection head: 3-layer MLP (in_dim -> hidden -> hidden -> bottleneck),
    L2-normalisation at the bottleneck, then a weight-normalised linear projection
    onto ``out_dim`` prototypes. The magnitude of the final layer (``weight_g``) is
    pinned to 1, so the final op is a cosine similarity with ``out_dim`` learnable
    unit-norm prototypes.
    """
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)
        # Cosine similarity with unit-norm prototypes (DINO/iBOT/DINOv2 style).
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1.0)
        self.last_layer.weight_g.requires_grad = False
        self.out_dim = out_dim

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class OTIS(nn.Module):
    """
        Towards generalisable time series representations with tiny encoders
    """
    def __init__(self, input_channels=1, time_steps=2500, patch_size=(1, 100),
                 embed_dim=1024, depth=24, num_heads=16,
                 num_register_tokens=2,
                 output_projection='decoder',
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 head_mlp_ratio=4., head_dropout=0.1, head_activation=nn.GELU,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, drop_path=0.1, drop_path_decoder=0.0,
                 rope_theta=10000.0, qk_norm=True, attention_sink=False,
                 swiglu=True, zero_init_residual=True,
                 layer_scale=None,
                 norm_patch_loss=False, masked_patch_loss=False,
                 self_distillation_loss=False,
                 ibot_out_dim=8192, ibot_teacher_temp=0.04, ibot_student_temp=0.1,
                 ibot_center_momentum=0.9, ibot_ema_decay=0.996,
                 ibot_teacher_mask_ratio=0.5,
                 ibot_teacher_real_superset=False,
                 ibot_teacher_forecast_mask_ratio=None,
                 ibot_teacher_drop_path=True,
                 ibot_freeze_cls_head_only=True,
                 ibot_share_head=False,
                 gram_enabled=False,
                 probabilistic_masking=False,
                 include_forecasting=False, forecasting_probability=0.25, forecasting_mask_ratio=0.5,
                 downstream=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # OTISv2 encoder specifics
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(input_channels, patch_size, embed_dim, flatten=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        max_num_patches_x = time_steps // patch_size[1]
        self.max_num_patches_x = max_num_patches_x

        # Register tokens (à la "Vision Transformers Need Registers", Darcet et al.)
        # Domain-agnostic learnable scratch slots that participate in self-attention.
        # (1, num_register_tokens, embed_dim)
        self.num_register_tokens = num_register_tokens
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim))

        self.zero_init_residual = zero_init_residual
        mlp_layer = SwiGLU if swiglu else Mlp
        act_layer = nn.SiLU if swiglu else nn.GELU

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  act_layer=act_layer, mlp_layer=mlp_layer, drop_path=drop_path, init_values=layer_scale)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # modify the attention operation to use AxialRoPE and consider attention masks
        # Encoder sees CLS + register tokens as leading "special" tokens that bypass RoPE.
        for block in self.blocks:
            block.forward = self._block_forward_wrapper(block)
            block.attn = AxialRoPEAttention(embed_dim, num_heads, qkv_bias=True, rope_theta=rope_theta,
                                            num_special_tokens=1 + self.num_register_tokens,
                                            qk_norm=qk_norm,
                                            attention_sink=attention_sink)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # OTISv2 output projection specifics
        self.output_projection = output_projection
        
        if self.output_projection == 'mlp':
            self.mask_token_encoder = nn.Parameter(torch.zeros(1, 1, embed_dim))

            self.mlp = Mlp(
                in_features=embed_dim,
                hidden_features=int(embed_dim * head_mlp_ratio),
                act_layer=head_activation,
                drop=head_dropout,
            )

            self.mlp_norm = norm_layer(embed_dim)
            self.mlp_pred = nn.Linear(embed_dim, patch_size[0] * patch_size[1] * input_channels, bias=True)
        else: # decoder
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                      act_layer=act_layer, mlp_layer=mlp_layer, drop_path=drop_path_decoder, init_values=layer_scale)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * input_channels, bias=True) # decoder to patch

            # modify the attention operation to use AxialRoPE and consider attention masks
            # Decoder also keeps CLS + register tokens at the front; they bypass RoPE.
            for block in self.decoder_blocks:
                block.forward = self._block_forward_wrapper(block)
                block.attn = AxialRoPEAttention(decoder_embed_dim, decoder_num_heads, qkv_bias=True, rope_theta=rope_theta,
                                                num_special_tokens=1 + self.num_register_tokens,
                                                qk_norm=qk_norm,
                                                attention_sink=attention_sink)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # Self-distillation (iBOT L_CLS + L_MIM) specifics.
        # Following DINOv3, a unified weight (``sd_weight`` in the engine) governs
        # both the CLS and MIM branches, so L_MIM is always active whenever
        # self-distillation is enabled. L_MIM is applied at the student's VISIBLE
        # token positions (not at the masked ones, as in vanilla iBOT) via the
        # subset-mask teacher: the teacher sees every student-visible token at
        # matched positions.
        self.self_distillation_loss = self_distillation_loss
        self.ibot_out_dim = ibot_out_dim
        self.ibot_teacher_temp = ibot_teacher_temp
        self.ibot_student_temp = ibot_student_temp
        self.ibot_center_momentum = ibot_center_momentum
        self.ibot_ema_decay = ibot_ema_decay
        self.ibot_teacher_mask_ratio = ibot_teacher_mask_ratio
        # True: teacher view = student's real patches + extra REAL patches
        # (_teacher_masking_real_superset). False: a longer prefix of the
        # student's noise order (_teacher_masking_from_student).
        self.ibot_teacher_real_superset = ibot_teacher_real_superset
        # Teacher mask ratio for forecasting views (real-superset builder only).
        # None = use ibot_teacher_mask_ratio for both masking schemes.
        if ibot_teacher_forecast_mask_ratio is not None and not ibot_teacher_real_superset:
            raise ValueError("ibot_teacher_forecast_mask_ratio requires ibot_teacher_real_superset")
        self.ibot_teacher_forecast_mask_ratio = ibot_teacher_forecast_mask_ratio
        # ibot_teacher_drop_path: True = the teacher keeps the student's
        # stochastic depth; False = the teacher runs without it.
        # ibot_freeze_cls_head_only: True = the first-epoch last-layer freeze
        # covers only the [CLS] head; False = it also covers the patch head.
        self.ibot_teacher_drop_path = ibot_teacher_drop_path
        self.ibot_freeze_cls_head_only = ibot_freeze_cls_head_only
        # Masking mode of the latest random_masking call (True = forecasting).
        # Read by the SD teacher to build its view.
        self._last_mask_forecast = False
        self.ibot_share_head = ibot_share_head

        if self_distillation_loss:
            # Student projection head (iBOT DINOHead). Teacher submodules are
            # built *after* ``initialise_weights()`` runs, so they snapshot the
            # fully-initialised student weights (see bottom of __init__).
            self.student_head = DINOHead(embed_dim, ibot_out_dim)
            self.register_buffer("center", torch.zeros(1, ibot_out_dim))
            self._last_teacher_cls_logits = None

            if not self.ibot_share_head:
                # DINOv3 default: a separate MIM head for patch-token distillation
                # (cfg.ibot.separate_head = True). Sharing with the CLS head is
                # supported for ablation via ``ibot_share_head=True``.
                self.student_mim_head = DINOHead(embed_dim, ibot_out_dim)

            self.register_buffer("patch_center", torch.zeros(1, ibot_out_dim))
            self._last_teacher_patch_logits = None
        # --------------------------------------------------------------------------

        self.norm_patch_loss = norm_patch_loss
        self.masked_patch_loss = masked_patch_loss

        self.probabilistic_masking = probabilistic_masking
        self.include_forecasting = include_forecasting
        self.forecasting_probability = forecasting_probability
        self.forecasting_mask_ratio = forecasting_mask_ratio

        self.downstream = downstream

        self.gram_enabled = gram_enabled

        self.initialise_weights()

        if self_distillation_loss:
            self._build_teacher()

        # Gram teacher (DINOv3 §4.2). Allocated at __init__ — rather than
        # lazily at activation — so that DDP sees the parameter set at wrap
        # time. Weights are meaningless until ``refresh_gram_teacher()`` is
        # called from the training loop; the Gram loss itself is gated on
        # ``gram_active`` in the engine, so these params are unused during
        # the stable phase.
        if gram_enabled:
            if not self_distillation_loss:
                raise ValueError(
                    "gram_enabled=True requires self_distillation_loss=True "
                    "(the Gram teacher is refreshed from the iBOT EMA teacher)."
                )
            self._build_gram_teacher()

    def _build_teacher(self):
        """Snapshot the fully-initialised student encoder into teacher submodules.

        Called at the end of ``__init__`` so that the teacher sees the custom
        initialisation applied by ``initialise_weights()`` (e.g. zero-init of
        ``attn.proj`` / ``mlp.fc2``) rather than the default PyTorch init.
        Teacher params are frozen from the optimiser and tracked via EMA.
        """
        self.teacher_patch_embed = copy.deepcopy(self.patch_embed)
        self.teacher_cls_token = nn.Parameter(self.cls_token.data.clone())
        self.teacher_register_tokens = nn.Parameter(self.register_tokens.data.clone())
        self.teacher_blocks = copy.deepcopy(self.blocks)
        self.teacher_norm = copy.deepcopy(self.norm)

        # weight_norm inside DINOHead blocks deepcopy; build a fresh head
        # and copy the student's parameters across explicitly.
        self.teacher_head = DINOHead(self.student_head.mlp[0].in_features, self.ibot_out_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())

        if self.self_distillation_loss and not self.ibot_share_head:
            self.teacher_mim_head = DINOHead(self.student_mim_head.mlp[0].in_features, self.ibot_out_dim)
            self.teacher_mim_head.load_state_dict(self.student_mim_head.state_dict())

        # deepcopy captures a closure over the original block obj; re-patch
        # so each teacher block's forward references its own (deep-copied) block.
        for block in self.teacher_blocks:
            block.forward = self._block_forward_wrapper(block)
            # The deepcopy inherits the student's DropPath, which model.train()
            # activates. With ``ibot_teacher_drop_path=False`` the EMA teacher
            # runs without stochastic depth (DINO / iBOT / DINOv2).
            if not self.ibot_teacher_drop_path:
                block.drop_path1 = nn.Identity()
                block.drop_path2 = nn.Identity()

        for p in self._teacher_params():
            p.requires_grad = False

    def activate_masked_loss(self):
        self.masked_patch_loss = True

    def _block_forward_wrapper(self, block_obj):
        """
        Modified version of def forward() of class Block() in timm.models.vision_transformer
        """
        # AxialRoPE version
        def my_forward(x, attn_mask=None, V=None, T=None, v_pos=None, t_pos=None):
            x = x + block_obj.drop_path1(block_obj.ls1(block_obj.attn(block_obj.norm1(x), attn_mask, V, T, v_pos, t_pos)))
            x = x + block_obj.drop_path2(block_obj.ls2(block_obj.mlp(block_obj.norm2(x))))
            return x

        return my_forward

    def initialise_weights(self):
        # initialise register tokens
        torch.nn.init.normal_(self.register_tokens, std=.02)

        # initialise patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        if self.output_projection == "decoder":
            torch.nn.init.normal_(self.mask_token, std=.02)
        else: # mlp
            torch.nn.init.normal_(self.mask_token_encoder, std=.02)

        # initialise nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # zero-init output projections so each block starts as identity (residual-only)
        if self.zero_init_residual:
            all_blocks = list(self.blocks)
            if self.output_projection == 'decoder':
                all_blocks += list(self.decoder_blocks)
            for block in all_blocks:
                nn.init.zeros_(block.attn.proj.weight)
                nn.init.zeros_(block.mlp.fc2.weight)
                if block.attn.proj.bias is not None:
                    nn.init.zeros_(block.attn.proj.bias)
                if block.mlp.fc2.bias is not None:
                    nn.init.zeros_(block.mlp.fc2.bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, samples):
        """
        samples: (N, C, H, W)
        x: (N, L, p*q*C)
        """
        p, q = self.patch_size
        assert samples.shape[2] % p == 0 and samples.shape[3] % q == 0

        h = samples.shape[2] // p
        w = samples.shape[3] // q
        x = samples.reshape(shape=(samples.shape[0], samples.shape[1], h, p, w, q))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(samples.shape[0], h * w, p * q * samples.shape[1]))
        return x

    def unpatchify(self, x, attn_mask):
        """
        x: (N, L, p*q*C)
        attn_mask: (N, C', T'], with C'=h=H/p, T'=w=W/q, L=C'*T'
        samples: (N, C, H, W)
        """
        p, q = self.patch_size
        h, w = attn_mask.shape[1], attn_mask.shape[2]
        assert h * w == x.shape[1]
        
        input_channels = int(x.shape[2] / (p*q))

        x = x.reshape(shape=(x.shape[0], h, w, p, q, input_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        samples = x.reshape(shape=(x.shape[0], input_channels, h * p, w * q))
        return samples

    def _jitter_mask_ratio(self, base_ratio):
        """Apply probabilistic-masking jitter to a base mask ratio.

        Off  -> returns ``base_ratio`` unchanged.
        On   -> ``base_ratio + U(0, 0.15)`` (masks *more*; the base ratio is
            the floor, so it bounds the visible-token count).
        The result is clipped to [0, 1) so downstream ``len_keep`` stays valid.
        """
        if not self.probabilistic_masking:
            return base_ratio
        ratio = base_ratio + np.random.uniform(0.0, 0.15)
        return float(min(max(ratio, 0.0), 0.999))

    def random_masking(self, x, attn_mask, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        attn_mask: [N, C', T'], with L=C'*T'
        """
        N, L, D = x.shape  # batch, length, dim
        masking_ratio = self._jitter_mask_ratio(mask_ratio)
        len_keep = math.ceil(L * (10 - 10 * masking_ratio)/10) # factor 10 to compensate float precision
        
        is_forecasting = self.downstream == "forecasting" or (self.include_forecasting and np.random.random() < self.forecasting_probability)
        self._last_mask_forecast = bool(is_forecasting)
        if is_forecasting:
            if self.include_forecasting:
                # pretraining
                forecasting_ratio = self._jitter_mask_ratio(self.forecasting_mask_ratio)
            else:
                # downstream finetuning
                forecasting_ratio = mask_ratio

            # [N, C', T']
            N, nb_of_channels, nb_of_patches = attn_mask.shape

            # Generate noise
            # [C', T']
            noise = torch.arange(0, nb_of_patches, device=x.device).to(torch.float32).repeat(nb_of_channels, 1).unsqueeze(0)
            # [C', T']
            noise.add_(torch.linspace(0, 0.5, steps=nb_of_channels, device=x.device).view(-1, 1))
            # [N, C', T']
            noise = noise.repeat(N, 1, 1)
            # [N, C', T']
            noise.mul_(attn_mask)

            # Determine maximum noise value
            # [N, 1, 1]
            noise_max = noise.flatten(1).max(dim=-1)[0].view(-1, 1, 1)

            # Create auxiliary mask
            # to set values of masked patches to infinity such that they are certainly removed
            # [N, 1, 1]
            # Visible time steps per variate, as an integer per sample (float64,
            # minus a small tolerance so that e.g. 18 * (1 - 0.2222) = 14.0004
            # rounds as intended and float error never adds a step).
            t_real = attn_mask[:, 0, :].sum(dim=-1).to(torch.float64)                    # (N,)
            keep_t = torch.ceil(t_real * (1.0 - forecasting_ratio) - 1e-6).long()
            keep_t = torch.minimum(keep_t.clamp(min=1), t_real.long())                   # (N,)
            # The batch keep count is derived from the SAME integers: it equals
            # the number of non-horizon slots (visible + padding) of the sample
            # with the most horizon tokens. No sample then exposes a horizon
            # patch, and no sample loses a planned-visible patch (padding, ranked
            # after the visible patches, absorbs the difference). Two separate
            # ceil operations used to disagree here.
            v_real = attn_mask.amax(dim=-1).sum(dim=-1).long()                           # (N,)
            horizon = t_real.long() - keep_t                                             # (N,)
            len_keep = max(int((L - v_real * horizon).min().item()), 1)
            len_keeps = keep_t.view(-1, 1, 1)
            # [N, C', T']
            aux_mask = torch.arange(nb_of_patches, device=x.device).expand(N, nb_of_channels, nb_of_patches) < len_keeps
            aux_mask = 1 - aux_mask.to(torch.float32)
            aux_mask.mul_(attn_mask)
            aux_mask = torch.nan_to_num(aux_mask * torch.inf, nan=0.0)

            # Apply auxiliary mask
            # [N, C', T']
            noise.add_(aux_mask)

            # Assign random values to padding tokens such that
            # visible_patches.values < padding_tokens.values < masked_patches.values
            # [N, C', T']
            padding_noise = noise_max + torch.rand(nb_of_channels, nb_of_patches, device=x.device).unsqueeze(0)
            noise.add_((1 - attn_mask) * padding_noise)

            # Flatten noise
            # [N, L]
            noise = noise.flatten(1)
        else:
            # [N, C', T']
            N, nb_of_channels, nb_of_patches = attn_mask.shape

            # Generate noise
            # [N, C', T']
            noise = torch.rand(N, nb_of_channels, nb_of_patches, device=x.device)
            noise.mul_(attn_mask)
            noise.add_(torch.nan_to_num((1 - attn_mask) * torch.inf, nan=0.0))

            # Create auxiliary mask
            # to set values of masked patches to infinity such that they are certainly removed
            # [N, 1, 1], nb of unmasked tokens per variate
            len_keeps = torch.ceil(attn_mask[:, 0, :].sum(dim=-1) * (10-10*masking_ratio)/10).to(torch.long).view(-1, 1, 1)
            # [N, C', T']
            aux_mask = torch.arange(nb_of_patches, device=x.device).expand(N, nb_of_channels, nb_of_patches) < len_keeps
            aux_mask = 1 - aux_mask.to(torch.float32)
            aux_mask.mul_(attn_mask)
            aux_mask = torch.nan_to_num(aux_mask * torch.inf, nan=0.0)

            # Sort noise
            # [N, C', T']
            noise, aux_ids_shuffle = torch.sort(noise, dim=-1)
            aux_ids_restore = torch.argsort(aux_ids_shuffle, dim=-1)

            # Apply auxiliary mask and restore noise
            # [N, C', T']
            noise.add_(aux_mask)
            noise = torch.gather(noise, dim=-1, index=aux_ids_restore)

            # Apply attention mask
            # [N, C', T']
            noise.mul_(attn_mask)

            # Assign random values to padding tokens such that
            # visible_patches.values < padding_tokens.values < masked_patches.values
            # [N, C', T']
            padding_noise = torch.rand(1, nb_of_channels, nb_of_patches, device=x.device) + 1
            noise.add_((1 - attn_mask) * padding_noise)

            # Flatten noise
            # [N, L]
            noise = noise.flatten(1)
        
        # sort noise for each sample
        # [N, L], the first len_keep indices correspond to the visible tokens, the remaining L-len_keep to the masked tokens
        ids_shuffle = torch.argsort(noise, dim=1)   # ascend: small is keep, large is remove
        # [N, L], to restore the original token order after concatenating masked tokens to the visible tokens (before the decoding)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # [N, len_keep]
        ids_keep = ids_shuffle[:, :len_keep]
        # [N, len_keep, D]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        # [N, L]
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0

        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, attn_mask, mask_ratio):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            attn_mask: (B, C', T'), with N=C'*T' patches

        output:
            x: (B, 1+R+N', D), with 1 cls token + R register tokens + N' visible patches
            mask: (B, N), with N (visible + mask) patches, 0 is keep, 1 is remove
            ids_restore: (B, N)
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)
        B, D, V, T = x.shape

        # flatten
        # (B, N, D), with N=C'*T'
        x = x.flatten(2).transpose(1, 2)

        # masking: length -> length * mask_ratio
        # (B, N', D), with N'=C'*T'*(1-mask_ratio)
        x, mask, ids_restore = self.random_masking(x, attn_mask, mask_ratio)
        N_vis = x.shape[1]

        # Compute (v, t) position of each visible patch from ids_restore
        # ids_shuffle maps shuffled→original; the first N_vis entries are the kept patches
        ids_shuffle = torch.argsort(ids_restore, dim=1)        # (B, V*T)
        ids_keep = ids_shuffle[:, :N_vis]                       # (B, N_vis)
        v_pos = ids_keep // T                                   # (B, N_vis)
        t_pos = ids_keep % T                                    # (B, N_vis)

        # Real / padding indicator for the VISIBLE tokens, in their own token
        # order. The masker tops the keep quota up with padding tokens
        # whenever V_real < V_max (it ranks pad noise between real-visible and
        # real-masked), so this is generally NOT all-ones. Gathering with
        # ids_keep is exact; the boolean-index form
        # ``attn_mask.flatten(1)[mask == 0]`` returns ascending
        # ORIGINAL index order, which coincides only while padding is
        # variate-only and desynchronises as soon as time is padded too
        # (variable-T inputs).
        # (B, N_vis)
        real_visible = attn_mask.flatten(1).gather(1, ids_keep)
        self._last_visible_real = real_visible

        # append cls token
        # (B, 1, D)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # append register tokens
        # (B, R, D)
        reg_tokens = self.register_tokens.expand(x.shape[0], -1, -1)

        # (B, 1+R+N', D) = [CLS, register, patches]
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        # apply Transformer blocks
        # (B, N')
        attn_mask_visible_patches = real_visible
        # (B, 1+R+N'), add cls and register tokens to attn mask
        attn_mask_with_special = torch.cat((
            torch.ones(size=(attn_mask.shape[0], 1 + self.num_register_tokens), device=x.device),
            attn_mask_visible_patches
        ), dim=1)

        for blk in self.blocks:
            x = blk(x, attn_mask_with_special, V, T, v_pos, t_pos)

        # (B, 1+R+N', D)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_with_masked_patches(self, x, attn_mask, mask_ratio):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            attn_mask: (B, C', T'), with N=C'*T' patches

        output:
            x: (B, 1+R+N, D), with 1 cls token + R register tokens + N (visible + masked) patches
            mask: (B, N), with N (visible + mask) patches, 0 is keep, 1 is remove
            ids_restore: (B, N)
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)
        B, D, V, T = x.shape

        # flatten
        # (B, N, D), with N=C'*T'
        x = x.flatten(2).transpose(1, 2)

        # masking: length -> length * mask_ratio
        # (B, N', D), with N'=C'*T'*(1-mask_ratio)
        x, mask, ids_restore = self.random_masking(x, attn_mask, mask_ratio)

        # append mask tokens to sequence
        # (B, N-N', D)
        mask_tokens = self.mask_token_encoder.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        # (B, N, D)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle

        # append cls token
        # (B, 1, D)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # append register tokens
        # (B, R, D)
        reg_tokens = self.register_tokens.expand(x.shape[0], -1, -1)

        # (B, 1+R+N, D) = [CLS, register, patches]
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        # apply Transformer blocks
        # (B, N)
        attn_mask_all_patches = attn_mask.flatten(1)
        # (B, 1+R+N), add cls and register tokens to attn mask
        attn_mask_with_special = torch.cat((
            torch.ones(size=(attn_mask.shape[0], 1 + self.num_register_tokens), device=x.device),
            attn_mask_all_patches
        ), dim=1)

        for blk in self.blocks:
            x = blk(x, attn_mask_with_special, V, T)

        # (B, 1+R+N, D)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_encoder_all_patches(self, x, attn_mask=None):
        """
        input:
            x: (B, 1, C, T), input signal of size CxT
            attn_mask: optional (B, C', T') patch-level padding mask (1 = real);
                padding patches are then excluded from attention

        output:
            x: (B, 1+R+N, D), with 1 cls token + R register tokens + N patches
        """
        # embed patches
        # (B, D, C', T')
        x = self.patch_embed(x)
        B, D, V, T = x.shape

        # flatten
        # (B, N, D), with N=C'*T'
        x = x.flatten(2).transpose(1, 2)

        # append cls token
        # (B, 1, D)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)

        # append register tokens
        # (B, R, D)
        reg_tokens = self.register_tokens.expand(x.shape[0], -1, -1)

        # (B, 1+R+N, D) = [CLS, register, patches]
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        # attention mask over [CLS, registers, patches]; None keeps the unmasked path
        mask_tokens = None
        if attn_mask is not None and not bool(attn_mask.all()):
            S = 1 + self.num_register_tokens
            mask_tokens = torch.cat((torch.ones(B, S, device=x.device, dtype=attn_mask.dtype),
                                     attn_mask.flatten(1)), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x, mask_tokens, V, T)

        x = self.norm(x)

        return x

    def forward_decoder(self, x, attn_mask, ids_restore):
        """
        input:
            x: (B, 1+R+N', D), with 1 cls token + R register tokens + N' visible patches
            attn_mask: (B, C', T'), with N=C'*T' (visible + mask) patches
            ids_restore: (B, N)

        output:
            x: (B, N, p*q*input_channels), with p=patch_size_x, q=patch_size_y
        """
        S = 1 + self.num_register_tokens  # number of special (cls + register) tokens

        # embed tokens
        # (B, 1+R+N', D_dec)
        x = self.decoder_embed(x)

        # preserve cls and register tokens
        # (B, S, D_dec)
        special_tokens = x[:, :S, :]

        # append mask tokens to sequence
        # +S because x includes cls and register tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + S - x.shape[1], 1)
        x_ = torch.cat([x[:, S:, :], mask_tokens], dim=1)  # no cls/register tokens
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # (B, 1+R+N, D_dec)
        x = torch.cat([special_tokens, x_], dim=1)  # prepend cls and register tokens

        B, _, D_dec = x.shape
        V = attn_mask.shape[-2]
        T = attn_mask.shape[-1]

        # apply Transformer blocks
        # (B, 1+R+N), add cls and register tokens to attn mask
        attn_mask_batch = torch.cat((
            torch.full((attn_mask.shape[0], S), 1.0, device=x.device),
            attn_mask.flatten(1)),
            dim=1)

        for blk in self.decoder_blocks:
            x = blk(x, attn_mask_batch, V, T)

        # (B, 1+R+N, D_dec)
        x = self.decoder_norm(x)

        # predictor projection
        # (B, 1+R+N, p*q*input_channels)
        x = self.decoder_pred(x)

        # remove cls and register tokens
        # (B, N, p*q*input_channels)
        x = x[:, S:, :]

        return x

    @torch.no_grad()
    def forward_encoder_collect_patch_norms(self, x, attn_mask):
        """
        Forward pass through the encoder with **all patches visible** (no
        masking), recording the per-token L2 norm of the **patch tokens** at
        every layer boundary. CLS and register tokens are intentionally
        excluded — only output / patch tokens are tracked.

        Diagnostic helper for tracking how patch-token norms evolve across
        layers during pre-training (à la "Vision Transformers Need Registers"
        Fig. 4a). This intentionally diverges from ``forward_encoder``: at
        training time the encoder only processes visible patches, but here we
        feed the entire patch grid through so the per-layer norms are computed
        on the same set of tokens for every sample / step.

        input:
            x: (B, 1, C, T)
            attn_mask: (B, C', T')  — flags non-padding patches

        output:
            patch_norms: list of (B, N) float tensors, length = depth + 2
                         (index 0 = pre-block-0 input, index i+1 = post block i,
                          index -1 = post final encoder norm)
            attn_mask_patches: (B, N) 0/1 mask flagging non-padding patches
            x: (B, 1+R+N, D) final encoder output (after norm)
        """
        S = 1 + self.num_register_tokens

        # embed patches → (B, D, V, T) → (B, N, D)
        x = self.patch_embed(x)
        B, D, V, T = x.shape
        x = x.flatten(2).transpose(1, 2)
        N = x.shape[1]

        # axial positions of all patches (none are dropped)
        all_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        v_pos = all_ids // T
        t_pos = all_ids % T

        # prepend CLS + register tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        reg_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        attn_mask_patches = attn_mask.flatten(1)
        attn_mask_with_special = torch.cat((
            torch.ones(size=(B, S), device=x.device),
            attn_mask_patches
        ), dim=1)

        # patch-token norms only (skip CLS + register slots)
        patch_norms = [x[:, S:, :].float().norm(dim=-1)]  # pre-block-0
        for blk in self.blocks:
            x = blk(x, attn_mask_with_special, V, T, v_pos, t_pos)
            patch_norms.append(x[:, S:, :].float().norm(dim=-1))

        x = self.norm(x)
        patch_norms.append(x[:, S:, :].float().norm(dim=-1))  # post final norm

        return patch_norms, attn_mask_patches, x

    @torch.no_grad()
    def forward_decoder_collect_patch_norms(self, x, attn_mask):
        """
        Forward pass through the decoder with **all patches visible** (no mask
        tokens injected), recording the per-token L2 norm of the **patch
        tokens** at every layer boundary. CLS and register tokens are
        intentionally excluded. Only valid when
        ``self.output_projection == 'decoder'``.

        Like ``forward_encoder_collect_patch_norms``, this diverges from the
        pre-training path — the decoder normally consumes a mix of visible and
        mask tokens, but here it sees the full set of encoded patches so the
        per-layer norms are computed on the same set of tokens for every
        sample / step. The provided ``x`` should therefore be the "all
        patches" encoder output produced by
        ``forward_encoder_collect_patch_norms``.

        input:
            x: (B, 1+R+N, D)  — full encoder output, no mask tokens
            attn_mask: (B, C', T')

        output:
            patch_norms: list of (B, N) float tensors, length = decoder_depth + 2
                         (index 0 = pre-block-0 input, index i+1 = post block i,
                          index -1 = post final decoder norm)
            attn_mask_patches: (B, N) 0/1 mask flagging non-padding patches
        """
        S = 1 + self.num_register_tokens

        # decoder embed; sequence already contains all patches (no mask tokens)
        x = self.decoder_embed(x)

        B = x.shape[0]
        V = attn_mask.shape[-2]
        T = attn_mask.shape[-1]

        attn_mask_patches = attn_mask.flatten(1)
        attn_mask_batch = torch.cat((
            torch.ones(size=(B, S), device=x.device),
            attn_mask_patches),
            dim=1)

        patch_norms = [x[:, S:, :].float().norm(dim=-1)]  # pre-block-0
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask_batch, V, T)
            patch_norms.append(x[:, S:, :].float().norm(dim=-1))

        x = self.decoder_norm(x)
        patch_norms.append(x[:, S:, :].float().norm(dim=-1))  # post final decoder norm

        return patch_norms, attn_mask_patches

    def forward_mlp(self, x):
        """
        input:
            x: (B, 1+R+N, D), with 1 cls token + R register tokens + N (visible + masked) patches

        output:
            x: (B, N, p*q*input_channels), with p=patch_size_x, q=patch_size_y
        """
        # remove cls and register tokens
        x = x[:, 1 + self.num_register_tokens:, :]

        # apply MLP
        # (B, N, D)
        x = self.mlp(x)

        # (B, N, D)
        x = self.mlp_norm(x)

        # predictor projection
        # (B, N, p*q*input_channels)
        x = self.mlp_pred(x)

        return x

    def forward_loss(self, samples, pred, attn_mask, mask):
        """
        samples: [N, C, H, W]
        pred: [N, L, p*q*C]
        attn_mask: [N, C', T'], with C'=H/p, T'=W/q and C'*T'=L
        mask: [N, L], 0 is keep, 1 is remove
        """
        # [N, L, p*q*C]
        target = self.patchify(samples) 

        if self.norm_patch_loss:
            # mean over last dim does not require consideration of the attention mask
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        # [N, L, p*q*C]
        loss = (pred - target) ** 2
        # [N, L, 1]
        attn_mask_loss = attn_mask.flatten(1).unsqueeze(-1)
        # [N, L, p*q*C]
        loss = loss * attn_mask_loss
        # [N, L], mean loss per patch
        # mean over last dim does not require consideration of the attention mask
        loss = loss.sum(dim=-1) / (loss.shape[-1] + 1e-9)

        # REGULARISATION (using normalised correlation coefficient of the actual signals)
        # [N, C, H, W]
        samples_hat = self.unpatchify(pred, attn_mask)
        # [N, C, H, W]
        attn_mask_input_space = torch.nn.functional.interpolate(attn_mask.unsqueeze(1), scale_factor=self.patch_size, mode="nearest")
        # [N, C, H, W]
        samples_hat = samples_hat * attn_mask_input_space

        # batch_size = len(samples)
        if self.masked_patch_loss:
            # compute loss only on masked patches
            # [N, C, H, W]
            combined_mask = attn_mask.flatten(1) * mask # attention mask combined with the actual mask (visible vs masked tokens)
            # combined_mask_input_space = torch.nn.functional.interpolate(combined_mask.reshape(attn_mask.shape).unsqueeze(1), 
            #                                                             scale_factor=self.patch_size, 
            #                                                             mode="nearest")
            
            # [N]
            # number of reconstructed (masked only) patches
            # may also be 0 if time series length < patch size
            # hence only consider the samples with nb_patches > 0 
            nb_patches = torch.sum(combined_mask, dim=-1)

            # [N]
            loss = (torch.sum(loss * mask, dim=-1) / nb_patches)[nb_patches > 0]

            # # [N, C, H, W]
            # samples_masked_patches = samples * combined_mask_input_space
            # samples_hat_masked_patches = samples_hat * combined_mask_input_space

            # # [N]
            # ncc = statistics.ncc(samples_masked_patches, samples_hat_masked_patches, combined_mask_input_space, keep_batch=True)[nb_patches > 0]
            
            # # [N]
            # nb_patches = nb_patches[nb_patches > 0]
        else:
            # compute loss on all (masked + visible) patches
            # [N]
            # number of reconstructed (masked + visible) patches
            nb_patches = torch.sum(attn_mask.flatten(1), dim=-1)
            
            # [N]
            loss = (torch.sum(loss, dim=-1) / nb_patches)[nb_patches > 0]

        # [N]
        ncc = statistics.ncc(samples, samples_hat, attn_mask_input_space, keep_batch=True)[nb_patches > 0]

        # [N]
        nb_patches = nb_patches[nb_patches > 0]

        # Per-sample -> batch reduction: w_s = 1/N -> per-sample mean (each
        # recording weighs 1/N).
        # Applied to BOTH signal-space terms (MSE + NCC); latent terms are untouched.
        w = torch.ones_like(nb_patches, dtype=loss.dtype)
        w = w / w.sum()
        loss_batch = (loss * w).sum()
        ncc_batch = (ncc * w).sum()

        return loss_batch, ncc_batch, samples_hat

    def _teacher_params(self):
        """Iterate over the teacher submodule parameters (used for freezing + EMA)."""
        yield self.teacher_cls_token
        yield self.teacher_register_tokens
        yield from self.teacher_patch_embed.parameters()
        yield from self.teacher_blocks.parameters()
        yield from self.teacher_norm.parameters()
        yield from self.teacher_head.parameters()
        if self.self_distillation_loss and not self.ibot_share_head:
            yield from self.teacher_mim_head.parameters()

    @torch.no_grad()
    def sync_teacher_from_student(self):
        """Copy the student weights into the EMA teacher (same pairing as the EMA update).

        Needed after loading a pre-trained student into a fresh model: the
        teacher was snapshotted from the randomly initialised student in
        ``__init__`` and would otherwise start the SD targets from noise.
        """
        if not self.self_distillation_loss:
            return
        for t, s in zip(self._teacher_params(), self._student_ema_params()):
            t.data.copy_(s.data)

    def _student_ema_params(self):
        """Iterate over the student params whose EMA feeds the teacher (same order)."""
        yield self.cls_token
        yield self.register_tokens
        yield from self.patch_embed.parameters()
        yield from self.blocks.parameters()
        yield from self.norm.parameters()
        yield from self.student_head.parameters()
        if self.self_distillation_loss and not self.ibot_share_head:
            yield from self.student_mim_head.parameters()

    def _teacher_masking_from_student(self, x_t_flat, mask_s, ids_restore_s):
        """
        Build teacher visible-token set as a superset of the student's visible set.

        Reuses the student's noise permutation (via ``ids_restore_s``) and takes a
        larger prefix of size ``len_keep_t``. Since the student's sort placed its
        visible tokens at positions ``[0, student_len_keep)``, any longer prefix
        strictly contains them — so every token visible to the student is also
        visible to the teacher. ``len_keep_t`` is clamped to be at least the
        student's len_keep (needed when the forecasting branch temporarily uses a
        smaller effective mask_ratio than ``ibot_teacher_mask_ratio``).
        """
        B, L, D = x_t_flat.shape
        # mask_s: 0=keep (visible), 1=remove (masked). len_keep is batch-uniform.
        student_len_keep = L - int(mask_s[0].sum().item())
        len_keep_t = math.ceil(L * (10 - 10 * self.ibot_teacher_mask_ratio) / 10)
        len_keep_t = max(len_keep_t, student_len_keep)

        ids_shuffle_s = torch.argsort(ids_restore_s, dim=1)
        ids_keep_t = ids_shuffle_s[:, :len_keep_t]

        x_t_masked = torch.gather(
            x_t_flat, dim=1,
            index=ids_keep_t.unsqueeze(-1).expand(-1, -1, D)
        )

        mask_t = torch.ones(B, L, device=x_t_flat.device)
        mask_t[:, :len_keep_t] = 0
        mask_t = torch.gather(mask_t, dim=1, index=ids_restore_s)

        return x_t_masked, mask_t, ids_keep_t

    def _teacher_masking_real_superset(self, x_t_flat, attn_mask, mask_s, ids_restore_s, forecast):
        """
        Build a teacher view that always adds REAL patches to the student's view.

        ``_teacher_masking_from_student`` takes a longer prefix of the
        student's noise order. That order ranks padding between the student's
        visible and masked patches, so the extra teacher slots go to padding
        first. A heavily padded sample then gives the teacher no extra real
        patch. The masked patches also share the key +inf, so CUDA picks them
        by index (variate 0 first), not at random.

        This builder counts real (non-padding) patches per sample n:

        * The teacher sees ``K_n = max(ceil(n_valid * (1 - rho_t)), n_s + 1)``
          real patches, capped at ``n_valid``. ``n_s`` is the number of real
          patches the student sees and ``rho_t = ibot_teacher_mask_ratio``
          (``ibot_teacher_forecast_mask_ratio`` for forecasting views, if set).
        * Random view: the ``K_n - n_s`` extras are drawn uniformly from the
          student-masked real patches.
        * Forecasting view: the teacher sees time steps ``t < P_n`` in every real
          variate, with ``P_n = max(ceil(K_n / V_n), P_s + 1)``, capped at the
          number of real time steps. ``V_n`` is the number of real variates and
          ``P_s`` the longest student prefix. The student's prefix can differ
          by one step across variates (its keep count is rounded over the whole
          sequence), so a common end step keeps the teacher view a clean
          forecasting view.

        Layout: the student's kept slots come first, in the student's order.
        So the first ``N_s`` teacher tokens stay aligned with the student's
        visible tokens (L_MIM needs no gather). The extras follow.
        ``len_keep_t`` is the batch maximum. Slots beyond a sample's own extras
        are filler and are excluded from attention by ``valid_t``.

        Returns:
            x_t_masked: (B, len_keep_t, D)
            ids_keep_t: (B, len_keep_t), flattened-grid positions
            valid_t:    (B, len_keep_t), 1 = attended, 0 = padding or filler
        """
        B, L, D = x_t_flat.shape
        _, V, T = attn_mask.shape
        device = x_t_flat.device
        valid = attn_mask.flatten(1) > 0                                    # (B, L)
        kept_s = mask_s == 0                                                # (B, L)
        student_len_keep = int(kept_s[0].sum().item())                      # batch-uniform
        cand = valid & ~kept_s                                              # student-masked real patches

        n_valid = valid.sum(dim=1)                                          # (B,)
        n_s = (valid & kept_s).sum(dim=1)                                   # (B,)
        rho_t = self.ibot_teacher_mask_ratio
        if forecast and getattr(self, 'ibot_teacher_forecast_mask_ratio', None) is not None:
            rho_t = self.ibot_teacher_forecast_mask_ratio
        k_t = torch.ceil(n_valid.float() * (10 - 10 * rho_t) / 10).long()
        k_t = torch.minimum(torch.maximum(k_t, n_s + 1), n_valid)
        n_extra = (k_t - n_s).clamp(min=0)                                  # (B,)

        if forecast:
            valid_vt = valid.view(B, V, T)
            n_var = valid_vt.any(dim=-1).sum(dim=-1).clamp(min=1)          # (B,) real variates
            n_time = valid_vt.sum(dim=-1).max(dim=-1)[0]                    # (B,) real time steps
            p_s = (valid & kept_s).view(B, V, T).sum(dim=-1).max(dim=-1)[0] # (B,) longest student prefix
            p_t = torch.ceil(k_t.float() / n_var.float()).long()
            p_t = torch.minimum(torch.maximum(p_t, p_s + 1), n_time)        # (B,) teacher end step
            t_idx = torch.arange(T, device=device).view(1, 1, T)
            sel = (cand.view(B, V, T) & (t_idx < p_t.view(B, 1, 1))).flatten(1)    # (B, L)
            # extras ordered by time, then variate
            v_idx = torch.arange(V, device=device).view(1, V, 1)
            extra_key = (t_idx * V + v_idx).expand(B, V, T).flatten(1).float()
        else:
            noise = torch.rand(B, L, device=device).masked_fill(~cand, float('inf'))
            rank = torch.argsort(torch.argsort(noise, dim=1), dim=1)
            sel = cand & (rank < n_extra.view(B, 1))                        # (B, L)
            extra_key = noise                                               # random order of extras

        # order: student kept slots (student order) < extras < rest (filler)
        key = torch.full((B, L), float('inf'), device=device)
        key = torch.where(sel, L + extra_key, key)
        key = torch.where(kept_s, ids_restore_s.float(), key)
        ids_order = torch.argsort(key, dim=1)
        len_keep_t = student_len_keep + int(sel.sum(dim=1).max().item())
        ids_keep_t = ids_order[:, :len_keep_t]

        x_t_masked = torch.gather(
            x_t_flat, dim=1,
            index=ids_keep_t.unsqueeze(-1).expand(-1, -1, D)
        )
        valid_t = (valid & (kept_s | sel)).float().gather(1, ids_keep_t)
        return x_t_masked, ids_keep_t, valid_t

    @torch.no_grad()
    def _teacher_forward(self, samples, attn_mask, mask_s, ids_restore_s, forecast=False):
        """
        Teacher encoder pass returning the [CLS] feature and the patch-token
        features at the student's visible positions.

        The teacher's visible token set is a random superset of the student's
        (derived from ``ids_restore_s``) of size ``len_keep_t`` (controlled by
        ``self.ibot_teacher_mask_ratio``, typically smaller than the student's
        mask_ratio). Since ``ids_keep_t[:, :N_s] == ids_keep_s`` by
        construction, the FIRST ``N_s`` teacher patch tokens are positionally
        aligned with the student's visible tokens — no gather needed for the
        MIM branch.

        With ``ibot_teacher_real_superset`` the teacher view is built by
        ``_teacher_masking_real_superset`` instead: it always adds real patches,
        and ``forecast`` (the student's masking mode for this view) selects
        random extras (False) or a longer forecasting view (True).

        Returns:
            cls_feat:    (B, D) teacher [CLS] feature (post-norm).
            patch_feats: (B, N_s, D) teacher features at the student's visible
                         positions, in the same per-sample order as the student
                         encoder output's patch slice.
        """
        x = self.teacher_patch_embed(samples)
        B, _, V, T = x.shape
        x = x.flatten(2).transpose(1, 2)

        # mask_s: 0=keep (visible), 1=remove. len_keep_s is batch-uniform.
        student_len_keep = x.shape[1] - int(mask_s[0].sum().item())

        if self.ibot_teacher_real_superset:
            x, ids_keep, attn_mask_visible_patches = self._teacher_masking_real_superset(
                x, attn_mask, mask_s, ids_restore_s, forecast)
        else:
            x, mask, ids_keep = self._teacher_masking_from_student(x, mask_s, ids_restore_s)
            # gather in the teacher's token order (ids_keep); boolean indexing
            # would return ascending grid order, which desyncs under time padding
            attn_mask_visible_patches = attn_mask.flatten(1).gather(1, ids_keep)

        v_pos = ids_keep // T
        t_pos = ids_keep % T

        cls_tokens = self.teacher_cls_token.expand(B, -1, -1)
        reg_tokens = self.teacher_register_tokens.expand(B, -1, -1)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        attn_mask_with_special = torch.cat((
            torch.ones(size=(attn_mask.shape[0], 1 + self.num_register_tokens), device=x.device),
            attn_mask_visible_patches
        ), dim=1)

        for blk in self.teacher_blocks:
            x = blk(x, attn_mask_with_special, V, T, v_pos, t_pos)
        x = self.teacher_norm(x)

        S = 1 + self.num_register_tokens
        cls_feat = x[:, 0, :]
        # first N_s teacher patch positions == student's visible positions (same order)
        patch_feats = x[:, S:S + student_len_keep, :]
        return cls_feat, patch_feats

    def compute_sd_loss(self, samples, attn_mask, mask_ratio, latent_u, mask_u, ids_restore_u,
                        forecast_u=False):
        """
        iBOT self-distillation: L_CLS on the [CLS] token + L_MIM on the student's
        VISIBLE patch tokens. Both branches are always computed when
        self-distillation is enabled; a single unified weight (``sd_weight`` in
        the engine) governs their contribution (DINOv3-style).

        The student uses two independent random masks at ``mask_ratio`` (views u
        and v). The teacher uses masks that are random *supersets* of each
        student mask (size controlled by ``self.ibot_teacher_mask_ratio``), so
        every token visible to the student is also visible to the teacher. This
        yields a more informed teacher CLS/patch than the current setup
        (teacher at the same mask_ratio as the student) without the full-sequence
        cost of vanilla iBOT. The subset construction also makes MIM cheap: the
        first ``N_s`` teacher visible tokens are positionally aligned with the
        student's visible tokens, so no gather is needed for the MIM CE.

        latent_u:       student encoder output (B, 1+R+N_s_u, D) from the main
                        forward — reused here as view u to avoid an extra pass.
        mask_u:         student mask for view u (B, L), 0=keep, 1=remove.
        ids_restore_u:  student permutation for view u (B, L); teacher reuses it
                        to derive a subset mask.
        Returns:
            cls_loss:    iBOT L_CLS scalar (~[0, ln K] at init).
            mim_loss:    iBOT L_MIM scalar on visible positions.
            cls_cos_sim: cosine similarity between student and teacher raw CLS
                         (diagnostic).
            cls_std:     std of L2-normalised student CLS across batch, scaled
                         by sqrt(D) (collapse detector).
        """
        S = 1 + self.num_register_tokens

        # ---- Student: reuse view u from the main forward + fresh view v ----
        cls_s_u = latent_u[:, 0, :]                                                  # (B, D)
        latent_v, mask_v, ids_restore_v = self.forward_encoder(samples, attn_mask, mask_ratio)
        forecast_v = self._last_mask_forecast
        cls_s_v = latent_v[:, 0, :]                                                  # (B, D)

        # Student CLS head on both views
        cls_logits_s_u = self.student_head(cls_s_u)                                  # (B, K)
        cls_logits_s_v = self.student_head(cls_s_v)                                  # (B, K)

        # ---- Teacher forwards — visible set is a random superset of the student's ----
        cls_t_u, patch_t_u = self._teacher_forward(samples, attn_mask, mask_u, ids_restore_u, forecast_u)
        cls_t_v, patch_t_v = self._teacher_forward(samples, attn_mask, mask_v, ids_restore_v, forecast_v)

        cls_logits_t_u = self.teacher_head(cls_t_u)                                  # (B, K)
        cls_logits_t_v = self.teacher_head(cls_t_v)                                  # (B, K)

        # ---- L_CLS branch: symmetrised cross-view CE ----
        p_t_cls_u = F.softmax((cls_logits_t_u - self.center) / self.ibot_teacher_temp, dim=-1).detach()
        p_t_cls_v = F.softmax((cls_logits_t_v - self.center) / self.ibot_teacher_temp, dim=-1).detach()

        log_p_s_cls_u = F.log_softmax(cls_logits_s_u / self.ibot_student_temp, dim=-1)
        log_p_s_cls_v = F.log_softmax(cls_logits_s_v / self.ibot_student_temp, dim=-1)

        cls_loss_uv = -(p_t_cls_v * log_p_s_cls_u).sum(dim=-1).mean()
        cls_loss_vu = -(p_t_cls_u * log_p_s_cls_v).sum(dim=-1).mean()
        cls_loss = 0.5 * (cls_loss_uv + cls_loss_vu)

        # ---- L_MIM branch on visible student patches (same-view u↔u, v↔v) ----
        # Patch slices — positionally aligned thanks to the subset-mask teacher.
        # Student patch features: (B, N_s, D)
        patch_s_u = latent_u[:, S:, :]
        patch_s_v = latent_v[:, S:, :]

        # Heads (may be shared with CLS head when ibot_share_head=True)
        if self.ibot_share_head:
            stud_mim_head = self.student_head
            teach_mim_head = self.teacher_head
        else:
            stud_mim_head = self.student_mim_head
            teach_mim_head = self.teacher_mim_head

        # Apply heads token-wise
        B_u, Ns_u, D_u = patch_s_u.shape
        mim_logits_s_u = stud_mim_head(patch_s_u.reshape(B_u * Ns_u, D_u)).reshape(B_u, Ns_u, -1)
        mim_logits_t_u = teach_mim_head(patch_t_u.reshape(B_u * Ns_u, D_u)).reshape(B_u, Ns_u, -1)

        B_v, Ns_v, D_v = patch_s_v.shape
        mim_logits_s_v = stud_mim_head(patch_s_v.reshape(B_v * Ns_v, D_v)).reshape(B_v, Ns_v, -1)
        mim_logits_t_v = teach_mim_head(patch_t_v.reshape(B_v * Ns_v, D_v)).reshape(B_v, Ns_v, -1)

        # Per-sample-then-batch reduction using DINOv3's masks_weight pattern.
        # Only valid (non-padding) visible positions contribute; per-sample
        # mean over those positions, then mean over the batch.
        # Valid flags in the student's token order (ids_keep), not in ascending
        # grid order: the two differ as soon as time is padded.
        ids_keep_s_u = torch.argsort(ids_restore_u, dim=1)[:, :Ns_u]
        ids_keep_s_v = torch.argsort(ids_restore_v, dim=1)[:, :Ns_v]
        valid_vis_u = attn_mask.flatten(1).gather(1, ids_keep_s_u).float()          # (B, N_s)
        valid_vis_v = attn_mask.flatten(1).gather(1, ids_keep_s_v).float()

        mw_u = valid_vis_u / valid_vis_u.sum(dim=-1, keepdim=True).clamp(min=1.0)
        mw_v = valid_vis_v / valid_vis_v.sum(dim=-1, keepdim=True).clamp(min=1.0)

        p_t_mim_u = F.softmax((mim_logits_t_u - self.patch_center) / self.ibot_teacher_temp, dim=-1).detach()
        p_t_mim_v = F.softmax((mim_logits_t_v - self.patch_center) / self.ibot_teacher_temp, dim=-1).detach()

        log_p_s_mim_u = F.log_softmax(mim_logits_s_u / self.ibot_student_temp, dim=-1)
        log_p_s_mim_v = F.log_softmax(mim_logits_s_v / self.ibot_student_temp, dim=-1)

        # same-view self-distillation (DINOv3 pattern): student_u <- teacher_u, etc.
        ce_u = -(p_t_mim_u * log_p_s_mim_u).sum(dim=-1)                              # (B, N_s)
        ce_v = -(p_t_mim_v * log_p_s_mim_v).sum(dim=-1)

        mim_loss_u = (ce_u * mw_u).sum() / B_u
        mim_loss_v = (ce_v * mw_v).sum() / B_v
        mim_loss = 0.5 * (mim_loss_u + mim_loss_v)

        # Cache teacher patch logits for the centre update (post-step).
        # Flatten token axis and keep only valid visible positions.
        valid_u_bool = valid_vis_u.reshape(-1).bool()
        valid_v_bool = valid_vis_v.reshape(-1).bool()
        teacher_patch_flat = torch.cat([
            mim_logits_t_u.reshape(-1, mim_logits_t_u.shape[-1])[valid_u_bool],
            mim_logits_t_v.reshape(-1, mim_logits_t_v.shape[-1])[valid_v_bool],
        ], dim=0)
        self._last_teacher_patch_logits = teacher_patch_flat.detach()

        # ---- Diagnostics ----
        cls_cos_sim = F.cosine_similarity(cls_s_v, cls_t_v.detach(), dim=-1).mean()
        cls_u_norm = F.normalize(cls_s_u, dim=-1)
        cls_std = cls_u_norm.std(dim=0).mean() * (cls_s_u.shape[-1] ** 0.5)

        # Cache teacher CLS logits for the centre update (called from the engine after optimiser step)
        self._last_teacher_cls_logits = torch.cat([cls_logits_t_u, cls_logits_t_v], dim=0).detach()

        return cls_loss, mim_loss, cls_cos_sim, cls_std

    @torch.no_grad()
    def momentum_update_teacher(self, m=None):
        """EMA update: teacher <- m * teacher + (1 - m) * student."""
        if not self.self_distillation_loss:
            return
        if m is None:
            m = self.ibot_ema_decay
        for sp, tp in zip(self._student_ema_params(), self._teacher_params()):
            tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)

    def set_last_layer_frozen(self, frozen: bool):
        """Freeze/unfreeze the weight-normalised last layer of the student head.

        DINO/iBOT recipe: zero student_head.last_layer grads for the first ~1 epoch
        to stabilise training. `weight_g` is already permanently frozen (norm pinned
        to 1); here we additionally toggle `weight_v` and bias (if any).
        """
        if not self.self_distillation_loss:
            return
        # The [CLS] head is always covered. The separate patch head is covered
        # too (iBOT freezes every ``last_layer``) unless
        # ``ibot_freeze_cls_head_only`` is set.
        heads = [self.student_head]
        if not self.ibot_share_head and not self.ibot_freeze_cls_head_only:
            heads.append(self.student_mim_head)
        for head in heads:
            last_layer = head.last_layer
            for p in last_layer.parameters():
                # weight_g is permanently frozen (norm-pinning); leave it alone.
                if p is last_layer.weight_g:
                    continue
                p.requires_grad = not frozen

    @torch.no_grad()
    def update_center(self, m_c=None):
        """EMA update of the centre buffer from the last teacher CLS logits."""
        if not self.self_distillation_loss or self._last_teacher_cls_logits is None:
            return
        if m_c is None:
            m_c = self.ibot_center_momentum

        # fp32: the cached logits are in the autocast dtype (bf16/fp16)
        batch_center = self._last_teacher_cls_logits.float().mean(dim=0, keepdim=True)  # (1, K)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(batch_center, op=dist.ReduceOp.SUM)
            batch_center = batch_center / dist.get_world_size()

        self.center.mul_(m_c).add_(batch_center, alpha=1.0 - m_c)

    @torch.no_grad()
    def update_patch_center(self, m_c=None):
        """EMA update of the MIM patch-centre buffer from the last teacher patch logits.

        TODO: DINOv3 uses Sinkhorn-Knopp for the MIM teacher normalisation. Our
        batches are small (~160 vs DINOv3's multi-thousand), so we keep the
        softmax-centering scheme as the MVP (DINOv2 Table 1 showed SK was
        neutral at B~3k). Revisit once we scale batch size significantly.
        """
        if not self.self_distillation_loss or self._last_teacher_patch_logits is None:
            return
        if self._last_teacher_patch_logits.numel() == 0:
            return
        if m_c is None:
            m_c = self.ibot_center_momentum

        # Local sum + count; then all-reduce to get a global mean over tokens.
        # fp32: the cached logits are in the autocast dtype; in fp16 the global
        # token count (~77k) would overflow (max 65504), in bf16 it is rounded
        local_sum = self._last_teacher_patch_logits.float().sum(dim=0, keepdim=True)  # (1, K)
        local_cnt = torch.tensor([self._last_teacher_patch_logits.shape[0]],
                                 device=local_sum.device, dtype=torch.float32)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(local_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(local_cnt, op=dist.ReduceOp.SUM)
        batch_center = local_sum / local_cnt.clamp(min=1.0)

        self.patch_center.mul_(m_c).add_(batch_center, alpha=1.0 - m_c)

    # ------------------------------------------------------------------
    # Gram anchor loss (DINOv3 §4.2)
    # ------------------------------------------------------------------
    def _gram_teacher_params(self):
        yield self.gram_teacher_cls_token
        yield self.gram_teacher_register_tokens
        yield from self.gram_teacher_patch_embed.parameters()
        yield from self.gram_teacher_blocks.parameters()
        yield from self.gram_teacher_norm.parameters()

    def _iter_gram_refresh_pairs(self):
        """Pairs of (EMA-teacher param, Gram-teacher param) in matching order."""
        yield self.teacher_cls_token, self.gram_teacher_cls_token
        yield self.teacher_register_tokens, self.gram_teacher_register_tokens
        yield from zip(self.teacher_patch_embed.parameters(),
                       self.gram_teacher_patch_embed.parameters())
        yield from zip(self.teacher_blocks.parameters(),
                       self.gram_teacher_blocks.parameters())
        yield from zip(self.teacher_norm.parameters(),
                       self.gram_teacher_norm.parameters())

    def _build_gram_teacher(self):
        """Allocate frozen Gram-teacher submodules as a snapshot of the EMA teacher.

        Called at the end of ``__init__`` when ``gram_enabled=True`` so that DDP
        sees the params at wrap time. Weights are refreshed in place from the
        current EMA teacher at activation (and periodically thereafter) via
        ``refresh_gram_teacher()``.
        """
        self.gram_teacher_patch_embed = copy.deepcopy(self.teacher_patch_embed)
        self.gram_teacher_cls_token = nn.Parameter(self.teacher_cls_token.data.clone())
        self.gram_teacher_register_tokens = nn.Parameter(self.teacher_register_tokens.data.clone())
        self.gram_teacher_blocks = copy.deepcopy(self.teacher_blocks)
        self.gram_teacher_norm = copy.deepcopy(self.teacher_norm)

        # re-patch block.forward closures onto the new (deep-copied) blocks
        for block in self.gram_teacher_blocks:
            block.forward = self._block_forward_wrapper(block)

        for p in self._gram_teacher_params():
            p.requires_grad = False

    @torch.no_grad()
    def refresh_gram_teacher(self):
        """In-place copy of current EMA teacher weights into the Gram teacher.

        Keeps the tensor identities intact so DDP / optimizer bookkeeping is
        not disturbed. Idempotent and cheap (O(#teacher_params) memcopy).
        """
        for src, dst in self._iter_gram_refresh_pairs():
            dst.data.copy_(src.data)

    def forward_encoder_gram(self, x, attn_mask, *, teacher: bool):
        """Encoder forward with ALL patches visible, attention padding-masked.

        Used for the DINOv3 §4.2 Gram anchor loss. Unlike ``forward_encoder``
        it drops no patches (no iBOT masking) and unlike
        ``forward_encoder_all_patches`` it respects ``attn_mask`` so padded
        positions do not attend.

        Args:
            x:         (B, C_in, V, T) raw samples.
            attn_mask: (B, V', T') 0/1 patch-grid validity mask.
            teacher:   if True, routes through frozen ``gram_teacher_*``
                       submodules; otherwise routes through the student
                       encoder (gradient flows).

        Returns:
            patches: (B, V'*T', D) patch-token features (CLS + register
                     dropped).
            valid:   (B, V'*T') 0/1 validity mask flattened along time axis.
            V:       number of variate patches (for axial RoPE bookkeeping).
            T:       number of time patches.
        """
        if teacher:
            patch_embed = self.gram_teacher_patch_embed
            cls_tok = self.gram_teacher_cls_token
            reg_tok = self.gram_teacher_register_tokens
            blocks = self.gram_teacher_blocks
            norm = self.gram_teacher_norm
        else:
            patch_embed = self.patch_embed
            cls_tok = self.cls_token
            reg_tok = self.register_tokens
            blocks = self.blocks
            norm = self.norm

        x = patch_embed(x)
        B, D, V, T = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, V*T, D)
        N = x.shape[1]

        all_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        v_pos = all_ids // T
        t_pos = all_ids % T

        S = 1 + self.num_register_tokens
        cls_tokens = cls_tok.expand(B, -1, -1)
        reg_tokens = reg_tok.expand(B, -1, -1)
        x = torch.cat((cls_tokens, reg_tokens, x), dim=1)

        valid = attn_mask.flatten(1)  # (B, N)
        attn_mask_with_special = torch.cat((
            torch.ones(size=(B, S), device=x.device, dtype=valid.dtype),
            valid,
        ), dim=1)

        for blk in blocks:
            x = blk(x, attn_mask_with_special, V, T, v_pos, t_pos)
        x = norm(x)

        return x[:, S:, :], valid, V, T

    def compute_gram_loss(self, samples, attn_mask, downsample_factor: int = 2):
        """DINOv3 §4.2 Gram anchor: ``|| G_s - G_t ||_F^2`` (mean-normalised).

        Gram teacher runs at native resolution; Gram student runs on a
        time-decimated copy of the same samples (default 2x). Teacher features
        are linearly interpolated along the time axis to match the student's
        patch count, then both sides are L2-normalised along the feature
        dimension and padded positions are zeroed so they contribute nothing
        to the Gram matrix. Returns a scalar loss. P-invariant: uses
        ``.mean()`` so the weight transfers across context lengths.
        """
        B, C_in, V_in, T_in = samples.shape
        ds = int(downsample_factor)

        # --- 1. Teacher forward (full-res, frozen, no-grad) ---
        # Teacher submodules are set to eval behaviourally via requires_grad=False
        # but we force eval mode locally to disable drop_path noise in the anchor.
        was_gram_teacher_training = self.gram_teacher_blocks.training
        self.gram_teacher_blocks.eval()
        self.gram_teacher_norm.eval()
        try:
            with torch.no_grad():
                X_t, valid_t, V_t, T_t = self.forward_encoder_gram(
                    samples, attn_mask, teacher=True)
        finally:
            if was_gram_teacher_training:
                self.gram_teacher_blocks.train()
                self.gram_teacher_norm.train()

        # --- 2. Decimate samples along time (antialias via box filter) ---
        # samples: (B, C_in, V_in, T_in) -> pool last dim by ``ds``.
        samples_flat = samples.reshape(B * C_in * V_in, 1, T_in)
        samples_ds = F.avg_pool1d(samples_flat, kernel_size=ds, stride=ds)
        T_s_in = samples_ds.shape[-1]
        samples_ds = samples_ds.reshape(B, C_in, V_in, T_s_in)

        # Downsample the patch-grid attention mask too. A student patch is
        # valid iff any of the ``ds`` source patches it subsumes was valid.
        Vg, Tg = attn_mask.shape[-2], attn_mask.shape[-1]
        attn_mask_ds = F.max_pool1d(
            attn_mask.float().reshape(B * Vg, 1, Tg),
            kernel_size=ds, stride=ds,
        ).reshape(B, Vg, Tg // ds)

        # --- 3. Student forward (downsampled, grad flows) ---
        X_s, valid_s, V_s, T_s = self.forward_encoder_gram(
            samples_ds, attn_mask_ds, teacher=False)
        # X_s: (B, V_s * T_s, D), valid_s: (B, V_s * T_s)

        # --- 4. Interpolate teacher features along the time axis to match ---
        D = X_t.shape[-1]
        # (B, V_t, T_t, D) -> (B*V_t, D, T_t) -> interp -> (B*V_t, D, T_s) -> (B, V_t*T_s, D)
        X_t = X_t.reshape(B, V_t, T_t, D).permute(0, 1, 3, 2).reshape(B * V_t, D, T_t)
        X_t = F.interpolate(X_t.float(), size=T_s, mode='linear', align_corners=False)
        X_t = X_t.to(X_s.dtype)
        X_t = X_t.reshape(B, V_t, D, T_s).permute(0, 1, 3, 2).reshape(B, V_t * T_s, D)

        # --- 5. Mask padded positions and L2-normalise ---
        valid_mask = valid_s.float().unsqueeze(-1)  # (B, N_s, 1)
        X_s = F.normalize(X_s, dim=-1) * valid_mask
        X_t = F.normalize(X_t, dim=-1) * valid_mask

        # --- 6. Gram matrices and mean-squared difference (P-invariant) ---
        G_s = torch.bmm(X_s, X_s.transpose(1, 2))
        G_t = torch.bmm(X_t, X_t.transpose(1, 2))
        return ((G_s - G_t) ** 2).mean()

    def forward(self, samples, attn_mask, mask_ratio=0.75):
        """
        samples: [N, C, H, W]
        attn_mask: [N, C', T'], with C'*T'=L and C'=H/p, T'=W/q
        """
        if self.output_projection == 'decoder':
            # latent of visible tokens
            latent, mask, ids_restore = self.forward_encoder(samples, attn_mask, mask_ratio)
            # Real/padding indicator of ``latent``'s patch rows, captured here
            # because ``compute_sd_loss`` runs a further forward_encoder pass
            # (view v) that overwrites ``_last_visible_real``. Consumed
            # by the feature-quality diagnostics in engine_pretrain.py, which
            # must not average over padded slots.
            self._metric_visible_real = self._last_visible_real
            # masking mode of view u, captured before compute_sd_loss masks view v
            forecast_u = self._last_mask_forecast
            pred = self.forward_decoder(latent, attn_mask, ids_restore)  # [N, L, p*q*C]
        else: # mlp
            # latent of all tokens
            latent, mask, ids_restore = self.forward_encoder_with_masked_patches(samples, attn_mask, mask_ratio)
            forecast_u = self._last_mask_forecast
            # this path keeps ALL patches in original grid order, so the
            # indicator is the attention mask itself
            self._metric_visible_real = attn_mask.flatten(1)
            pred = self.forward_mlp(latent)  # [N, L, p*q*C]

        loss, ncc, samples_hat = self.forward_loss(samples, pred, attn_mask, mask)

        if self.self_distillation_loss:
            cls_loss, mim_loss, cls_cos_sim, cls_std = self.compute_sd_loss(
                samples, attn_mask, mask_ratio, latent, mask, ids_restore,
                forecast_u=forecast_u)
        else:
            cls_loss = torch.tensor([0.0], dtype=torch.float32, device=samples.device)
            mim_loss = torch.tensor([0.0], dtype=torch.float32, device=samples.device)
            cls_cos_sim = torch.tensor([0.0], dtype=torch.float32, device=samples.device)
            cls_s = latent[:, 0, :]
            cls_std = F.normalize(cls_s, dim=-1).std(dim=0).mean() * (cls_s.shape[-1] ** 0.5)

        return loss, ncc, cls_loss, mim_loss, cls_cos_sim, cls_std, samples_hat, mask, latent


def otis_base_dec160d4b(**kwargs):    # nb_params: 7.1M encoder, 1.70M decoder
    model = OTIS(
        embed_dim=192, depth=12, num_heads=3,                               # dim=64 per head
        decoder_embed_dim=160, decoder_depth=4, decoder_num_heads=5,        # dim=32 per head
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_small_dec160d4b(**kwargs):  # nb_params: 14.77M encoder, 1.71M decoder
    model = OTIS(
        embed_dim=256, depth=14, num_heads=4,                               # dim=64 per head
        decoder_embed_dim=160, decoder_depth=4, decoder_num_heads=5,        # dim=32 per head
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_large_dec160d4b(**kwargs):   # nb_params: 40M encoder, 1.74M decoder
    model = OTIS(
        embed_dim=320, depth=24, num_heads=5,                               # dim=64 per head
        decoder_embed_dim=160, decoder_depth=4, decoder_num_heads=5,        # dim=32 per head
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model

def otis_huge_dec160d4b(**kwargs):    # nb_params: 116M encoder, 1.78M decoder
    model = OTIS(
        embed_dim=448, depth=36, num_heads=7,                               # dim=64 per head
        decoder_embed_dim=160, decoder_depth=4, decoder_num_heads=5,        # dim=32 per head
        mlp_ratio=4, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), # alternatively DyT
        **kwargs)
    return model