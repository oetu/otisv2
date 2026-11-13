# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm:     https://github.com/huggingface/pytorch-image-models
# DyT:      https://github.com/jiachenzhu/DyT
# RoFormer: https://arxiv.org/abs/2104.09864                    (Su et al., RoPE)
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sdpa_mask_with_sink(attn_mask, sink_logit, B, num_heads, N_real, ref):
    """
    Build an SDPA float attention mask that combines (optional) padding and an
    (optional) per-head sink column.

    Returns a float mask of shape (B, num_heads, 1, N_real [+ 1]) or None.
    Sink K/V are appended as zero rows by the caller; the sink-column score is
    set to `sink_logit[h]` via this float mask (additive on logits before
    softmax). Real-key columns get 0 on valid positions and -inf on masked.

    Float masks force SDPA off the bool-mask Flash path, but Flash bf16 with
    additive bias is still used on recent PyTorch.
    """
    has_sink = sink_logit is not None
    has_pad = attn_mask is not None
    if not has_sink and not has_pad:
        return None
    if has_pad:
        real = ref.new_zeros(B, 1, 1, N_real).masked_fill(
            (~attn_mask.bool()).view(B, 1, 1, N_real), float('-inf'),
        ).expand(B, num_heads, 1, N_real)
    else:
        real = ref.new_zeros(B, num_heads, 1, N_real)
    if has_sink:
        sink_col = sink_logit.view(1, num_heads, 1, 1).expand(B, -1, 1, 1)
        return torch.cat([real, sink_col], dim=-1)
    return real


def _append_sink_kv(k, v):
    """Append a single zero-row to K and V along the sequence axis (per head)."""
    sink = k.new_zeros(k.size(0), k.size(1), 1, k.size(3))
    return torch.cat([k, sink], dim=-2), torch.cat([v, sink], dim=-2)


class Attention(nn.Module):
    """
    Single-projection multi-head self-attention via PyTorch's fused SDPA kernel.
    Mirrors AxialRoPEAttention's attention numerics but without RoPE — used as
    the no-positional-info baseline.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 qk_norm=False, attention_sink=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.attention_sink = attention_sink
        self.scale = 1.0 if qk_norm else self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.qk_norm:
            self.q_norm = nn.RMSNorm(self.head_dim)
            self.k_norm = nn.RMSNorm(self.head_dim)

        if self.attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(num_heads))

        self.attn_map = None

    def forward(self, x, attn_mask=None):
        B, N, D = x.shape  # D = embed_dim
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each (B, num_heads, N, head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.attention_sink:
            k, v = _append_sink_kv(k, v)
            sdpa_mask = _sdpa_mask_with_sink(
                attn_mask, self.sink_logit, B, self.num_heads, N, q,
            )
        elif attn_mask is not None:
            sdpa_mask = attn_mask.bool().unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N) True = attend
        else:
            sdpa_mask = None

        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )

        attn = attn.transpose(1, 2).reshape(B, N, D)
        x = self.proj(attn)
        x = self.proj_drop(x)
        return x


class AxialRoPEAttention(nn.Module):
    """
    Apply RoPE positional embeddings along both temporal and variate dimensions (axial RoPE).
    Supports arbitrary sequence lengths during inference through dynamic frequency computation.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., rope_theta=10000.0,
                 num_special_tokens=1, qk_norm=True, attention_sink=False):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.attention_sink = attention_sink
        # When qk_norm is on, RMSNorm bounds the attention logits and the
        # 1/sqrt(d) softmax scale becomes redundant.
        self.scale = 1.0 if qk_norm else self.head_dim ** -0.5
        self.rope_theta = rope_theta
        # Number of leading non-patch tokens (CLS + register tokens) skipped by RoPE.
        self.num_special_tokens = num_special_tokens

        # Each axis gets half of the head dimension
        assert self.head_dim % 2 == 0, 'head_dim must be divisible by 2 for axial RoPE'
        self.axial_dim = self.head_dim // 2  # dimension per axis

        # Compute frequency bases (these are independent of sequence length)
        self.register_buffer('inv_freq_v', self._compute_inv_freq())
        self.register_buffer('inv_freq_t', self._compute_inv_freq())

        # Cache for frequencies (will be computed on-demand)
        self._freqs_v_cache = {}
        self._freqs_t_cache = {}

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.attention_sink:
            self.sink_logit = nn.Parameter(torch.zeros(num_heads))

        self.attn_map = None

    def _compute_inv_freq(self):
        """Compute inverse frequency bases for RoPE."""
        # θ_i = 10000^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (self.rope_theta ** (torch.arange(0, self.axial_dim, 2).float() / self.axial_dim))
        return inv_freq

    def _get_freqs(self, seq_len, inv_freq, cache):
        """
        Get or compute frequency matrix for given sequence length.
        Uses caching for efficiency but supports arbitrary lengths.
        """
        if seq_len not in cache:
            # Compute frequencies for this sequence length
            position = torch.arange(seq_len, device=inv_freq.device, dtype=inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', position, inv_freq)  # (seq_len, axial_dim // 2)
            cache[seq_len] = freqs
        return cache[seq_len]

    def _apply_rotary_emb(self, x, freqs):
        """
        Apply rotary embeddings to x using precomputed frequency angles.
        x:     (B, num_heads, N, axial_dim)
        freqs: (N, axial_dim // 2)          – shared across batch
           or  (B, N, axial_dim // 2)       – per-sample (masked sequences)
        """
        x_rot = x[..., ::2]   # (B, num_heads, N, axial_dim // 2)
        x_im  = x[..., 1::2]  # (B, num_heads, N, axial_dim // 2)

        cos = freqs.cos()
        sin = freqs.sin()

        if cos.ndim == 2:
            # (N, d) -> (1, 1, N, d) for broadcasting with (B, num_heads, N, d)
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        elif cos.ndim == 3:
            # (B, N, d) -> (B, 1, N, d) for broadcasting with (B, num_heads, N, d)
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)

        x_rotated = torch.cat([
            x_rot * cos - x_im * sin,
            x_rot * sin + x_im * cos,
        ], dim=-1)

        return x_rotated

    def forward(self, x, attn_mask=None, V=None, T=None, v_pos=None, t_pos=None):
        """
        x: (B, S+N, D) with S leading special tokens (CLS + register) + N patches
        attn_mask: (B, S+N) attention mask
        V: number of variates (used to generate frequency table)
        T: number of temporal patches (used to generate frequency table)
        v_pos: (B, N) variate position index for each patch [optional]
        t_pos: (B, N) temporal position index for each patch [optional]
              If v_pos/t_pos are None, dense row-major grid order is assumed.
        """
        B, N_total, D = x.shape
        S = self.num_special_tokens

        # Generate QKV
        qkv = self.qkv(x).reshape(B, N_total, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # Each (B, num_heads, N_total, head_dim)

        # Apply axial RoPE to patches (skip the S leading special tokens)
        if V is not None and T is not None:
            # Split q and k into special tokens and patches
            q_special = q[:, :, :S]  # (B, num_heads, S, head_dim)
            k_special = k[:, :, :S]
            q_patches = q[:, :, S:]  # (B, num_heads, N, head_dim)
            k_patches = k[:, :, S:]

            N = q_patches.shape[2]

            # Determine position indices
            if v_pos is None or t_pos is None:
                # Dense row-major grid (no masking or patches already in original order)
                v_pos_idx = torch.arange(V, device=x.device).unsqueeze(1).expand(V, T).reshape(-1)[:N]
                t_pos_idx = torch.arange(T, device=x.device).unsqueeze(0).expand(V, T).reshape(-1)[:N]
            else:
                # Explicit per-patch positions (e.g. after masking + shuffling)
                v_pos_idx = v_pos  # (B, N) or (N,)
                t_pos_idx = t_pos

            # Split head_dim into two halves for the two axes
            q_v, q_t = q_patches.split(self.axial_dim, dim=-1)  # (B, num_heads, N, axial_dim)
            k_v, k_t = k_patches.split(self.axial_dim, dim=-1)

            # Get frequency tables for each axis
            freqs_v = self._get_freqs(V, self.inv_freq_v, self._freqs_v_cache)  # (V, axial_dim // 2)
            freqs_t = self._get_freqs(T, self.inv_freq_t, self._freqs_t_cache)  # (T, axial_dim // 2)

            # Select frequencies for each patch
            freqs_v_sel = freqs_v[v_pos_idx]  # (B, N, axial_dim//2) or (N, axial_dim//2)
            freqs_t_sel = freqs_t[t_pos_idx]

            # Apply RoPE per axis
            q_v = self._apply_rotary_emb(q_v, freqs_v_sel)
            k_v = self._apply_rotary_emb(k_v, freqs_v_sel)
            q_t = self._apply_rotary_emb(q_t, freqs_t_sel)
            k_t = self._apply_rotary_emb(k_t, freqs_t_sel)

            # Recombine
            q_patches = torch.cat([q_v, q_t], dim=-1)  # (B, num_heads, N, head_dim)
            k_patches = torch.cat([k_v, k_t], dim=-1)

            q = torch.cat([q_special, q_patches], dim=2)
            k = torch.cat([k_special, k_patches], dim=2)

        # QK-norm: RMS-normalise Q and K after RoPE to stabilise attention logits
        if self.qk_norm:
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))

        # Fused scaled-dot-product attention (Flash / mem-efficient kernels).
        # When qk_norm is on, self.scale=1.0; otherwise the standard 1/sqrt(d).
        if self.attention_sink:
            k, v = _append_sink_kv(k, v)
            sdpa_mask = _sdpa_mask_with_sink(
                attn_mask, self.sink_logit, B, self.num_heads, N_total, q,
            )
        elif attn_mask is not None:
            # (B, N_total) with 1=attend, 0=mask → boolean (B, 1, 1, N_total) for SDPA
            sdpa_mask = attn_mask.bool().unsqueeze(1).unsqueeze(2)
        else:
            sdpa_mask = None

        dropout_p = self.attn_drop.p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sdpa_mask,
            dropout_p=dropout_p,
            scale=self.scale,
        )  # (B, num_heads, N_total, head_dim)

        out = out.transpose(1, 2)  # (B, N_total, num_heads, head_dim)
        out = out.reshape(B, N_total, D)

        out = self.proj(out)
        out = self.proj_drop(out)

        return out


class DyT(nn.Module):
    def __init__(self, num_features, alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1) * alpha_init_value)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return x * self.weight + self.bias