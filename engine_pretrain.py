# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------
import os

import math
import sys
import time
from typing import Iterable
import random

import torch

import numpy as np

import wandb

import util.misc as misc
import util.lr_sched as lr_sched
import util.statistics as statistics
from util.sine_patch_sim import plot_sine_patch_similarity
from util.benchmarks import UEA_DATASETS
from util.dataset import load_samples

import matplotlib
matplotlib.use('Agg')           # prevents tkinter error
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

from sklearn.feature_selection import r_regression
from sklearn.base import clone as sklearn_clone

import json


def _amp_dtype(args):
    """Resolve CUDA AMP autocast dtype from args (default bf16, OTISv2 native)."""
    return torch.float16 if getattr(args, 'amp_dtype', 'bf16') == 'fp16' else torch.bfloat16


def _feature_quality_metrics(latent, real_visible, S, R):
    """Feature-quality diagnostics over the REAL visible patch tokens only.

    ``latent`` is (B, S+N', D) = [CLS, R registers, N' visible patches], and the
    visible patch set is NOT all real: both maskers fix the keep quota over the
    *padded* grid and rank padding tokens between the real-visible and the
    real-masked ones, so every sample with ``V_real < V_max`` has its quota
    topped up with padding (measured ~48% of the visible set on the
    pre-training corpus at micro-batch 64). Padded slots embed to a
    near-constant token, i.e. one direction in feature space, which pulls
    pairwise cosine toward 1 and eats RankMe's spectral entropy. Averaging over
    them therefore made every encoder look more collapsed than it is, and by an
    encoder-dependent amount (the inflation is largest where the real tokens
    are least collapsed), so it also compressed between-model gaps.

    ``real_visible`` (B, N') is the model's own 1/0 indicator for those rows,
    aligned with the token order. Definitions match the paper's evaluation
    protocol (App. "Feature quality"): patch-token cosine similarity is the
    off-diagonal mean over real-real pairs with denominator
    ``sum_b n_b (n_b - 1)``; RankMe is taken over the real tokens alone.

    Returns (cls_patch_sim, register_patch_sim, patch_patch_sim, rankme,
    rankme_gap, rankme_cls), all scalar tensors. ``register_patch_sim`` is 0
    when R == 0. The last two are the SAMPLE-level RankMe the reference
    defines (one row per sample); ``rankme`` is the token-level variant kept
    for continuity with logged history.
    """
    latent = latent.float()
    real = real_visible.float()                                        # (B, N')
    n_real = real.sum().clamp(min=1.0)

    cls_token = torch.nn.functional.normalize(latent[:, 0, :], dim=-1)       # (B, D)
    patch_tokens = torch.nn.functional.normalize(latent[:, S:, :], dim=-1)   # (B, N', D)

    cls_patch_sim = (torch.einsum('bd,bnd->bn', cls_token, patch_tokens) * real).sum() / n_real

    if R > 0:
        register_tokens = torch.nn.functional.normalize(latent[:, 1:S, :], dim=-1)   # (B, R, D)
        register_patch_sim = (
            torch.einsum('brd,bnd->brn', register_tokens, patch_tokens)
            * real.unsqueeze(1)
        ).sum() / (n_real * R)
    else:
        register_patch_sim = torch.zeros((), device=latent.device)

    # Patch-token pairwise similarity (collapse diagnostic), real-real pairs only
    pp_sim = torch.einsum('bid,bjd->bij', patch_tokens, patch_tokens)        # (B, N', N')
    pp_N = pp_sim.shape[1]
    off_diag = ~torch.eye(pp_N, dtype=torch.bool, device=pp_sim.device)
    pair = real.unsqueeze(2) * real.unsqueeze(1) * off_diag.unsqueeze(0)     # (B, N', N')
    patch_patch_sim = (pp_sim * pair).sum() / pair.sum().clamp(min=1.0)

    # RankMe — effective rank of patch outputs (Garrido et al., ICML 2023). Computed
    # on raw, un-normalised patch tokens flattened across the batch. Bounded
    # [1, min(n_real, D)]; higher = more directions used = less collapsed.
    rankme = statistics.rankme(latent[:, S:, :][real.bool()])

    # Sample-level RankMe on the reference's own Z: one embedding row per sample,
    # so B x D rather than (B N') x D. GAP is that sample's mean over its REAL
    # visible patch tokens, CLS is its [CLS] token. These measure ACROSS-sample
    # diversity — what a frozen downstream probe consumes — where the token-level
    # score above measures within-sample spatial diversity.
    #
    # Ceiling caveat: the bound is min(B, D) with B the per-GPU MICRO-batch, so
    # both are censored whenever the micro-batch is smaller than the encoder
    # width (b=64 at d=192 caps them at 64). They are comparable along a run and
    # across runs at matched micro-batch and width, never against the offline
    # protocol (N = 25,600).
    n_per_sample = real.sum(dim=1, keepdim=True).clamp(min=1.0)               # (B, 1)
    gap = (latent[:, S:, :] * real.unsqueeze(-1)).sum(dim=1) / n_per_sample   # (B, D)
    rankme_gap = statistics.rankme(gap)
    rankme_cls = statistics.rankme(latent[:, 0, :])

    return (cls_patch_sim, register_patch_sim, patch_patch_sim,
            rankme, rankme_gap, rankme_cls)


def _visible_real(model, latent):
    """Fetch the model's real/padding indicator for ``latent``'s patch rows.

    ``OTIS.forward`` stashes it (``_metric_visible_real``) because the token
    order is the masker's, not the grid's, so it cannot be reconstructed from
    ``attn_mask``/``mask`` alone. Falls back to all-ones for models that predate
    the stash, which reproduces the old padding-inclusive numbers rather than
    crashing.
    """
    _m = model.module if hasattr(model, "module") else model
    n_patch = latent.shape[1] - (1 + _m.num_register_tokens)
    real = getattr(_m, "_metric_visible_real", None)
    if real is None or tuple(real.shape) != (latent.shape[0], n_patch):
        return torch.ones(latent.shape[0], n_patch, device=latent.device)
    return real


def _compute_masked_metrics(samples, samples_hat, attn_mask, mask, patch_size):
    """Compute MSE/MAE on masked patches and return upsampled masks."""
    # (B, 1, V, T) — 0 is padding, 1 is actual value
    attn_mask_input_space = torch.nn.functional.interpolate(
        attn_mask.unsqueeze(1), scale_factor=patch_size, mode="nearest")

    # (B, 1, V, T) — 0 is keep, 1 is remove
    mask_input_space = torch.nn.functional.interpolate(
        mask.reshape(attn_mask.shape).unsqueeze(1), scale_factor=patch_size, mode="nearest")

    # (B, 1, V, T)
    combined_mask = attn_mask_input_space * mask_input_space
    samples_diff = samples - samples_hat

    # evaluation only on the masked patches
    mse = ((samples_diff**2) * combined_mask).sum() / (combined_mask.sum() + 1e-9)
    mae = (abs(samples_diff) * combined_mask).sum() / (combined_mask.sum() + 1e-9)

    return attn_mask_input_space, mask_input_space, combined_mask, mse.item(), mae.item()


def _compute_variate_stats(samples, samples_hat, combined_mask, var_idx, sample_idx):
    """Compute per-variate NCC and MSE for a single sample."""
    s = samples[sample_idx, 0, var_idx]
    s_hat = samples_hat[sample_idx, 0, var_idx]
    cm = combined_mask[sample_idx, 0, var_idx]

    ncc_val = statistics.ncc(s, s_hat)
    ncc_masked = statistics.ncc(s, s_hat, cm)
    mse_masked = ((s - s_hat)**2 * cm).sum() / (cm.sum() + 1e-9)
    mse_all = ((s - s_hat)**2).mean()

    return ncc_val, ncc_masked, mse_masked, mse_all


def _plot_variate(x, x_hat, x_hat_masked, mask_var, ncc_val, ncc_masked, mse_all, mse_masked, var_idx, subplot_offset):
    """Plot 4 subplots for a single variate: input, overlay, reconstruction, masked-only."""
    t = range(0, x.shape[-1], 1)

    plt.subplot(8, 1, subplot_offset)
    plt.title(f"Input (variate {var_idx})")
    plt.plot(t, x, color='black')

    plt.subplot(8, 1, subplot_offset + 1)
    plt.title(f"Input vs Reconstruction (NCC {ncc_val.item():.2f}, MSE {mse_all.item():.4f}, masked patches in gray)")
    plt.plot(t, x, color='black')
    plt.plot(t, x_hat, color='darkorange')
    plt.fill_between(t,
                     y1=min(x.min(), x_hat.min()),
                     y2=max(x.max(), x_hat.max()),
                     where=mask_var, color='gray', alpha=0.25)

    plt.subplot(8, 1, subplot_offset + 2)
    plt.title(f"Reconstruction (NCC {ncc_val.item():.2f}, MSE {mse_all.item():.4f}, masked patches in gray)")
    plt.plot(t, x_hat, color='darkorange')
    plt.fill_between(t, y1=x_hat.min(), y2=x_hat.max(),
                     where=mask_var, color='gray', alpha=0.25)

    plt.subplot(8, 1, subplot_offset + 3)
    plt.title(f"Reconstruction of masked patches (NCC {ncc_masked.item():.2f}, MSE {mse_masked.item():.4f}, masked patches in gray)")
    plt.plot(t, x_hat_masked, color='darkorange')
    plt.fill_between(t, y1=x_hat_masked.min(), y2=x_hat_masked.max(),
                     where=mask_var, color='gray', alpha=0.25)
    indices_visible = np.where(mask_var == False)
    plt.scatter(indices_visible, x_hat_masked[indices_visible],
                color='white', s=7, zorder=2)


def _plot_reconstruction(samples, samples_hat, combined_mask, mask_input_space, attn_mask_input_space, idx):
    """Generate 8-subplot reconstruction figure for a single sample and return wandb Image."""
    max_steps = int(attn_mask_input_space[idx, 0, 0, :].sum())

    x = samples[idx][..., :max_steps].detach().cpu().numpy()
    x_hat = samples_hat[idx][..., :max_steps].detach().cpu().numpy()
    x_hat_masked = (samples_hat[idx] * combined_mask[idx])[..., :max_steps].detach().cpu().numpy()

    # Variate 0
    ncc_0, ncc_0m, mse_0, mse_0_all = _compute_variate_stats(
        samples, samples_hat, combined_mask, 0, idx)
    mask_0 = (mask_input_space[idx, 0, 0, :max_steps] == 1).cpu().numpy()

    # Second variate (random if multivariate, else variate 0 again)
    max_variates = int(attn_mask_input_space[idx, 0, :, 0].sum())
    if max_variates > 1:
        var_idx = random.randint(1, max_variates - 1)
        ncc_1, ncc_1m, mse_1, mse_1_all = _compute_variate_stats(
            samples, samples_hat, combined_mask, var_idx, idx)
        mask_1 = (mask_input_space[idx, 0, var_idx, :max_steps] == 1).cpu().numpy()
    else:
        var_idx = 0
        ncc_1, ncc_1m, mse_1, mse_1_all = ncc_0, ncc_0m, mse_0, mse_0_all
        mask_1 = mask_0

    plt.close('all')
    plt.figure(figsize=(8, 8))

    _plot_variate(x[0, 0, :], x_hat[0, 0, :], x_hat_masked[0, 0, :], mask_0,
                  ncc_0, ncc_0m, mse_0_all, mse_0, 0, subplot_offset=1)
    _plot_variate(x[0, var_idx, :], x_hat[0, var_idx, :], x_hat_masked[0, var_idx, :], mask_1,
                  ncc_1, ncc_1m, mse_1_all, mse_1, var_idx, subplot_offset=5)

    plt.tight_layout()
    return wandb.Image(plt)


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None,
                    gram_active: bool = False,
                    intra_kshot_state: dict = None):
    """``intra_kshot_state`` (optional, dict): when provided, mid-epoch UEA
    online k-shot evals are appended to ``state['records']`` as dicts
    ``{epoch, step, global_step, train_tokens, cls, gap}``. Frequency is
    controlled by ``args.online_kshot_intra_epoch_steps`` (0 = disabled).
    State must include keys ``records`` (list), ``tokens_per_step`` (int)
    and ``data_base`` (str path to UEA mmap)."""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    # Optional losses — callers that don't expose these CLI args (e.g. forecasting)
    # fall back to 0.0 and the corresponding branches are skipped.
    sd_weight = getattr(args, 'sd_weight', 0.0)
    koleo_weight = getattr(args, 'koleo_weight', 0.0)
    gram_weight = getattr(args, 'gram_weight', 0.0)
    gram_ds_factor = getattr(args, 'gram_downsample_factor', 2)
    gram_on = bool(gram_active and gram_weight > 0.0)

    optimizer.zero_grad()

    # iBOT freeze schedule is updated per step inside the loop (continuous
    # ``step/len + epoch`` progress, same as the teacher-temp / EMA cosine
    # schedules). The previous epoch-wise integer comparison degenerated for
    # short-horizon runs (e=1 → frozen for the entire run).

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    training_history = {}

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        start_time = time.time()
        samples, attn_mask = batch[0], batch[1]

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_wsd(optimizer, data_iter_step / len(data_loader) + epoch, args,
                                                warmup_fraction=getattr(args, 'warmup_fraction', 0.1),
                                                decay_fraction=getattr(args, 'decay_fraction', 0.1))

        # iBOT teacher-temp warmup + freeze schedule: continuous epoch
        # progress (matches the EMA cosine schedule below).
        if sd_weight > 0.0:
            _m = (model.module if hasattr(model, "module") else model)
            epoch_progress = data_iter_step / len(data_loader) + epoch
            # Teacher-temp warmup: linear from tau_start to tau_end.
            warmup_frac = getattr(args, 'ibot_teacher_temp_warmup_fraction', 0.0)
            if warmup_frac > 0.0:
                warmup_epochs = warmup_frac * args.epochs
                ramp = min(epoch_progress / max(warmup_epochs, 1e-8), 1.0)
                tau_start = getattr(args, 'ibot_teacher_temp_start', 0.04)
                tau_end = args.ibot_teacher_temp
                _m.ibot_teacher_temp = tau_start + ramp * (tau_end - tau_start)
            # Last-layer freeze: head's projection last layer is frozen while
            # epoch_progress < freeze_frac * E. Continuous comparison fixes the
            # short-horizon degeneracy of the prior integer-epoch check.
            freeze_frac = getattr(args, 'ibot_freeze_last_layer_fraction', 0.0)
            _m.set_last_layer_frozen(epoch_progress < freeze_frac * args.epochs)

        # push samples to device
        samples = samples.to(device, non_blocking=True)
        attn_mask = attn_mask.to(device, non_blocking=True)

        # compute model prediction
        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            loss, ncc, cls_loss, mim_loss, cls_cos_sim, cls_std, samples_hat, mask, latent = model(samples,
                                                                                                   attn_mask,
                                                                                                   mask_ratio=args.mask_ratio)
            # Gram anchor loss (DINOv3 §4.2). Called on model.module so the Gram
            # teacher (frozen, requires_grad=False) is reached directly. Student
            # params are shared with the main forward and their grads are already
            # tracked by DDP's Reducer, so the Gram backward path works under
            # find_unused_parameters=True without extra wiring.
            if gram_on:
                _m = (model.module if hasattr(model, "module") else model)
                gram_loss = _m.compute_gram_loss(samples, attn_mask,
                                                 downsample_factor=gram_ds_factor)
            else:
                gram_loss = torch.tensor(0.0, device=loss.device)

        batch_size = len(samples)

        # number of register tokens (model may be DDP-wrapped)
        R = (model.module.num_register_tokens if hasattr(model, "module") else model.num_register_tokens)
        S = 1 + R  # cls + register

        # Feature-quality diagnostics, over the REAL visible patch tokens only —
        # padded slots would otherwise dominate (~48% of the visible set) and
        # inflate every similarity while depressing RankMe. See
        # _feature_quality_metrics for the definitions and the protocol they match.
        # latent: (B, 1+R+N', D) = [CLS, register, patches]
        cls_patch_sim, register_patch_sim, patch_patch_sim, patch_rankme, \
            gap_rankme, cls_rankme = \
            _feature_quality_metrics(latent, _visible_real(model, latent), S, R)

        # KoLeo regulariser on encoder-output CLS tokens
        if koleo_weight > 0.0:
            cls_tokens = latent[:, 0, :]  # (B, D)
            koleo_gs = getattr(args, "koleo_group_size", 0)
            koleo = statistics.koleo_loss(cls_tokens, group_size=koleo_gs if koleo_gs and koleo_gs > 0 else None)
        else:
            koleo = torch.tensor(0.0, device=loss.device)

        loss_value = loss.item()
        ncc_value = ncc.item()
        cls_loss_value = cls_loss.item()
        mim_loss_value = mim_loss.item()
        cls_cos_sim_value = cls_cos_sim.item()
        cls_std_value = cls_std.item()
        cls_patch_sim_value = cls_patch_sim.item()
        register_patch_sim_value = register_patch_sim.item()
        patch_patch_sim_value = patch_patch_sim.item()
        rankme_value = patch_rankme.item()
        rankme_gap_value = gap_rankme.item()
        rankme_cls_value = cls_rankme.item()
        koleo_value = koleo.item()
        gram_loss_value = gram_loss.item()
        total_loss_value = (loss_value
                            + args.ncc_weight * (1 - ncc_value)
                            + sd_weight * cls_loss_value + sd_weight * mim_loss_value
                            + koleo_weight * koleo_value
                            + gram_weight * gram_loss_value)

        # check the TOTAL loss (all terms) on all ranks, so every rank exits together
        if not misc.all_ranks_finite(total_loss_value):
            print(f"Non-finite loss on some rank (total={total_loss_value}, mse={loss_value}, ncc={ncc_value}, "
                  f"cls={cls_loss_value}, mim={mim_loss_value}, koleo={koleo_value}), stopping training")
            sys.exit(1)

        loss /= accum_iter
        ncc /= accum_iter
        cls_loss /= accum_iter
        mim_loss /= accum_iter
        koleo /= accum_iter
        gram_loss /= accum_iter

        total_loss = (loss
                      + args.ncc_weight * (1 - ncc)
                      + sd_weight * cls_loss + sd_weight * mim_loss
                      + koleo_weight * koleo
                      + gram_weight * gram_loss)
        grad_norm = loss_scaler(total_loss, optimizer, parameters=model.parameters(),
                                clip_grad=getattr(args, 'clip_grad', None),
                                update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
            # iBOT: EMA update of teacher + centre after the optimiser step
            if sd_weight > 0.0:
                _m = (model.module if hasattr(model, "module") else model)
                # Optional cosine schedule for teacher EMA: base -> 1.0
                if getattr(args, 'ibot_ema_cosine', False):
                    progress = (data_iter_step / len(data_loader) + epoch) / max(args.epochs, 1e-8)
                    progress = min(max(progress, 0.0), 1.0)
                    m_base = args.ibot_ema_decay
                    m_now = 1.0 - (1.0 - m_base) * 0.5 * (1.0 + math.cos(math.pi * progress))
                    _m.momentum_update_teacher(m_now)
                else:
                    _m.momentum_update_teacher()
                _m.update_center()
                _m.update_patch_center()

        total_time = time.time() - start_time

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        metric_logger.meters['total_loss'].update(total_loss_value, n=batch_size)

        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)
        metric_logger.meters['cls_loss'].update(cls_loss_value, n=batch_size)
        metric_logger.meters['mim_loss'].update(mim_loss_value, n=batch_size)
        metric_logger.meters['cls_cos_sim'].update(cls_cos_sim_value, n=batch_size)
        metric_logger.meters['cls_std'].update(cls_std_value, n=batch_size)
        metric_logger.meters['cls_patch_sim'].update(cls_patch_sim_value, n=batch_size)
        metric_logger.meters['register_patch_sim'].update(register_patch_sim_value, n=batch_size)
        metric_logger.meters['patch_patch_sim'].update(patch_patch_sim_value, n=batch_size)
        metric_logger.meters['rankme'].update(rankme_value, n=batch_size)
        metric_logger.meters['rankme_gap'].update(rankme_gap_value, n=batch_size)
        metric_logger.meters['rankme_cls'].update(rankme_cls_value, n=batch_size)
        metric_logger.meters['koleo'].update(koleo_value, n=batch_size)
        metric_logger.meters['gram_loss'].update(gram_loss_value, n=batch_size)

        # Pre-clip gradient norm (only produced on optimiser update steps). Logging this
        # helps pick a sensible --clip_grad max-norm: inspect the running distribution
        # (p50/p90/p99) across training and set clip_grad slightly above the bulk.
        grad_norm_value_reduce = None
        if grad_norm is not None:
            grad_norm_value = grad_norm.item()
            metric_logger.meters['grad_norm'].update(grad_norm_value, n=batch_size)
            grad_norm_value_reduce = misc.all_reduce_mean(grad_norm_value)

        attn_mask_input_space, mask_input_space, combined_mask, mse_value, mae_value = \
            _compute_masked_metrics(samples, samples_hat, attn_mask, mask, args.patch_size)

        metric_logger.meters['mse'].update(mse_value, n=batch_size)
        metric_logger.meters['mae'].update(mae_value, n=batch_size)

        total_loss_value_reduce = misc.all_reduce_mean(total_loss_value)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        ncc_value_reduce = misc.all_reduce_mean(ncc_value)
        cls_loss_value_reduce = misc.all_reduce_mean(cls_loss_value)
        mim_loss_value_reduce = misc.all_reduce_mean(mim_loss_value)
        cls_cos_sim_value_reduce = misc.all_reduce_mean(cls_cos_sim_value)
        cls_std_value_reduce = misc.all_reduce_mean(cls_std_value)
        cls_patch_sim_value_reduce = misc.all_reduce_mean(cls_patch_sim_value)
        register_patch_sim_value_reduce = misc.all_reduce_mean(register_patch_sim_value)
        patch_patch_sim_value_reduce = misc.all_reduce_mean(patch_patch_sim_value)
        rankme_value_reduce = misc.all_reduce_mean(rankme_value)
        rankme_gap_value_reduce = misc.all_reduce_mean(rankme_gap_value)
        rankme_cls_value_reduce = misc.all_reduce_mean(rankme_cls_value)
        koleo_value_reduce = misc.all_reduce_mean(koleo_value)
        gram_loss_value_reduce = misc.all_reduce_mean(gram_loss_value)
        mse_value_reduce = misc.all_reduce_mean(mse_value)
        mae_value_reduce = misc.all_reduce_mean(mae_value)

        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('lr', lr, epoch_1000x)

            log_writer.add_scalar('train/train_total_loss', total_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_ncc', ncc_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_cls_loss', cls_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_mim_loss', mim_loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_cls_cos_sim', cls_cos_sim_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_cls_std', cls_std_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_cls_patch_sim', cls_patch_sim_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_register_patch_sim', register_patch_sim_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_patch_patch_sim', patch_patch_sim_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_rankme', rankme_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_rankme_gap', rankme_gap_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_rankme_cls', rankme_cls_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_koleo', koleo_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_gram_loss', gram_loss_value_reduce, epoch_1000x)
            if grad_norm_value_reduce is not None:
                log_writer.add_scalar('train/train_grad_norm', grad_norm_value_reduce, epoch_1000x)
            # evaluation only on the masked patches
            log_writer.add_scalar('train/train_mse', mse_value_reduce, epoch_1000x)
            log_writer.add_scalar('train/train_mae', mae_value_reduce, epoch_1000x)

        # Optional intra-epoch online k-shot eval. Runs every
        # `online_kshot_intra_epoch_steps` mini-batches, on the main process
        # only, on the same UEA mmap base as end-of-epoch eval.
        intra_freq = int(getattr(args, 'online_kshot_intra_epoch_steps', 0) or 0)
        if intra_kshot_state is not None and intra_freq > 0 and misc.is_main_process():
            step1 = data_iter_step + 1
            if step1 % intra_freq == 0 or step1 == len(data_loader):
                _kshot_hist = evaluate_online_kshot(
                    model.module if hasattr(model, "module") else model,
                    device,
                    intra_kshot_state["data_base"],
                )
                cls = float(_kshot_hist.get("online/kshot_cls_avg_acc_balanced", float("nan")))
                gap = float(_kshot_hist.get("online/kshot_avg_acc_balanced", float("nan")))
                global_step = epoch * len(data_loader) + step1
                rec = {
                    "epoch": int(epoch),
                    "step": int(step1),
                    "global_step": int(global_step),
                    "train_tokens": int(global_step * intra_kshot_state["tokens_per_step"]),
                    "cls": cls,
                    "gap": gap,
                    # Token-collapse / rank diagnostics - current mini-batch
                    # values (already reduced across DDP ranks). Sampled at
                    # the same cadence as the k-shot probe so the figure can
                    # plot all four curves on a shared x-axis without W&B.
                    "rankme": float(rankme_value_reduce),
                    "rankme_gap": float(rankme_gap_value_reduce),
                    "rankme_cls": float(rankme_cls_value_reduce),
                    "patch_patch_sim": float(patch_patch_sim_value_reduce),
                }
                intra_kshot_state["records"].append(rec)
                sink = intra_kshot_state.get("sink_path")
                if sink:
                    with open(sink, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec) + "\n")
                # Restore train mode (evaluate_online_kshot calls model.eval()).
                model.train(True)

        if args.wandb == True and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if misc.is_main_process():
                wandb.log({"epoch_1000x": epoch_1000x,
                           "time_per_step[sec]": total_time,
                           "lr": lr,
                           "train_total_loss": total_loss_value_reduce,
                           "train_loss": loss_value_reduce,
                           "train_ncc": ncc_value_reduce,
                           "train_cls_loss": cls_loss_value_reduce,
                           "train_mim_loss": mim_loss_value_reduce,
                           "train_cls_cos_sim": cls_cos_sim_value_reduce,
                           "train_cls_std": cls_std_value_reduce,
                           "train_cls_patch_sim": cls_patch_sim_value_reduce,
                           "train_register_patch_sim": register_patch_sim_value_reduce,
                           "train_patch_patch_sim": patch_patch_sim_value_reduce,
                           "train_rankme": rankme_value_reduce,
                           "train_rankme_gap": rankme_gap_value_reduce,
                           "train_rankme_cls": rankme_cls_value_reduce,
                           "train_koleo": koleo_value_reduce,
                           "train_gram_loss": gram_loss_value_reduce,
                           # evaluation only on the masked patches
                           "train_mse": mse_value_reduce,
                           "train_mae": mae_value_reduce,
                           **({"train_grad_norm": grad_norm_value_reduce}
                              if grad_norm_value_reduce is not None else {})},
                          step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # wandb
    if args.wandb == True:
        training_history['epoch'] = epoch

        if (epoch % 1) == 0 or (epoch == args.epochs - 1):
            idx = random.randint(0, len(samples)-1)
            training_history["Reconstruction"] = _plot_reconstruction(
                samples, samples_hat, combined_mask, mask_input_space, attn_mask_input_space, idx)

            if misc.is_main_process():
                sine_model = model.module if hasattr(model, "module") else model
                sine_img = plot_sine_patch_similarity(sine_model, time_steps=2400, patch_width=24)
                if sine_img is not None:
                    training_history["Sine Patch Similarity"] = sine_img

    return train_stats, training_history

def _fit_regression_per_target(estimator, X, y, m):
    """Fit one clone of ``estimator`` per target on its labelled rows only (mask = 1).

    Fitting all targets jointly on zero-filled labels would treat every missing
    label as a true 0. For OLS / ridge the per-target fits equal the joint fit
    when nothing is masked.
    """
    fits = []
    for i in range(y.shape[-1]):
        rows = m[:, i] > 0
        fits.append(sklearn_clone(estimator).fit(X[rows], y[rows, i]) if rows.sum() >= 2 else None)
    return fits


def _predict_regression_per_target(fits, X):
    return np.stack([f.predict(X) if f is not None else np.full(X.shape[0], np.nan) for f in fits], axis=-1)


def _regression_metrics_masked(preds, y, m):
    """Per-target RMSE / MAE / PCC / R² over labelled rows only (nan if < 2 rows)."""
    out = {k: [] for k in ("rmse", "mae", "pcc", "r2")}
    for i in range(y.shape[-1]):
        rows = m[:, i] > 0
        if rows.sum() < 2 or not np.isfinite(preds[rows, i]).all():
            for k in out:
                out[k].append(np.nan)
            continue
        yt, yp = y[rows, i], preds[rows, i]
        out["rmse"].append(root_mean_squared_error(yt, yp))
        out["mae"].append(mean_absolute_error(yt, yp))
        out["pcc"].append(r_regression(yp.reshape(-1, 1), yt)[0])
        out["r2"].append(r2_score(yt, yp))
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _masked_gap(tokens, S, attn_mask):
    """Mean over the patch tokens (after the S special tokens), real patches only."""
    patches = tokens[:, S:, :]
    if attn_mask is None or bool(attn_mask.all()):
        return patches.mean(dim=1)          # no padding: plain mean (unchanged numerics)
    w = attn_mask.flatten(1).unsqueeze(-1).to(patches.dtype)
    return (patches * w).sum(dim=1) / w.sum(dim=1).clamp(min=1)


@torch.no_grad()
def evaluate_online(estimator, model, device, train_dataloader, val_dataloader, args=None):
    # switch to evaluation mode
    model.eval()

    online_history = {}

    # training
    train_embeddings = []
    train_labels = []
    train_masks = []
    S = 1 + model.num_register_tokens  # cls + register
    for data, label, label_mask, *rest in train_dataloader:
        data = data.to(device, non_blocking=True)
        label = label * label_mask
        train_labels.append(label.to(device, non_blocking=True))
        train_masks.append(label_mask)
        # patch-level padding mask from collate_fn_ft (None for 3-tuple loaders)
        pad_mask = rest[0].to(device, non_blocking=True) if rest else None

        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            tokens = model.forward_encoder_all_patches(data, attn_mask=pad_mask)
        # globally average pooled patch tokens, real patches only
        train_embeddings.append(_masked_gap(tokens, S, pad_mask))

    train_embeddings = torch.cat(train_embeddings, dim=0)
    train_embeddings = train_embeddings.cpu().numpy()  # Convert to numpy for sklearn
    train_labels = torch.cat(train_labels, dim=0)
    train_labels = train_labels.cpu()

    # Convert one-hot labels to class indices for fitting if needed
    if args.online_evaluation_task == "classification":
        train_labels_for_fit = train_labels.argmax(dim=-1).numpy() if train_labels.ndim > 1 and train_labels.shape[-1] > 1 else train_labels.numpy()
    else:
        train_labels_for_fit = train_labels.numpy()

    if args.online_evaluation_task == "regression":
        # per target, on labelled rows only (masked labels are not zeros)
        train_masks = torch.cat(train_masks, dim=0).cpu().numpy()
        reg_fits = _fit_regression_per_target(estimator, train_embeddings, train_labels_for_fit, train_masks)
    else:
        estimator.fit(train_embeddings, train_labels_for_fit) # only fit with training data

    if args.online_evaluation_task == "classification":
        train_probs = estimator.predict_proba(train_embeddings)  # Keep as numpy, don't convert to torch
        # Convert one-hot labels to class indices if needed
        train_labels_indices = train_labels.argmax(dim=-1).numpy() if train_labels.ndim > 1 and train_labels.shape[-1] > 1 else train_labels.numpy()

        classifier_f1_train = f1_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1), average="macro")
        classifier_precision_train = precision_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1), average="macro")
        classifier_recall_train = recall_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1), average="macro")
        classifier_acc_train = accuracy_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1))
        classifier_acc_balanced_train = balanced_accuracy_score(y_true=train_labels_indices, y_pred=train_probs.argmax(axis=-1))
        if args.online_num_classes > 2:
            classifier_auc_train = roc_auc_score(y_true=train_labels_indices, y_score=train_probs, average="macro", multi_class="ovr")
        else:
            classifier_auc_train = roc_auc_score(y_true=train_labels_indices, y_score=train_probs[:, 1], average="macro")
        # Use one-hot labels directly if already one-hot, otherwise convert
        train_labels_onehot = train_labels.numpy() if train_labels.ndim > 1 and train_labels.shape[-1] > 1 else np.eye(args.online_num_classes)[train_labels.numpy()]
        classifier_auprc_train = average_precision_score(y_true=train_labels_onehot, y_score=train_probs, average="macro")
    elif args.online_evaluation_task == "regression":
        train_preds = _predict_regression_per_target(reg_fits, train_embeddings)
        reg_train = _regression_metrics_masked(train_preds, train_labels.numpy(), train_masks)

    # validation
    val_embeddings = []
    val_labels = []
    val_masks = []
    for data, label, label_mask, *rest in val_dataloader:
        data = data.to(device, non_blocking=True)
        label = label * label_mask
        val_labels.append(label.to(device, non_blocking=True))
        val_masks.append(label_mask)
        pad_mask = rest[0].to(device, non_blocking=True) if rest else None

        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            tokens = model.forward_encoder_all_patches(data, attn_mask=pad_mask)
        val_embeddings.append(_masked_gap(tokens, S, pad_mask))

    val_embeddings = torch.cat(val_embeddings, dim=0)
    val_embeddings = val_embeddings.cpu().numpy()  # Convert to numpy for sklearn
    val_labels = torch.cat(val_labels, dim=0)
    val_labels = val_labels.cpu()

    if args.online_evaluation_task == "classification":
        val_probs = estimator.predict_proba(val_embeddings)  # Keep as numpy, don't convert to torch
        # Convert one-hot labels to class indices if needed
        val_labels_indices = val_labels.argmax(dim=-1).numpy() if val_labels.ndim > 1 and val_labels.shape[-1] > 1 else val_labels.numpy()

        classifier_f1_val = f1_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1), average="macro")
        classifier_precision_val = precision_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1), average="macro")
        classifier_recall_val = recall_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1), average="macro")
        classifier_acc_val = accuracy_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1))
        classifier_acc_balanced_val = balanced_accuracy_score(y_true=val_labels_indices, y_pred=val_probs.argmax(axis=-1))
        if args.online_num_classes > 2:
            classifier_auc_val = roc_auc_score(y_true=val_labels_indices, y_score=val_probs, average="macro", multi_class="ovr")
        else:
            classifier_auc_val = roc_auc_score(y_true=val_labels_indices, y_score=val_probs[:, 1], average="macro")
        # Use one-hot labels directly if already one-hot, otherwise convert
        val_labels_onehot = val_labels.numpy() if val_labels.ndim > 1 and val_labels.shape[-1] > 1 else np.eye(args.online_num_classes)[val_labels.numpy()]
        classifier_auprc_val = average_precision_score(y_true=val_labels_onehot, y_score=val_probs, average="macro")
    elif args.online_evaluation_task == "regression":
        val_masks = torch.cat(val_masks, dim=0).cpu().numpy()
        val_preds = _predict_regression_per_target(reg_fits, val_embeddings)
        reg_val = _regression_metrics_masked(val_preds, val_labels.numpy(), val_masks)

    # stats
    if args.online_evaluation_task == "classification":
        online_history['online/train_f1'] = classifier_f1_train
        online_history['online/train_precision'] = classifier_precision_train
        online_history['online/train_recall'] = classifier_recall_train
        online_history['online/train_acc'] = classifier_acc_train
        online_history['online/train_acc_balanced'] = classifier_acc_balanced_train
        online_history['online/train_auc'] = classifier_auc_train
        online_history['online/train_auprc'] = classifier_auprc_train

        online_history['online/val_f1'] = classifier_f1_val
        online_history['online/val_precision'] = classifier_precision_val
        online_history['online/val_recall'] = classifier_recall_val
        online_history['online/val_acc'] = classifier_acc_val
        online_history['online/val_acc_balanced'] = classifier_acc_balanced_val
        online_history['online/val_auc'] = classifier_auc_val
        online_history['online/val_auprc'] = classifier_auprc_val
    elif args.online_evaluation_task == "regression":
        # nanmean: a target without >= 2 labelled rows is skipped
        for split, reg in (("train", reg_train), ("val", reg_val)):
            for k in ("rmse", "mae", "pcc", "r2"):
                online_history[f'online/{split}_{k}'] = np.nanmean(reg[k], axis=-1)

    return online_history


def _load_uea_dataset(data_base, name):
    """Load a UEA dataset's train/test data and labels, returning tensors and class indices."""
    train_data = load_samples(os.path.join(data_base, name, "train.pt"))
    train_labels = torch.load(os.path.join(data_base, name, "train_labels.pt"), map_location="cpu", weights_only=False)
    test_data = load_samples(os.path.join(data_base, name, "test.pt"))
    test_labels = torch.load(os.path.join(data_base, name, "test_labels.pt"), map_location="cpu", weights_only=False)

    # data: list of tensors (V, T) -> (N, 1, V, T)
    X_train = torch.stack([s.unsqueeze(0) for s in train_data], dim=0).float()
    X_test = torch.stack([s.unsqueeze(0) for s in test_data], dim=0).float()

    # labels: one-hot (N, C) -> class indices (N,)
    y_train = train_labels.argmax(dim=-1).long() if train_labels.ndim == 2 else train_labels.long()
    y_test = test_labels.argmax(dim=-1).long() if test_labels.ndim == 2 else test_labels.long()

    return X_train, y_train, X_test, y_test


def _extract_embeddings(model, X, device, batch_size=16):
    """Extract CLS and globally average-pooled patch latents (excluding register tokens).

    batch_size only tiles the all-patches forward (embeddings are concatenated, no
    cross-sample interaction) -> a pure numerical no-op. Kept small (16) so the
    long-series datasets fit on 45 GiB GPUs."""
    S = 1 + model.num_register_tokens  # cls + register
    cls_embs, gap_embs = [], []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size].to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            # (B, 1+R+N, D) = [CLS, register, patches]
            emb = model.forward_encoder_all_patches(batch)
        cls_embs.append(emb[:, 0, :].cpu())
        gap_embs.append(emb[:, S:, :].mean(dim=1).cpu())
    return torch.cat(cls_embs, dim=0), torch.cat(gap_embs, dim=0)


def _prototypical_acc(train_cls, train_gap, train_y, test_cls, test_gap, test_y,
                     nb_classes, k_shots, n_episodes, rng):
    """Run k-shot prototypical classification for CLS and GAP latents.

    Returns (acc_cls, acc_gap): mean balanced accuracy across episodes.
    """
    cls_accs, gap_accs = [], []
    for _ in range(n_episodes):
        # sample k support examples per class (shared across CLS and GAP)
        support_idx = []
        for c in range(nb_classes):
            class_idx = (train_y == c).nonzero(as_tuple=True)[0].tolist()
            if len(class_idx) <= k_shots:
                support_idx.extend(class_idx)
            else:
                support_idx.extend(rng.choice(class_idx, size=k_shots, replace=False).tolist())

        support_y = train_y[support_idx]

        for train_emb, test_emb, accs in (
            (train_cls, test_cls, cls_accs),
            (train_gap, test_gap, gap_accs),
        ):
            support_emb = train_emb[support_idx]

            # compute prototypes
            prototypes = torch.zeros(nb_classes, support_emb.shape[1])
            for c in range(nb_classes):
                mask = support_y == c
                if mask.any():
                    prototypes[c] = support_emb[mask].mean(dim=0)

            # cosine similarity classification
            prototypes_norm = torch.nn.functional.normalize(prototypes, dim=1)
            query_norm = torch.nn.functional.normalize(test_emb, dim=1)
            preds = (query_norm @ prototypes_norm.T).argmax(dim=1)

            accs.append(balanced_accuracy_score(test_y.numpy(), preds.numpy()))

    return float(np.mean(cls_accs)), float(np.mean(gap_accs))


@torch.no_grad()
def evaluate_online_kshot(model, device, data_base, k_shots=5, n_episodes=3, seed=0):
    """
    5-shot prototypical evaluation on the 23 UEA datasets.
    Returns a dict with per-dataset balanced accuracy and the average.
    """
    model.eval()
    rng = np.random.RandomState(seed)
    online_history = {}
    cls_accs, gap_accs = [], []

    for name, V, T, C in UEA_DATASETS:
        try:
            X_train, y_train, X_test, y_test = _load_uea_dataset(data_base, name)
        except FileNotFoundError:
            print(f"  [online kshot] SKIP {name} (data not found)")
            continue

        train_cls, train_gap = _extract_embeddings(model, X_train, device)
        test_cls, test_gap = _extract_embeddings(model, X_test, device)

        acc_cls, acc_gap = _prototypical_acc(
            train_cls, train_gap, y_train,
            test_cls, test_gap, y_test,
            C, k_shots, n_episodes, rng,
        )
        cls_accs.append(acc_cls)
        gap_accs.append(acc_gap)
        online_history[f'online/{name}_acc_balanced'] = acc_gap * 100

    if gap_accs:
        avg_gap = float(np.mean(gap_accs)) * 100
        online_history['online/kshot_avg_acc_balanced'] = avg_gap
        print(f"  [online kshot] avg balanced acc (gap): {avg_gap:.1f}% ({len(gap_accs)} datasets)")

    if cls_accs:
        avg_cls = float(np.mean(cls_accs)) * 100
        online_history['online/kshot_cls_avg_acc_balanced'] = avg_cls
        print(f"  [online kshot] avg balanced acc (cls): {avg_cls:.1f}% ({len(cls_accs)} datasets)")

    return online_history


@torch.no_grad()
def evaluate_online_forecast(model, device, data_path, args, mask_ratio=0.25,
                             time_steps=2400, batch_size=32):
    """
    Online forecasting evaluation on synthetic sine waves (or any data file of the downstream format).
    Uses the first `time_steps` time steps of each sample; the first `1 - mask_ratio` fraction is
    visible, the remaining `mask_ratio` fraction is forecast via right-sided masking.
    Returns an online_history dict with MSE/MAE on the forecasted region, NCC over the forecast
    window and over the full signal, and a wandb reconstruction image.
    """
    if not os.path.exists(data_path):
        print(f"  [online forecast] SKIP (data not found: {data_path})")
        return {}

    # load samples: list of tensors (1, V, T_full)
    tensors_all = load_samples(data_path)

    # trim to the largest multiple of patch_width that is <= time_steps
    pw = args.patch_width
    T = (time_steps // pw) * pw
    # each tensor is already (C=1, V, T_full); stacking yields (N, 1, V, T)
    X = torch.stack([t[..., :T] for t in tensors_all], dim=0).float()

    # Temporarily switch model into right-sided ("forecasting") mask mode with a
    # deterministic horizon equal to `mask_ratio`.
    saved = (model.downstream, model.include_forecasting, model.probabilistic_masking)
    model.downstream = "forecasting"
    model.include_forecasting = False
    model.probabilistic_masking = False
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_ncc_masked = 0.0
    total_ncc_full = 0.0
    n_batches = 0
    last_batch = None

    try:
        for i in range(0, len(X), batch_size):
            samples = X[i:i + batch_size].to(device, non_blocking=True)  # (B, 1, V, T)
            B, _, V, T_ = samples.shape
            # (B, V', T') — all patches are valid
            attn_mask = torch.ones(B, V, T_ // pw, device=device)

            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, ncc, cls_loss, mim_loss, cls_cos_sim, cls_std, samples_hat, mask, latent = model(
                    samples, attn_mask, mask_ratio=mask_ratio
                )

            attn_mask_input_space, mask_input_space, combined_mask, mse_val, mae_val = \
                _compute_masked_metrics(samples, samples_hat, attn_mask, mask, args.patch_size)

            # NCC over the forecast window only (combined_mask = masked-and-valid)
            ncc_masked = statistics.ncc(samples, samples_hat, combined_mask, keep_batch=True).mean().item()

            total_mse += mse_val
            total_mae += mae_val
            total_ncc_masked += ncc_masked
            total_ncc_full += ncc.item()
            n_batches += 1

            last_batch = (samples, samples_hat, combined_mask, mask_input_space,
                          attn_mask_input_space)
    finally:
        model.downstream, model.include_forecasting, model.probabilistic_masking = saved

    if n_batches == 0:
        return {}

    avg_mse = total_mse / n_batches
    avg_mae = total_mae / n_batches
    avg_ncc_masked = total_ncc_masked / n_batches
    avg_ncc_full = total_ncc_full / n_batches

    online_history = {
        'online/forecast_mse': avg_mse,
        'online/forecast_mae': avg_mae,
        'online/forecast_ncc': avg_ncc_masked,
        'online/forecast_ncc_full': avg_ncc_full,
    }

    print(f"  [online forecast] mse={avg_mse:.4f}  mae={avg_mae:.4f}  "
          f"ncc(horizon)={avg_ncc_masked:.3f}  ncc(full)={avg_ncc_full:.3f}  ({len(X)} samples)")

    # reconstruction figure (main process only — wandb.Image is only consumed there)
    if misc.is_main_process() and last_batch is not None:
        samples_l, samples_hat_l, cm_l, m_l, am_l = last_batch
        try:
            online_history['online/forecast_reconstruction'] = _plot_reconstruction(
                samples_l, samples_hat_l, cm_l, m_l, am_l, idx=0
            )
        except Exception as e:
            print(f"  [online forecast] failed to plot reconstruction: {e}")

    return online_history


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    # switch to evaluation mode
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    test_history = {}
    embeddings = []

    # Optional losses — forecasting callers don't expose these CLI args.
    sd_weight = getattr(args, 'sd_weight', 0.0)
    koleo_weight = getattr(args, 'koleo_weight', 0.0)

    for batch in metric_logger.log_every(data_loader, 10, header):
        # push samples to device
        samples = batch[0]
        samples = samples.to(device, non_blocking=True)

        attn_mask = batch[1]
        attn_mask = attn_mask.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", dtype=_amp_dtype(args)):
            loss, ncc, cls_loss, mim_loss, cls_cos_sim, cls_std, samples_hat, mask, latent = model(samples,
                                                                                                   attn_mask,
                                                                                                   mask_ratio=args.mask_ratio)

        if args.save_embeddings:
            # latent of shape (B, 1+R+N', D)
            embedding = latent[:, :1, :].mean(dim=1) # (B, D), CLS token
            embeddings.append(embedding)

        batch_size = len(samples)

        # number of register tokens (model may be DDP-wrapped)
        R = (model.module.num_register_tokens if hasattr(model, "module") else model.num_register_tokens)
        S = 1 + R  # cls + register

        # Feature-quality diagnostics, over the REAL visible patch tokens only
        # (see _feature_quality_metrics; the train path uses the same helper).
        cls_patch_sim, register_patch_sim, patch_patch_sim, patch_rankme, \
            gap_rankme, cls_rankme = \
            _feature_quality_metrics(latent, _visible_real(model, latent), S, R)

        # KoLeo regulariser on encoder-output CLS tokens
        if koleo_weight > 0.0:
            cls_tokens = latent[:, 0, :]  # (B, D)
            koleo_gs = getattr(args, "koleo_group_size", 0)
            koleo = statistics.koleo_loss(cls_tokens, group_size=koleo_gs if koleo_gs and koleo_gs > 0 else None)
        else:
            koleo = torch.tensor(0.0, device=loss.device)

        loss_value = loss.item()
        ncc_value = ncc.item()
        cls_loss_value = cls_loss.item()
        mim_loss_value = mim_loss.item()
        cls_cos_sim_value = cls_cos_sim.item()
        cls_std_value = cls_std.item()
        cls_patch_sim_value = cls_patch_sim.item()
        register_patch_sim_value = register_patch_sim.item()
        patch_patch_sim_value = patch_patch_sim.item()
        rankme_value = patch_rankme.item()
        rankme_gap_value = gap_rankme.item()
        rankme_cls_value = cls_rankme.item()
        koleo_value = koleo.item()

        metric_logger.update(loss=loss_value)

        total_loss_value = (loss_value
                            + args.ncc_weight * (1 - ncc_value)
                            + sd_weight * cls_loss_value + sd_weight * mim_loss_value
                            + koleo_weight * koleo_value)
        metric_logger.meters['total_loss'].update(total_loss_value, n=batch_size)

        metric_logger.meters['ncc'].update(ncc_value, n=batch_size)
        metric_logger.meters['cls_loss'].update(cls_loss_value, n=batch_size)
        metric_logger.meters['mim_loss'].update(mim_loss_value, n=batch_size)
        metric_logger.meters['cls_cos_sim'].update(cls_cos_sim_value, n=batch_size)
        metric_logger.meters['cls_std'].update(cls_std_value, n=batch_size)
        metric_logger.meters['cls_patch_sim'].update(cls_patch_sim_value, n=batch_size)
        metric_logger.meters['register_patch_sim'].update(register_patch_sim_value, n=batch_size)
        metric_logger.meters['patch_patch_sim'].update(patch_patch_sim_value, n=batch_size)
        metric_logger.meters['rankme'].update(rankme_value, n=batch_size)
        metric_logger.meters['rankme_gap'].update(rankme_gap_value, n=batch_size)
        metric_logger.meters['rankme_cls'].update(rankme_cls_value, n=batch_size)
        metric_logger.meters['koleo'].update(koleo_value, n=batch_size)

        attn_mask_input_space, mask_input_space, combined_mask, mse_value, mae_value = \
            _compute_masked_metrics(samples, samples_hat, attn_mask, mask, args.patch_size)

        metric_logger.meters['mse'].update(mse_value, n=batch_size)
        metric_logger.meters['mae'].update(mae_value, n=batch_size)

    if args.save_embeddings and misc.is_main_process():
        embeddings = torch.cat(embeddings, dim=0).to(device="cpu", dtype=torch.float32).detach() # (B, D)
        
        embeddings_path = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        
        file_name = f"embeddings_{epoch}.pt"
        torch.save(embeddings, os.path.join(embeddings_path, file_name))

    # gather the stats from all processes (not in the final test pass, which
    # runs on the main process only: a collective there would deadlock)
    if epoch != -1:
        metric_logger.synchronize_between_processes()
    print("Averaged validation stats:", metric_logger)

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    # tensorboard
    if log_writer is not None:
        log_writer.add_scalar('val/val_total_loss', test_stats["total_loss"], epoch)
        log_writer.add_scalar('val/val_loss', test_stats["loss"], epoch)
        log_writer.add_scalar('val/val_ncc', test_stats["ncc"], epoch)
        log_writer.add_scalar('val/val_cls_loss', test_stats["cls_loss"], epoch)
        log_writer.add_scalar('val/val_mim_loss', test_stats["mim_loss"], epoch)
        log_writer.add_scalar('val/val_cls_cos_sim', test_stats["cls_cos_sim"], epoch)
        log_writer.add_scalar('val/val_cls_std', test_stats["cls_std"], epoch)
        log_writer.add_scalar('val/val_cls_patch_sim', test_stats["cls_patch_sim"], epoch)
        log_writer.add_scalar('val/val_register_patch_sim', test_stats["register_patch_sim"], epoch)
        log_writer.add_scalar('val/val_patch_patch_sim', test_stats["patch_patch_sim"], epoch)
        log_writer.add_scalar('val/val_rankme', test_stats["rankme"], epoch)
        log_writer.add_scalar('val/val_rankme_gap', test_stats["rankme_gap"], epoch)
        log_writer.add_scalar('val/val_rankme_cls', test_stats["rankme_cls"], epoch)
        log_writer.add_scalar('val/val_koleo', test_stats["koleo"], epoch)
        # evaluation only on the masked patches
        log_writer.add_scalar('val/val_mse', test_stats["mse"], epoch)
        log_writer.add_scalar('val/val_mae', test_stats["mae"], epoch)

    # wandb
    if args.wandb == True:
        test_history['epoch'] = epoch
        test_history['val_total_loss'] = test_stats["total_loss"]
        test_history['val_loss'] = test_stats["loss"]
        test_history['val_ncc'] = test_stats["ncc"]
        test_history['val_cls_loss'] = test_stats["cls_loss"]
        test_history['val_mim_loss'] = test_stats["mim_loss"]
        test_history['val_cls_cos_sim'] = test_stats["cls_cos_sim"]
        test_history['val_cls_std'] = test_stats["cls_std"]
        test_history['val_cls_patch_sim'] = test_stats["cls_patch_sim"]
        test_history['val_register_patch_sim'] = test_stats["register_patch_sim"]
        test_history['val_patch_patch_sim'] = test_stats["patch_patch_sim"]
        test_history['val_rankme'] = test_stats["rankme"]
        test_history['val_rankme_gap'] = test_stats["rankme_gap"]
        test_history['val_rankme_cls'] = test_stats["rankme_cls"]
        test_history['val_koleo'] = test_stats["koleo"]
        # evaluation only on the masked patches
        test_history['val_mse'] = test_stats["mse"]
        test_history['val_mae'] = test_stats["mae"]

        if (epoch % 5) == 0 or (epoch == args.epochs - 1):
            idx = random.randint(0, len(samples)-1)
            test_history["Val Reconstruction"] = _plot_reconstruction(
                samples, samples_hat, combined_mask, mask_input_space, attn_mask_input_space, idx)

    return test_stats, test_history