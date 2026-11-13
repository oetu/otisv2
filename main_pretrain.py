# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------
import os
import argparse
import math

import json
from typing import Tuple
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
# from torch.utils.tensorboard import SummaryWriter
import wandb
os.environ["WANDB__SERVICE_WAIT"] = "500"

from util.dataset import TimeSeriesDataset, TimeSeriesDatasetMmap
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop
from util.token_l2_norms import (
    track_token_l2_norms,
    plot_token_l2_norms,
    plot_last_layer_norm_evolution,
)

import models_otis
from sklearn.linear_model import LogisticRegression, LinearRegression

from engine_pretrain import train_one_epoch, evaluate_online, evaluate_online_kshot, evaluate_online_forecast, evaluate


def _build_context_schedule(start_steps, end_steps, num_epochs, patch_width):
    """Build a list of context lengths that gradually increase across annealing epochs.

    Each value is rounded up to a multiple of patch_width so that patching remains valid.
    """
    schedule = []
    for i in range(num_epochs):
        frac = i / max(num_epochs - 1, 1)
        raw = start_steps + (end_steps - start_steps) * frac
        aligned = int(math.ceil(raw / patch_width)) * patch_width
        schedule.append(min(aligned, end_steps))
    return schedule


# which metrics are "better when lower" vs "better when higher"
_DECREASING_METRICS = {"total_loss", "loss", "mse", "mae", "patch_patch_sim", "cls_loss", "mim_loss", "koleo"}
_BEST_INIT = {'epoch': -1, 'total_loss': np.inf, 'loss': np.inf, 'ncc': 0.0, 'cls_loss': np.inf, 'mim_loss': np.inf, 'koleo': np.inf, 'mse': np.inf, 'mae': np.inf, 'patch_patch_sim': np.inf, 'rankme': 0.0, 'rankme_gap': 0.0, 'rankme_cls': 0.0, 'kshot_cls_acc': 0.0}


def _update_best_eval_scores(best_eval_scores, new_value, reverse):
    """Track the top-K eval scores, evicting the worst when full.
    Returns True if new_value was accepted (i.e. checkpoint should be saved)."""
    comparator = max if not reverse else min
    if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
        best_eval_scores['count'] += 1
    elif new_value == comparator(new_value, *best_eval_scores['eval_criterion']):
        # new_value is not better than the worst tracked score
        return False
    else:
        best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=reverse)
        best_eval_scores['eval_criterion'].pop()
    best_eval_scores['eval_criterion'].append(new_value)
    return True


def _update_best_stats(best_stats, eval_stats):
    """Update best stats with min/max depending on the metric."""
    for key in best_stats:
        if key in eval_stats:
            fn = min if key in _DECREASING_METRICS else max
            best_stats[key] = fn(best_stats[key], eval_stats[key])


def get_args_parser():
    parser = argparse.ArgumentParser('OTISv2 pre-training', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='otis_base_dec160d4b', type=str, metavar='MODEL',
                        help='Name of model to train (default: otis_base_dec160d4b)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch compile')

    # RoPE parameter
    parser.add_argument('--rope_theta', default=200000.0, type=float,
                        help='Theta value for Rotary Positional Embeddings')
        
    parser.add_argument('--amp_dtype', default='bf16', type=str, choices=['fp16', 'bf16'],
                            help='CUDA AMP autocast dtype for the training forward pass (default bf16 = OTISv2 native; '
                                'fp16 matches OTIS legacy). GradScaler is auto-enabled under fp16.')

    # Register tokens
    parser.add_argument('--num_register_tokens', default=4, type=int,
                        help='Number of register tokens (default: 4)')
    
    parser.add_argument('--attention_sink', action=argparse.BooleanOptionalAction, default=True,
                        help='Enable per-head learnable attention sink logit '
                             '(StreamingLLM / DeepSeek-V4 trick). Default: enabled; pass --no-attention_sink to disable.')

    parser.add_argument('--univariate', action='store_true', default=False,
                        help='Univariate time series analysis (i.e. treat each variate independently)')

    parser.add_argument('--input_channels', type=int, default=1, metavar='N',
                        help='input channels')
    parser.add_argument('--time_steps', type=int, default=5000, metavar='N',
                        help='input length')
    parser.add_argument('--input_size', default=(1, 12, 5000), type=Tuple,
                        help='samples input size')
                        
    parser.add_argument('--patch_height', type=int, default=1, metavar='N',
                        help='patch height')
    parser.add_argument('--patch_width', type=int, default=100, metavar='N',
                        help='patch width')
    parser.add_argument('--patch_size', default=(1, 100), type=Tuple,
                        help='patch size')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate for encoder (default: 0.1)')
    parser.add_argument('--qk_norm', action='store_true', default=False,
                        help='Enable QK-norm in AxialRoPEAttention (default: QK-norm disabled).')
    parser.add_argument('--swiglu', action=argparse.BooleanOptionalAction, default=True,
                        help='Use SwiGLU MLP with SiLU activation in encoder + decoder blocks. '
                             'Pass --no-swiglu to fall back to the standard GELU MLP (default: enabled).')
    parser.add_argument('--zero_init_residual', action=argparse.BooleanOptionalAction, default=True,
                        help='Zero-init attn.proj and mlp.fc2 of every block so each block starts as identity. '
                             'Pass --no-zero_init_residual to keep the default xavier_uniform init (default: enabled).')
    parser.add_argument('--drop_path_decoder', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate for decoder (default: 0.1)')
    parser.add_argument('--layer_scale', type=float, default=None,
                        help='Layer scale init value (default: None, disabled)')

    parser.add_argument('--norm_patch_loss', action='store_true', default=False,
                        help='Use (per-patch) normalised values as targets for computing loss')
    parser.add_argument('--masked_patch_loss', action='store_true', default=False,
                        help='Compute loss only on masked patches')

    parser.add_argument('--ncc_weight', type=float, default=0.1,
                        help='Add normalised cross-correlation (ncc) as additional loss term')

    # iBOT L_CLS + L_MIM self-distillation hyperparameters (only used when --sd_weight > 0)
    parser.add_argument('--sd_weight', type=float, default=0.1,
                        help='Unified weight for iBOT self-distillation: applied to both L_CLS '
                             '(on the [CLS] token) and L_MIM (on the student VISIBLE patches) '
                             'following the DINOv3 convention of a single ibot.loss_weight. '
                             '0 = disables both branches.')
    parser.add_argument('--ibot_out_dim', type=int, default=8192,
                        help='Number of prototypes in the DINO projection head')
    parser.add_argument('--ibot_teacher_temp', type=float, default=0.07,
                        help='Softmax temperature applied to the teacher CLS logits')
    parser.add_argument('--ibot_student_temp', type=float, default=0.1,
                        help='Softmax temperature applied to the student CLS logits')
    parser.add_argument('--ibot_center_momentum', type=float, default=0.9,
                        help='EMA momentum for the teacher centre buffer')
    parser.add_argument('--ibot_ema_decay', type=float, default=0.994,
                        help='EMA decay for updating the teacher network from the student (base value)')
    parser.add_argument('--ibot_ema_cosine', action='store_true', default=False,
                        help='Use DINO-style cosine schedule for teacher EMA: m(t) = 1 - (1-m_base)*0.5*(1+cos(pi*t/T))')
    parser.add_argument('--ibot_teacher_temp_start', type=float, default=0.04,
                        help='Starting teacher softmax temperature for linear warmup')
    parser.add_argument('--ibot_teacher_temp_warmup_fraction', type=float, default=0.04,
                        help='Fraction of total epochs for linear warmup of teacher temp from '
                             '--ibot_teacher_temp_start to --ibot_teacher_temp (0 = disabled, hold at --ibot_teacher_temp)')
    parser.add_argument('--ibot_freeze_last_layer_fraction', type=float, default=0.02,
                        help='Fraction of total epochs over which student_head.last_layer is frozen (DINO/iBOT '
                             'trick; iBOT default is ~1/epochs, e.g. 0.00125 for 800 ep; 0 = disabled)')
    parser.add_argument('--ibot_teacher_mask_ratio', type=float, default=0.5,
                        help='Teacher mask ratio for iBOT L_CLS / L_MIM distillation. The teacher sees a random '
                             'superset of the student visible set of this size, so every token visible to '
                             'the student is also visible to the teacher. Must be <= --mask_ratio; at 0 the '
                             'teacher sees all tokens (vanilla iBOT), at --mask_ratio it matches the student.')
    parser.add_argument('--ibot_teacher_real_superset', action='store_true', default=False,
                        help='Build the teacher view from real patches: the student real patches plus extra real '
                             'patches (at least one, up to 1 - --ibot_teacher_mask_ratio of the real patches). '
                             'Random views get random extras; forecasting views are extended in time. Off (default): '
                             'the teacher view extends the student view along the student noise order, whose extra '
                             'slots go to padding first.')
    parser.add_argument('--ibot_teacher_forecast_mask_ratio', type=float, default=None,
                        help='Teacher mask ratio for forecasting views (requires --ibot_teacher_real_superset). '
                             'The teacher then sees time steps up to 1 - this ratio in every real variate. '
                             'None = use --ibot_teacher_mask_ratio for both masking schemes.')
    parser.add_argument('--ibot_teacher_drop_path', action=argparse.BooleanOptionalAction, default=True,
                        help='The EMA teacher keeps the drop path of the student, which is active in train mode '
                             '(default: enabled). Pass --no-ibot_teacher_drop_path to run the teacher without '
                             'stochastic depth.')
    parser.add_argument('--ibot_freeze_cls_head_only', action=argparse.BooleanOptionalAction, default=True,
                        help='The last-layer freeze at the start of training (--ibot_freeze_last_layer_fraction) '
                             'covers only the [CLS] head (default: enabled). Pass --no-ibot_freeze_cls_head_only '
                             'to also cover the separate patch (MIM) head, as in iBOT.')
    parser.add_argument('--ibot_share_head', action='store_true', default=False,
                        help='Share the DINOHead between L_CLS and L_MIM (ablation). '
                             'Default: separate heads (DINOv3 default, assert cfg.ibot.separate_head=True).')

    parser.add_argument('--koleo_weight', type=float, default=0.025,
                        help='Weight for KoLeo regulariser on CLS tokens (0 = disabled)')
    parser.add_argument('--koleo_group_size', type=int, default=32,
                        help='KoLeo group size: apply KoLeo on non-overlapping groups of this many CLS tokens '
                             'and average (DINOv3 §3.2). Set <=0 to disable grouping (use full batch).')

    # Gram anchor loss (DINOv3 §4.2). Activates late in training; anchors the patch
    # similarity matrix X X^T to a frozen snapshot of the EMA teacher at high resolution.
    parser.add_argument('--gram_weight', type=float, default=0.0,
                        help='Weight for the Gram anchor loss (DINOv3 §4.2). 0 = disabled.')
    parser.add_argument('--gram_downsample_factor', type=int, default=2,
                        help='Temporal downsample factor applied to the student input for the Gram branch '
                             '(teacher sees native resolution). Default: 2.')
    parser.add_argument('--gram_refresh_every_n_epochs', type=int, default=1,
                        help='Refresh the frozen Gram teacher from the current EMA teacher every N epochs '
                             'during the active phase (default: 1, i.e. once per epoch).')
    parser.add_argument('--gram_start_fraction', type=float, default=None,
                        help='Fraction of total epochs at which Gram activates. Defaults to the start of the '
                             'WSD annealing phase (1 - decay_fraction).')
    parser.add_argument('--pre_gram_checkpoint', action='store_true', default=False,
                        help='If set, save a checkpoint at end of (gram_start_epoch - 1) to output_dir')

    # Augmentation parameters
    parser.add_argument('--probabilistic_masking', action='store_true', default=False,
                        help='Randomly vary the masking ratio during pretraining.')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')
    parser.add_argument('--include_forecasting', action='store_true', default=False,
                        help='Include forecasting during pretraining (i.e. right-sided masking).')
    parser.add_argument('--forecasting_probability', default=0.25, type=float,
                        help='Probability for forecasting (i.e. right-sided masking).')
    parser.add_argument('--forecasting_mask_ratio', default=0.5, type=float,
                        help='Masking ratio for forecasting (percentage of removed patches).')

    parser.add_argument('--crop_lower_bnd', default=0.5, type=float,
                        help='Lower boundary of the cropping ratio (default: 0.5)')
    parser.add_argument('--crop_upper_bnd', default=1.0, type=float,
                        help='Upper boundary of the cropping ratio (default: 1.0)')

    parser.add_argument('--jitter_sigma', default=0.2, type=float,
                        help='Jitter sigma N(0, sigma) (default: 0.2)')
    parser.add_argument('--rescaling_sigma', default=0.5, type=float,
                        help='Rescaling sigma N(0, sigma) (default: 0.5)')
    # deprecated no-op: no transform uses it; kept (hidden, default None) so that
    # existing launchers that still pass --freq_shift_delta keep parsing
    parser.add_argument('--freq_shift_delta', type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--ft_surr_phase_noise', default=0.075, type=float,
                        help='Phase noise magnitude (default: 0.075)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.1)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Gradient clipping max-norm (default: None; iBOT/DINO recommend 3.0 when --sd_weight > 0)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 32')
    parser.add_argument('--min_lr', type=float, default=None, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (should be 0.1x of peak lr)')

    parser.add_argument('--layer_decay', type=float, default=0.975,
                        help='layer-wise lr decay (default: 0.975)')
    parser.add_argument('--patch_embed_lr_mult', type=float, default=1.0,
                        help='Multiplier applied on top of layer-wise decay to patch_embed '
                             'parameters (default: 1.0; DINOv3 / BEiT use 0.2 to stabilise '
                             'long pre-training runs by slowing the patch projection).')
    parser.add_argument('--student_head_wd_mult', type=float, default=1.0,
                        help='WD multiplier for student_head (CLS / DINO branch) parameters. '
                             'Default 1.0 (no-op). Higher values apply ViT-2022-style '
                             'high-WD-on-head pressure. Has no effect on '
                             'student_head.last_layer, which is weight-normalised.')

    # MUON optimizer parameters
    parser.add_argument('--use_muon', action='store_true', default=False,
                        help='Use MUON optimizer for 2D hidden weight matrices')
    parser.add_argument('--muon_lr', type=float, default=None, metavar='LR',
                        help='MUON learning rate, absolute (overrides muon_blr scaling)')
    parser.add_argument('--muon_blr', type=float, default=0.02, metavar='LR',
                        help='MUON base learning rate, scaled with batch size (default: 0.02)')
    parser.add_argument('--muon_momentum', type=float, default=0.95,
                        help='MUON momentum coefficient (default: 0.95)')
    parser.add_argument('--muon_weight_decay', type=float, default=0.01,
                        help='MUON weight decay (default: 0.01)')

    parser.add_argument('--warmup_fraction', type=float, default=0.1,
                        help='Fraction of total epochs for the WSD warmup phase (default: 0.1)')
    parser.add_argument('--decay_fraction', type=float, default=0.1,
                        help='Fraction of total epochs for the WSD decay/annealing phase (default: 0.1)')
    parser.add_argument('--subepoch_lr_schedule', action=argparse.BooleanOptionalAction, default=True,
                        help='Use sub-epoch (fractional) warmup/decay lengths instead of int(epochs*fraction). '
                             'Ensures short runs (epochs <= 9, where int(epochs*0.1)=0) still warm up and anneal '
                             'to min_lr, rather than training at constant peak LR. Required for clean '
                             'data-constrained scaling sweeps that include low-epoch cells. On by default; '
                             'pass --no-subepoch_lr_schedule for a legacy constant-peak schedule.')

    # Annealing parameters
    parser.add_argument('--annealing_data_path', default='', type=str,
                        help='Path to dataset used for the annealing phase (default: disabled)')
    parser.add_argument('--annealing_time_steps', type=int, default=4096,
                        help='Final context length during annealing phase (default: 4096)')
    parser.add_argument('--annealing_rewarmup_epochs', type=int, default=0,
                        help='Number of re-warmup epochs at the start of the annealing phase (default: 0)')
    parser.add_argument('--annealing_context_schedule', action='store_true', default=False,
                        help='Gradually increase context length across annealing epochs instead of jumping')
    parser.add_argument('--pre_annealing_checkpoint', action='store_true', default=False,
                        help='If set, save a checkpoint at end of (decay_start_epoch - 1) to output_dir')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # * Finetuning params
    parser.add_argument('--pretrained_encoder', default='',
                        help='load encoder from checkpoint')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='make encoder (i.e. the feature extractor) non-trainable, i.e., only train the decoder')
    # Dataset parameters
    eval_criterions = ['epoch', 'total_loss', 'loss', 'ncc', 'cls_loss', 'mim_loss', 'koleo', 'mse', 'mae', 'patch_patch_sim', 'rankme', 'rankme_gap', 'rankme_cls', 'kshot_cls_acc']
    parser.add_argument('--eval_criterion', default='patch_patch_sim', type=str, choices=eval_criterions,
                        help='pretraining evaluation metric (default: patch_patch_sim)')
    
    parser.add_argument('--data_path', default='_.pt', type=str,
                        help='dataset path')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path')
    parser.add_argument('--subset_tokens', default=None, type=int,
                        help='If set, restrict training to a stratified, '
                             'deterministic subset of this many tokens-per-epoch '
                             '(V * T/P summed). Subsets are nested supersets across '
                             'budgets given the same --subset_seed and --data_path.')
    parser.add_argument('--subset_seed', default=2026, type=int,
                        help='Seed for the per-domain permutation that defines '
                             'the token-budget subset (see --subset_tokens).')
    
    parser.add_argument('--online_evaluation', action='store_true', default=False,
                        help='Perform online evaluation of a downstream task')
    parser.add_argument('--online_evaluation_task', default='classification', type=str,
                        help='Online downstream task (default: classification)')
    parser.add_argument('--online_num_classes', default=2, type=int,
                        help='Online classification task classes (default: 2)')
    parser.add_argument('--online_kshot', action='store_true', default=False,
                        help='5-shot prototypical evaluation on 23 UEA datasets')
    parser.add_argument('--online_kshot_data_base', default='', type=str,
                        help='Base directory containing UEA dataset folders')
    parser.add_argument('--online_kshot_intra_epoch_steps', default=0, type=int,
                        help='Run online k-shot eval every N mini-batches inside an epoch '
                             '(0 = disabled). '
                             'Records appended to <output_dir>/intra_kshot.jsonl.')
    parser.add_argument('--online_forecast', action='store_true', default=False,
                        help='Online forecasting evaluation on synthetic sine waves')
    parser.add_argument('--online_forecast_data_path', default='', type=str,
                        help='Path to the data file used for online forecasting')
    parser.add_argument('--online_forecast_time_steps', default=2400, type=int,
                        help='Number of leading time steps used from each sample')
    parser.add_argument('--online_forecast_mask_ratio', default=0.25, type=float,
                        help='Right-sided forecast horizon as a fraction of the used time steps')
    
    parser.add_argument('--lower_bnd', type=int, default=0, metavar='N',
                        help='lower_bnd')
    parser.add_argument('--upper_bnd', type=int, default=0, metavar='N',
                        help='upper_bnd')

    parser.add_argument('--data_path_online', default='_.pt', type=str,
                        help='dataset path for the online evaluation')
    parser.add_argument('--labels_path_online', default='_.pt', type=str,
                        help='labels path for the online evaluation')
    parser.add_argument('--labels_mask_path_online', default='', type=str,
                        help='labels path (default: None)')
    
    parser.add_argument('--val_data_path_online', default='', type=str,
                        help='validation dataset path for the online evaluation')
    parser.add_argument('--val_labels_path_online', default='', type=str,
                        help='validation labels path for the online evaluation')
    parser.add_argument('--val_labels_mask_path_online', default='', type=str,
                        help='labels path (default: None)')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='',
                        help='path where to tensorboard log (default: ./logs)')
    parser.add_argument('--wandb', action='store_true', default=False)
    parser.add_argument('--wandb_entity', default='', type=str,
                        help='entity of the current run')
    parser.add_argument('--wandb_project', default='', type=str,
                        help='project where to wandb log')
    parser.add_argument('--wandb_id', default='', type=str,
                        help='id of the current run')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--save_embeddings', action='store_true', default=False,
                        help='save encoder embeddings (i.e. of visible tokens)')

    # Token L2-norm tracking (à la "Vision Transformers Need Registers" Fig. 4a)
    parser.add_argument('--track_token_norms', action='store_true', default=False,
                        help='Track per-layer L2 norm of patch tokens (encoder + decoder) '
                             'on a fixed subset of pre-training samples and log to wandb')
    parser.add_argument('--track_token_norms_samples', type=int, default=4096,
                        help='Number of random samples used for the token-norm diagnostic '
                             '(default: 4096)')
    parser.add_argument('--track_token_norms_freq', type=int, default=1,
                        help='How often (in epochs) to run the token-norm diagnostic '
                             '(default: 1, every epoch)')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=24, type=int)
    parser.add_argument('--pin_mem', action='store_true', default=True,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')

    # Distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor)')

    return parser


def main(args):
    args.patch_size = (args.patch_height, args.patch_width)

    if args.eval_criterion == 'kshot_cls_acc' and not args.online_kshot:
        raise ValueError("--eval_criterion kshot_cls_acc requires --online_kshot to be enabled")

    print(f"cuda devices: {torch.cuda.device_count()}")
    misc.init_distributed_mode(args)
    # args.distributed = False

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    print(f"rank: {misc.get_rank()}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True

    # load data
    dataset_train = TimeSeriesDatasetMmap(data_path=args.data_path,
                                          univariate=args.univariate,
                                          train=True,
                                          args=args,
                                          subset_tokens=args.subset_tokens,
                                          subset_seed=args.subset_seed)
    print("Training set size: ", len(dataset_train))

    if args.val_data_path:
        dataset_val = TimeSeriesDatasetMmap(data_path=args.val_data_path,
                                            univariate=args.univariate,
                                            train=False,
                                            args=args)
        print("Validation set size: ", len(dataset_val))

    num_tasks = misc.get_world_size()
    print(f"num_tasks: {num_tasks}")
    global_rank = misc.get_rank()
    print(f"global_rank: {global_rank}")
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)

    if args.val_data_path:
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                    'This will slightly alter validation results as extra duplicate entries are added to achieve '
                    'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    log_writer = None

    # wandb logging
    if args.wandb == True and misc.is_main_process():
        config = vars(args)
        wandb_settings = wandb.Settings(init_timeout=600)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config, entity=args.wandb_entity, settings=wandb_settings)
        else:
            wandb.init(project=args.wandb_project, config=config, entity=args.wandb_entity, settings=wandb_settings)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_train.collate_fn,
        pin_memory=args.pin_mem,
        drop_last=False,
        persistent_workers=True,
    )

    if args.val_data_path:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            # shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset_val.collate_fn,
            pin_memory=args.pin_mem,
            drop_last=False,
            persistent_workers=True,
        )

    # Token-L2-norm tracking: build a fixed subset dataloader on the main
    # process so the per-layer norm diagnostic is computed on the same set of
    # samples each epoch (and only once per training run, not per rank). We
    # spin up a *separate* dataset instance with a frozen copy of ``args`` so
    # the in-place mutations performed during the annealing phase
    # (``args.time_steps``, ``args.crop_lower_bnd``, ``args.jitter_sigma``, …)
    # do not silently change which tokens are tracked across epochs.
    token_norm_loader = None
    norm_track_n = 0
    if args.track_token_norms and misc.is_main_process():
        import copy as _copy
        track_args = _copy.copy(args)  # shallow copy: args holds only scalars
        # train=False: a clean view (first time_steps window, no crop / jitter /
        # gain), so the epoch-to-epoch norm curves show training drift only
        dataset_train_track = TimeSeriesDatasetMmap(
            data_path=args.data_path,
            univariate=args.univariate,
            train=False,
            N_val=1,
            args=track_args,
        )
        norm_track_n = min(args.track_token_norms_samples, len(dataset_train_track))
        norm_track_rng = np.random.default_rng(seed=42)
        norm_track_indices = norm_track_rng.choice(
            len(dataset_train_track), size=norm_track_n, replace=False).tolist()
        token_norm_subset = torch.utils.data.Subset(dataset_train_track, norm_track_indices)
        token_norm_loader = torch.utils.data.DataLoader(
            token_norm_subset,
            batch_size=args.batch_size,
            num_workers=min(4, args.num_workers),
            collate_fn=TimeSeriesDatasetMmap.collate_fn,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
            persistent_workers=True,
        )
        print(f"Token-L2-norm tracking enabled: {norm_track_n} fixed samples, "
              f"every {args.track_token_norms_freq} epoch(s)")

    # History of last-block (pre-final-norm) patch-token L2 norms, one entry
    # per diagnostic call. Used at end-of-training to render a Fig 4(b)-style
    # multi-curve density plot showing the gradual emergence of the high-norm
    # outlier tail across pre-training.
    last_layer_norm_history_enc = []
    last_layer_norm_history_dec = []
    last_layer_norm_enc_dim = None
    last_layer_norm_dec_dim = None

    # Per-layer token L2 norms snapshot taken at the final epoch — kept around
    # so the multi-layer "Token L2 Norms" figure can be re-rendered offline.
    final_token_norms_enc = None
    final_token_norms_dec = None
    final_token_norms_epoch = None

    # online evaluation
    if args.online_evaluation:
        dataset_online_train = TimeSeriesDataset(data_path=args.data_path_online, 
                                                 labels_path=args.labels_path_online, 
                                                 labels_mask_path=args.labels_mask_path_online, 
                                                 downstream_task=args.online_evaluation_task, 
                                                 univariate=args.univariate,
                                                 train=True, 
                                                 args=args)
        dataset_online_val = TimeSeriesDataset(data_path=args.val_data_path_online,
                                               labels_path=args.val_labels_path_online,
                                               labels_mask_path=args.val_labels_mask_path_online,
                                               downstream_task=args.online_evaluation_task,
                                               univariate=args.univariate,
                                               train=False,
                                               N_val=5,
                                               args=args)

        print("Online training set size: ", len(dataset_online_train))
        print("Online validation set size: ", len(dataset_online_val))

        sampler_online_train = torch.utils.data.DistributedSampler(
            dataset_online_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

        if args.dist_eval:
            if len(dataset_online_val) % num_tasks != 0:
                print('Warning: Enabling distributed online evaluation with an eval dataset not divisible '
                      'by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added '
                      'to achieve equal num of samples per-process.')
            sampler_online_val = torch.utils.data.DistributedSampler(
                dataset_online_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
            )
        else:
            sampler_online_val = torch.utils.data.SequentialSampler(dataset_online_val)

        data_loader_online_train = torch.utils.data.DataLoader(
            dataset_online_train, 
            sampler=sampler_online_train,
            shuffle=False,
            batch_size=128,
            num_workers=args.num_workers,
            collate_fn=dataset_online_train.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        data_loader_online_val = torch.utils.data.DataLoader(
            dataset_online_val, 
            sampler=sampler_online_val,
            shuffle=False,
            batch_size=128,
            num_workers=args.num_workers,
            collate_fn=dataset_online_val.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

    # define the model
    model = models_otis.__dict__[args.model](
        input_channels=args.input_channels,
        time_steps=args.time_steps,
        patch_size=args.patch_size,
        norm_patch_loss=args.norm_patch_loss,
        masked_patch_loss=args.masked_patch_loss,
        self_distillation_loss=(args.sd_weight > 0.0),
        ibot_out_dim=args.ibot_out_dim,
        ibot_teacher_temp=args.ibot_teacher_temp,
        ibot_student_temp=args.ibot_student_temp,
        ibot_center_momentum=args.ibot_center_momentum,
        ibot_ema_decay=args.ibot_ema_decay,
        ibot_teacher_mask_ratio=args.ibot_teacher_mask_ratio,
        ibot_teacher_real_superset=args.ibot_teacher_real_superset,
        ibot_teacher_forecast_mask_ratio=args.ibot_teacher_forecast_mask_ratio,
        ibot_teacher_drop_path=args.ibot_teacher_drop_path,
        ibot_freeze_cls_head_only=args.ibot_freeze_cls_head_only,
        ibot_share_head=args.ibot_share_head,
        gram_enabled=(args.gram_weight > 0.0),
        probabilistic_masking=args.probabilistic_masking,
        include_forecasting=args.include_forecasting,
        forecasting_probability=args.forecasting_probability,
        forecasting_mask_ratio=args.forecasting_mask_ratio,
        drop_path=args.drop_path,
        drop_path_decoder=args.drop_path_decoder,
        layer_scale=args.layer_scale,
        rope_theta=args.rope_theta,
        qk_norm=args.qk_norm,
        swiglu=args.swiglu,
        zero_init_residual=args.zero_init_residual,
        attention_sink=args.attention_sink,
        num_register_tokens=args.num_register_tokens
    )

    new_patch_size = False
    if args.pretrained_encoder:
        checkpoint = torch.load(args.pretrained_encoder, map_location='cpu', weights_only=False)

        print("Load pretrained encoder from: %s" % args.pretrained_encoder)
        checkpoint_model = checkpoint['model']
        resync_teacher = False   # set when teacher weights cannot be taken from the checkpoint

        # check if new and old patch_size match
        nb_channels_ckpt = checkpoint_model['patch_embed.proj.weight'].shape[-3]
        nb_channels_model = args.input_size[0]

        checkpoint_patch_size = checkpoint_model['patch_embed.proj.weight'].shape[-2:]
        patch_height_ckpt, patch_width_ckpt = checkpoint_patch_size[0], checkpoint_patch_size[1]
        patch_height_model, patch_width_model = args.patch_size[0], args.patch_size[1]

        if nb_channels_ckpt != nb_channels_model or patch_height_ckpt != patch_height_model or patch_width_ckpt != patch_width_model:
            new_patch_size = True
            # initialise new patch_embed module (student and EMA teacher)
            for key in ["patch_embed.proj.weight", "patch_embed.proj.bias",
                        "patch_embed.norm.weight", "patch_embed.norm.bias",
                        "teacher_patch_embed.proj.weight", "teacher_patch_embed.proj.bias",
                        "teacher_patch_embed.norm.weight", "teacher_patch_embed.norm.bias"]:
                if key in checkpoint_model:
                    print(f"Removing key {key} from pretrained checkpoint")
                    del checkpoint_model[key]
            print("Initialising new patch_embed")

            # initialise new decoder_pred module
            for key in ["decoder_pred.weight", "decoder_pred.bias"]:
                if key in checkpoint_model:
                    print(f"Removing key {key} from pretrained checkpoint")
                    del checkpoint_model[key]
            print("Initialising new decoder_pred")

        # initialise new decoder
        print("Initialising new decoder")
        # initialise new decoder_embed, decoder_blocks, decoder_norm, decoder_pred
        for key in list(checkpoint_model.keys()):
            if "decoder" in key:
                print(f"Removing key {key} from pretrained checkpoint")
                del checkpoint_model[key]
                print(f"Initialising new {key}")

        # initialise new mask_token
        key = "mask_token"
        if key in checkpoint_model:
            print(f"Removing key {key} from pretrained checkpoint")
            del checkpoint_model[key]
            print(f"Initialising new {key}")

        # Drop any legacy domain_token keys (no longer part of the architecture)
        checkpoint_model.pop('domain_token', None)

        # handle register_tokens shape mismatch
        if 'register_tokens' in checkpoint_model:
            ckpt_shape = checkpoint_model['register_tokens'].shape
            model_shape = model.register_tokens.shape
            if ckpt_shape != model_shape:
                print(f"Removing key register_tokens from pretrained checkpoint "
                      f"(shape {list(ckpt_shape)} != {list(model_shape)})")
                del checkpoint_model['register_tokens']
                print("Initialising new register_tokens")
                # the teacher copy has the same shape mismatch; it is re-synced below
                checkpoint_model.pop('teacher_register_tokens', None)
                resync_teacher = True

        # load pretrained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)

        # Self-distillation: the teacher was snapshotted from the random student
        # in __init__. Without teacher weights in the checkpoint (trimmed release
        # or non-SD checkpoint), or with a new patch embedding, re-sync it from
        # the loaded student so the SD targets do not start from noise.
        if getattr(model, "self_distillation_loss", False):
            has_teacher = any(k.startswith("teacher_") for k in checkpoint_model)
            if not has_teacher or new_patch_size or resync_teacher:
                model.sync_teacher_from_student()
                print("Teacher re-synced from the loaded student "
                      f"({'no teacher weights in checkpoint' if not has_teacher else 'new patch size / register count'})")

    skip_list = []
    if args.pretrained_encoder and args.freeze_encoder:
        # partially freeze the mode:
        # freeze patch_embed
        for n, p in model.patch_embed.named_parameters():
            p.requires_grad = False
            skip_list.append(f"patch_embed.{n}")
        # freeze encoder
        for n, p in model.blocks[:].named_parameters():
            p.requires_grad = False
            skip_list.append(f"blocks.{n}")
        # freeze norm
        for n, p in model.norm.named_parameters():
            p.requires_grad = False
            skip_list.append(f"norm.{n}")

    if new_patch_size == True:
        # unfreeze patch_embed
        for n, p in model.patch_embed.named_parameters():
            p.requires_grad = True
            skip_list = [module for module in skip_list if "patch_embed" not in module]
        # unfreeze norm
        for n, p in model.norm.named_parameters():
            p.requires_grad = True
            skip_list = [module for module in skip_list if module not in ["norm.weight", "norm.bias"]]
    
    print(skip_list)

    if args.compile:
        model.forward = torch.compile(model.forward, dynamic=True)
    model.to(device, non_blocking=True)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_encoder_prefixes = ("decoder", "projector", "predictor")
    n_params_encoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not any(n.startswith(k) for k in non_encoder_prefixes))
    n_params_decoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "decoder" in n)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))
    print('Number of params (M): %.2f' % (n_parameters / 1.e6))
    print('Number of encoder params (M): %.2f' % (n_params_encoder / 1.e6))
    print('Number of decoder params (M): %.2f' % (n_params_decoder / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 32

    if args.min_lr is None:
        args.min_lr = args.lr * 0.1

    # MUON LR: sub-linear batch size scaling (exponent ~0.5, per modded-nanogpt)
    if args.use_muon and args.muon_lr is None:
        args.muon_lr = args.muon_blr * (eff_batch_size / 32) ** 0.5

    print("base lr: %.2e" % (args.lr * 32 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    if args.use_muon:
        print("muon lr: %.2e" % args.muon_lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    
    # following timm: set wd as 0 for bias and norm layers
    param_groups = misc.add_weight_decay_timm_lrd(model_without_ddp, args.weight_decay,
                                                   layer_decay=args.layer_decay,
                                                   patch_embed_lr_mult=args.patch_embed_lr_mult,
                                                   student_head_wd_mult=args.student_head_wd_mult,
                                                   use_muon=args.use_muon)
    optimizer = misc.build_optimizer(param_groups, args)
    print(optimizer)
    loss_scaler = NativeScaler(amp_dtype=args.amp_dtype)

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    # On resume, rehydrate the last-block token-L2-norm history from disk so the
    # next save of last_layer_norms.pt preserves epochs trained before the
    # interruption. Only entries strictly before args.start_epoch are kept,
    # to avoid duplicating the resumed epoch.
    if args.resume and args.output_dir and misc.is_main_process():
        prev_history_path = os.path.join(args.output_dir, "last_layer_norms.pt")
        if os.path.isfile(prev_history_path):
            prev = torch.load(prev_history_path, map_location="cpu", weights_only=False)
            for ep, t in (prev.get("encoder") or []):
                if ep < args.start_epoch:
                    last_layer_norm_history_enc.append((ep, t))
            for ep, t in (prev.get("decoder") or []):
                if ep < args.start_epoch:
                    last_layer_norm_history_dec.append((ep, t))
            last_layer_norm_enc_dim = prev.get("encoder_embed_dim", last_layer_norm_enc_dim)
            last_layer_norm_dec_dim = prev.get("decoder_embed_dim", last_layer_norm_dec_dim)
            print(f"Resumed last_layer_norms history: "
                  f"{len(last_layer_norm_history_enc)} enc / "
                  f"{len(last_layer_norm_history_dec)} dec entries pre-{args.start_epoch}")

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)

    # compute annealing start epoch (tied to WSD decay phase)
    decay_start_epoch = args.epochs - int(args.epochs * args.decay_fraction)
    annealing_activated = False
    annealing_epochs = args.epochs - decay_start_epoch
    original_time_steps = args.time_steps

    # Gram anchor activation (DINOv3 §4.2). Defaults to the start of the WSD
    # annealing phase; can be overridden via --gram_start_fraction.
    gram_enabled = args.gram_weight > 0.0
    if args.gram_start_fraction is None:
        gram_start_epoch = decay_start_epoch
    else:
        gram_start_epoch = int(args.epochs * args.gram_start_fraction)
    # a run resumed after the Gram start keeps the Gram snapshot loaded from the
    # checkpoint (the periodic refresh below continues on its usual epochs)
    gram_activated = gram_enabled and args.start_epoch > gram_start_epoch

    # Annealing is triggered either by a dataset switch (an explicit annealing
    # data path) or by a context-length schedule on the pretrain data. If
    # neither is set, annealing stays off.
    annealing_data_switch = bool(args.annealing_data_path)
    annealing_enabled = annealing_data_switch or args.annealing_context_schedule

    # pre-compute context length schedule for gradual extension
    if args.annealing_context_schedule and annealing_epochs > 0:
        annealing_context_steps = _build_context_schedule(
            original_time_steps, args.annealing_time_steps, annealing_epochs, args.patch_width)
    else:
        annealing_context_steps = None

    print(f"Start training for {args.epochs} epochs")
    if annealing_enabled:
        print(f"Annealing phase starts at epoch {decay_start_epoch} "
              f"(data_switch: {annealing_data_switch}, "
              f"data: {args.annealing_data_path or '-'}, "
              f"time_steps: {args.annealing_time_steps}, "
              f"rewarmup: {args.annealing_rewarmup_epochs}, "
              f"context_schedule: {args.annealing_context_schedule})")

    # save original augmentation / batch settings for annealing scaling
    orig_jitter_sigma = args.jitter_sigma
    orig_rescaling_sigma = args.rescaling_sigma
    orig_ft_surr_phase_noise = args.ft_surr_phase_noise
    orig_crop_lower_bnd = args.crop_lower_bnd
    orig_batch_size = args.batch_size
    orig_accum_iter = args.accum_iter

    # keep references needed after dataset_train is freed during annealing
    train_dataset_len = len(dataset_train)

    best_stats = _BEST_INIT.copy()
    best_eval_scores = {'count':1, 'nb_ckpts_max':3, 'eval_criterion':[best_stats[args.eval_criterion]]}
    # On resume, rebuild the top-K list from this run's checkpoint files, so the
    # next evaluations are compared with the scores already on disk (state after
    # k saves: [sentinel] + k scores with count k+1 while k < K, else the K best).
    if args.resume and args.output_dir and args.eval_criterion != "epoch":
        _nb = best_eval_scores['nb_ckpts_max']
        _scores = sorted(misc.checkpoint_scores(args.output_dir, args.eval_criterion),
                         reverse=args.eval_criterion not in _DECREASING_METRICS)
        if _scores:
            if len(_scores) < _nb:
                best_eval_scores['eval_criterion'] = [best_stats[args.eval_criterion]] + _scores
                best_eval_scores['count'] = len(_scores) + 1
            else:
                best_eval_scores['eval_criterion'] = _scores[:_nb]
                best_eval_scores['count'] = _nb
            print(f"Resume: top-{_nb} '{args.eval_criterion}' scores from disk: {_scores[:_nb]}")
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        # --- annealing phase activation (once) ---
        # ">=": a run resumed inside the annealing phase activates it at its first epoch
        if annealing_enabled and epoch >= decay_start_epoch and not annealing_activated:
            annealing_activated = True
            resumed_in_annealing = epoch > decay_start_epoch

            # reduce augmentations (don't eliminate — keeps regularization)
            args.jitter_sigma = orig_jitter_sigma * 0.25
            args.rescaling_sigma = orig_rescaling_sigma * 0.5
            args.ft_surr_phase_noise = orig_ft_surr_phase_noise * 0.5
            # crop strength is a bound with identity at 1.0 (not a magnitude
            # with identity at 0), so anneal it towards 1.0 — never below the
            # configured value, else a run with cropping disabled
            # (crop_lower_bnd 1.0) would have it silently re-enabled here
            args.crop_lower_bnd = max(orig_crop_lower_bnd, 0.8)
            args.crop_upper_bnd = 1.0

            # keep the same masking strategy (don't switch to forecasting-only)
            # keep the same loss formulation (don't switch to masked-only)
            model_without_ddp.activate_masked_loss()

            # set initial context length for annealing
            if annealing_context_steps is not None:
                args.time_steps = annealing_context_steps[0]
            elif annealing_data_switch:
                args.time_steps = args.annealing_time_steps

            # reset optimizer state for the new data distribution (MiniCPM-style);
            # on a resume inside annealing the loaded optimizer state already is the
            # post-reset state, so it is kept
            if not resumed_in_annealing:
                param_groups = misc.add_weight_decay_timm_lrd(model_without_ddp, args.weight_decay,
                                                               layer_decay=args.layer_decay,
                                                               patch_embed_lr_mult=args.patch_embed_lr_mult,
                                                               student_head_wd_mult=args.student_head_wd_mult,
                                                               use_muon=args.use_muon)
                optimizer = misc.build_optimizer(param_groups, args)
                loss_scaler = NativeScaler(amp_dtype=args.amp_dtype)

            # scale batch size inversely with the SQUARE of context length ratio to
            # account for quadratic attention memory: O((T/patch_width)²)
            args.batch_size = max(1, int(orig_batch_size * (original_time_steps / args.time_steps) ** 1.5))
            args.accum_iter = max(1, round(orig_batch_size * orig_accum_iter / args.batch_size))

            if annealing_data_switch:
                annealing_dataset = TimeSeriesDatasetMmap(data_path=args.annealing_data_path,
                                                           univariate=args.univariate,
                                                           train=True, args=args)
                dataset_desc = f"dataset {args.annealing_data_path}, size={len(annealing_dataset)}"
                # free the original pre-training dataset (no longer needed)
                del dataset_train
            else:
                # No annealing paths: keep the pretrain dataset and let the
                # context-length schedule (and reduced augmentations / reset
                # optimizer / rewarmup) do the annealing work on it.
                annealing_dataset = dataset_train
                dataset_desc = f"pretrain dataset retained, size={len(annealing_dataset)}"

            sampler_train = torch.utils.data.DistributedSampler(
                annealing_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
            data_loader_train = torch.utils.data.DataLoader(
                annealing_dataset,
                sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                collate_fn=TimeSeriesDatasetMmap.collate_fn,
                pin_memory=args.pin_mem,
                drop_last=False,
                persistent_workers=True,
            )
            print(f"Epoch {epoch}: Annealing phase activated — {dataset_desc}, "
                  f"time_steps={args.time_steps}, "
                  f"batch_size={args.batch_size} (was {orig_batch_size}), "
                  f"accum_iter={args.accum_iter} (was {orig_accum_iter}), "
                  f"{'optimizer kept (resumed inside annealing)' if resumed_in_annealing else 'optimizer reset'}, "
                  f"augmentations reduced")

        # --- gradual context extension within annealing ---
        if annealing_activated and annealing_context_steps is not None:
            annealing_epoch_idx = epoch - decay_start_epoch
            new_time_steps = annealing_context_steps[min(annealing_epoch_idx, len(annealing_context_steps) - 1)]
            if new_time_steps != args.time_steps:
                args.time_steps = new_time_steps

                # rescale batch size for the new context length (quadratic attention)
                args.batch_size = max(1, int(orig_batch_size * (original_time_steps / args.time_steps) ** 1.5))
                args.accum_iter = max(1, round(orig_batch_size * orig_accum_iter / args.batch_size))

                # rebuild dataloader with updated batch size
                sampler_train = torch.utils.data.DistributedSampler(
                    annealing_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                data_loader_train = torch.utils.data.DataLoader(
                    annealing_dataset,
                    sampler=sampler_train,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    collate_fn=TimeSeriesDatasetMmap.collate_fn,
                    pin_memory=args.pin_mem,
                    drop_last=False,
                    persistent_workers=True,
                )
                print(f"Epoch {epoch}: Context length → {args.time_steps}, "
                      f"batch_size → {args.batch_size}, accum_iter → {args.accum_iter}")

        data_loader_train.sampler.set_epoch(epoch)

        # --- Gram anchor: (re)snapshot teacher at activation + periodic boundaries ---
        if gram_enabled and epoch >= gram_start_epoch:
            if not gram_activated:
                model_without_ddp.refresh_gram_teacher()
                gram_activated = True
                print(f"Epoch {epoch}: Gram anchor activated "
                      f"(weight={args.gram_weight}, "
                      f"downsample_factor={args.gram_downsample_factor}, "
                      f"refresh_every_n_epochs={args.gram_refresh_every_n_epochs}).")
            elif ((epoch - gram_start_epoch) % args.gram_refresh_every_n_epochs) == 0:
                model_without_ddp.refresh_gram_teacher()
                print(f"Epoch {epoch}: Gram teacher refreshed from current EMA teacher.")

        intra_kshot_state = None
        if args.online_kshot and args.online_kshot_intra_epoch_steps > 0:
            patches_per_sample_t = int(args.time_steps) // int(args.patch_width)
            corpus_tokens = int((dataset_train.index[:, 1] * patches_per_sample_t).sum())
            tokens_per_step = corpus_tokens // max(len(data_loader_train), 1)
            intra_kshot_state = {
                "records": [],
                "data_base": args.online_kshot_data_base,
                "tokens_per_step": tokens_per_step,
                "sink_path": (os.path.join(args.output_dir, "intra_kshot.jsonl")
                              if args.output_dir else None),
            }
        train_stats, train_history = train_one_epoch(model, data_loader_train, optimizer, device, epoch, loss_scaler,
                                                     log_writer=log_writer, args=args,
                                                     gram_active=gram_activated,
                                                     intra_kshot_state=intra_kshot_state)

        eval_stats = train_stats
        if args.val_data_path:
            val_stats, val_history = evaluate(data_loader_val, model, device, epoch, 
                                              log_writer=log_writer, args=args)
            eval_stats = val_stats

        # online evaluation of the downstream task
        online_history = {}
        if args.online_kshot and (epoch % 2 == 0 or epoch == args.epochs - 1):
            online_history = evaluate_online_kshot(model=model_without_ddp, device=device,
                                                   data_base=args.online_kshot_data_base,
                                                   seed=args.seed)
        elif args.online_evaluation and (epoch % 10 == 0 or epoch == args.epochs - 1):
            if args.online_evaluation_task == "classification":
                online_estimator = LogisticRegression(class_weight='balanced', max_iter=2000)
            elif args.online_evaluation_task == "regression":
                online_estimator = LinearRegression()

            online_history = evaluate_online(estimator=online_estimator, model=model_without_ddp, device=device,
                                             train_dataloader=data_loader_online_train,
                                             val_dataloader=data_loader_online_val, args=args)

        # online forecasting — independent of the discriminative evaluation above
        if args.online_forecast and (epoch % 2 == 0 or epoch == args.epochs - 1):
            online_history.update(
                evaluate_online_forecast(model=model_without_ddp, device=device,
                                         data_path=args.online_forecast_data_path,
                                         args=args,
                                         mask_ratio=args.online_forecast_mask_ratio,
                                         time_steps=args.online_forecast_time_steps)
            )

        # Expose the online k-shot CLS balanced accuracy as a selectable eval metric.
        # Absent on epochs where k-shot didn't run; the block below skips selection in that case.
        if 'online/kshot_cls_avg_acc_balanced' in online_history:
            eval_stats['kshot_cls_acc'] = online_history['online/kshot_cls_avg_acc_balanced']

        save_kwargs = dict(args=args, model=model, model_without_ddp=model_without_ddp,
                           optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch,
                           nb_ckpts_max=best_eval_scores['nb_ckpts_max'])

        if args.eval_criterion == "epoch":
            best_stats['epoch'] = epoch
            if args.output_dir:
                _update_best_eval_scores(best_eval_scores, epoch, reverse=True)
                misc.save_model(**save_kwargs)
        elif args.eval_criterion not in eval_stats:
            # Criterion not computed this epoch (e.g. kshot_cls_acc runs every
            # other epoch). Skip checkpointing / early-stop for this step.
            print(f"[eval] skipping checkpoint: '{args.eval_criterion}' not available this epoch")
        else:
            decreasing = args.eval_criterion in _DECREASING_METRICS
            mode = "decreasing" if decreasing else "increasing"
            score = eval_stats[args.eval_criterion]

            early_stop_fn = early_stop.evaluate_decreasing_metric if decreasing else early_stop.evaluate_increasing_metric
            if misc.broadcast_flag(early_stop_fn(val_metric=score)):
                print("Early stopping the training")
                break

            if args.output_dir and _update_best_eval_scores(best_eval_scores, score, reverse=not decreasing):
                misc.save_best_model(**save_kwargs, test_stats=eval_stats,
                                     evaluation_criterion=args.eval_criterion, mode=mode)

        # always persist the final-epoch checkpoint, independent of eval_criterion
        if args.output_dir and epoch == args.epochs - 1 and args.eval_criterion != "epoch":
            misc.save_model(**save_kwargs)

        # Persist a pre-annealing checkpoint once, at the end of the last epoch before the annealing phase.
        if args.output_dir and epoch == decay_start_epoch - 1 and args.pre_annealing_checkpoint:
            misc.save_model(**save_kwargs)

        # Persist a pre-Gram checkpoint once, at the end of the last epoch before the Gram anchor activates.
        # Refresh the Gram teacher first so the checkpoint embeds the EMA-teacher snapshot the Gram loss will
        # anchor against — equivalent to the refresh that fires at the start of gram_start_epoch.
        if args.output_dir and gram_enabled and epoch == gram_start_epoch - 1 and args.pre_gram_checkpoint:
            model_without_ddp.refresh_gram_teacher()
            misc.save_model(**save_kwargs)

        _update_best_stats(best_stats, eval_stats)

        print(f"Total Loss / Loss / Normalised Cross-Correlation (NCC) / CLS Loss / MIM Loss / KoLeo Loss / Mean Squared Error (MSE) / ",
              f"Mean Absolute Error (MAE) of the network on {len(dataset_val) if args.val_data_path else train_dataset_len} val samples: {eval_stats['total_loss']:.4f} / ",
              f"{eval_stats['loss']:.4f} / {eval_stats['ncc']:.2f} / {eval_stats['cls_loss']:.4f} / {eval_stats['mim_loss']:.4f} / {eval_stats['koleo']:.4f} / {eval_stats['mse']:.2f} / ",
              f"{eval_stats['mae']:.2f}")

        print(f"Min Total Loss / Min Loss / Max NCC / Min CLS Loss / Min MIM Loss / Min KoLeo Loss / Min MSE / Min MAE: ",
              f"{best_stats['total_loss']:.4f} / {best_stats['loss']:.4f} / {best_stats['ncc']:.2f} / ",
              f"{best_stats['cls_loss']:.4f} / {best_stats['mim_loss']:.4f} / {best_stats['koleo']:.4f} / {best_stats['mse']:.2f} / {best_stats['mae']:.2f}\n")
        
        total_time = time.time() - start_time
        online_log_keys = ('online/kshot_avg_acc_balanced',
                           'online/kshot_cls_avg_acc_balanced',
                           'online/forecast_mse',
                           'online/forecast_mae',
                           'online/forecast_ncc',
                           'online/forecast_ncc_full')
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **({f'val_{k}': v for k, v in val_stats.items()} if args.val_data_path else {}),
                     **{k: online_history[k] for k in online_log_keys if k in online_history},
                     'n_parameters': n_parameters,
                     'epoch': epoch,
                     'time_per_epoch' : total_time}
        
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # Token-L2-norm diagnostic on a fixed subset of pre-training samples.
        # ``token_norm_loader`` is only built on the main process, so this
        # block is implicitly rank-0 only.
        token_norm_history = {}
        if (token_norm_loader is not None
                and (epoch % args.track_token_norms_freq == 0 or epoch == args.epochs - 1)):
            enc_norms, dec_norms, enc_dim, dec_dim = track_token_l2_norms(
                model_without_ddp, token_norm_loader, device,
                num_samples=norm_track_n)
            if args.wandb:
                fig_img = plot_token_l2_norms(
                    enc_norms, dec_norms,
                    encoder_embed_dim=enc_dim, decoder_embed_dim=dec_dim)
                if fig_img is not None:
                    token_norm_history["Token L2 Norms"] = fig_img

            # Record the last-block, pre-final-norm distribution (index -2;
            # index -1 is post-LayerNorm, which flattens the outlier tail).
            if enc_norms is not None and len(enc_norms) >= 2:
                last_layer_norm_history_enc.append((epoch, enc_norms[-2].clone()))
                last_layer_norm_enc_dim = enc_dim
            if dec_norms is not None and len(dec_norms) >= 2:
                last_layer_norm_history_dec.append((epoch, dec_norms[-2].clone()))
                last_layer_norm_dec_dim = dec_dim

            # Stash the full per-layer norms so the "Token L2 Norms" figure
            # can be regenerated offline. Refreshed every diagnostic step
            # (not just at the final epoch) so an early-stop or crash still
            # leaves the latest snapshot on disk.
            final_token_norms_enc = (
                [t.clone() for t in enc_norms] if enc_norms is not None else None)
            final_token_norms_dec = (
                [t.clone() for t in dec_norms] if dec_norms is not None else None)
            final_token_norms_epoch = epoch

            # Persist the per-epoch history and latest snapshot to disk now,
            # so the .pt files survive an early-stop or crash mid-run.
            if args.output_dir and (last_layer_norm_history_enc or last_layer_norm_history_dec):
                torch.save({
                    "encoder": last_layer_norm_history_enc,
                    "decoder": last_layer_norm_history_dec,
                    "encoder_embed_dim": last_layer_norm_enc_dim,
                    "decoder_embed_dim": last_layer_norm_dec_dim,
                }, os.path.join(args.output_dir, "last_layer_norms.pt"))

                if final_token_norms_enc is not None or final_token_norms_dec is not None:
                    torch.save({
                        "encoder": final_token_norms_enc,
                        "decoder": final_token_norms_dec,
                        "encoder_embed_dim": last_layer_norm_enc_dim,
                        "decoder_embed_dim": last_layer_norm_dec_dim,
                        "epoch": final_token_norms_epoch,
                    }, os.path.join(args.output_dir, "token_norms_final.pt"))

        if args.wandb and misc.is_main_process():
            log_data = {**train_history,
                        **(val_history if args.val_data_path else {}),
                        **online_history,
                        **token_norm_history,
                        "Time per epoch [sec]": total_time}
            wandb.log(log_data)

    # End-of-training: render the last-block patch-token L2 norm evolution
    # (à la Darcet et al. 2024, Fig 4(b)). The raw history .pt files are
    # refreshed inside the per-epoch diagnostic block above.
    if (last_layer_norm_history_enc or last_layer_norm_history_dec) and misc.is_main_process():
        if args.wandb:
            evo_img = plot_last_layer_norm_evolution(
                encoder_history=last_layer_norm_history_enc or None,
                decoder_history=last_layer_norm_history_dec or None,
                encoder_embed_dim=last_layer_norm_enc_dim,
                decoder_embed_dim=last_layer_norm_dec_dim)
            if evo_img is not None:
                wandb.log({"Last-layer token L2 norm evolution": evo_img})

    if args.wandb and misc.is_main_process():
        wandb.log({f'Best Statistics/{k}': v for k, v in best_stats.items()})
        wandb.finish()
        exit(0)


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)