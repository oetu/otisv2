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

import json
from typing import Tuple
import numpy as np
import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

# from torch.utils.tensorboard import SummaryWriter
import wandb
# os.environ["WANDB__SERVICE_WAIT"] = "500"

# assert timm.__version__ == "0.3.2" # version check
from timm.layers.weight_init import trunc_normal_
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy #, LabelSmoothingCrossEntropy

from util.dataset import TimeSeriesDataset
import util.lr_decay as lrd
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.callbacks import EarlyStop

import models_encoder

from engine_finetune import train_one_epoch, evaluate, extract_embeddings
from util.token_l2_norms import track_token_l2_norms_encoder, plot_token_l2_norms


def get_args_parser():
    parser = argparse.ArgumentParser('OTISv2 fine-tuning', add_help=False)
    # Basic parameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--model', default='otis_base', type=str, metavar='MODEL',
                        help='Name of model to train (default: otis_base)')
    parser.add_argument('--compile', action='store_true', default=False,
                        help='Use torch compile')
    
    # RoPE parameter
    parser.add_argument('--rope_theta', default=10000.0, type=float,
                        help='Theta value for Rotary Positional Embeddings')
    parser.add_argument('--qk_norm', action='store_true', default=False,
                        help='Enable QK-norm in AxialRoPEAttention (default: QK-norm disabled). '
                             'Must match the pre-trained checkpoint — QK-norm is parameter-free, '
                             'so a mismatch loads cleanly but silently changes the softmax scale.')
    parser.add_argument('--attention_sink', action='store_true', default=False,
                        help='Enable per-head learnable attention sink logit '
                             '(StreamingLLM / DeepSeek-V4 trick). Default: disabled.')

    # Register tokens
    parser.add_argument('--num_register_tokens', default=2, type=int,
                        help='Number of register tokens (default: 2)')

    parser.add_argument('--univariate', action='store_true', default=False,
                        help='Univariate time series analysis (i.e. treat each variate independently)')
    
    parser.add_argument('--input_channels', type=int, default=1, metavar='N',
                        help='input channels')
    parser.add_argument('--input_variates', type=int, default=12, metavar='N',
                        help='input variates')
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
                        help='Drop path rate (default: 0.1)')
                        
    # Augmentation parameters
    parser.add_argument('--masking_blockwise', action='store_true', default=False,
                        help='Masking blockwise in channel and time dimension (instead of random masking)')
    parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--mask_c_ratio', default=0.0, type=float,
                        help='Masking ratio in channel dimension (percentage of removed patches)')
    parser.add_argument('--mask_t_ratio', default=0.0, type=float,
                        help='Masking ratio in time dimension (percentage of removed patches)')

    parser.add_argument('--crop_lower_bnd', default=0.75, type=float,
                        help='Lower boundary of the cropping ratio (default: 0.75)')
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

    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Colour jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)')

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 4')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

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

    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--subepoch_lr_schedule', action=argparse.BooleanOptionalAction, default=True,
                        help='Fractional warm-up / decay lengths (epochs * fraction), so runs with fewer than '
                             '10 epochs still warm up and decay. --no-subepoch_lr_schedule: whole epochs (legacy).')
    parser.add_argument('--warmup_fraction', type=float, default=0.1,
                        help='Fraction of total epochs for the WSD warmup phase (default: 0.1)')
    parser.add_argument('--decay_fraction', type=float, default=0.1,
                        help='Fraction of total epochs for the WSD decay/annealing phase (default: 0.1)')

    # Callback parameters
    parser.add_argument('--patience', default=-1, type=float,
                        help='Early stopping whether val is worse than train for specified nb of epochs (default: -1, i.e. no early stopping)')
    parser.add_argument('--max_delta', default=0, type=float,
                        help='Early stopping threshold (val has to be worse than (train+delta)) (default: 0)')

    # Criterion parameters
    parser.add_argument('--weighted_loss', action='store_true', default=False,
                        help='Apply inverse frequency weighted loss (default: False)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    
    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true', default=False)
    parser.add_argument('--attention_pool', action='store_true', default=False)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters
    parser.add_argument('--downstream_task', default='classification', type=str,
                        help='downstream task (default: classification)')
    eval_criterions = ["epoch", "loss", "acc", "acc_balanced", "precision", "recall", "f1", "auroc", "auprc", "cohen", 
                       "avg", "rmse", "mae", "pcc", "r2"]
    parser.add_argument('--eval_criterion', default='auroc', type=str, choices=eval_criterions,
                        help='downstream task evaluation metric (default: auroc)')
    
    parser.add_argument('--data_path', default='', type=str,
                        help='dataset path (default: None)')
    parser.add_argument('--labels_path', default='', type=str,
                        help='labels path (default: None)')
    parser.add_argument('--labels_mask_path', default='', type=str,
                        help='labels path (default: None)')

    parser.add_argument('--val_data_path', default='', type=str,
                        help='validation dataset path (default: None)')
    parser.add_argument('--val_labels_path', default='', type=str,
                        help='validation labels path (default: None)')
    parser.add_argument('--val_labels_mask_path', default='', type=str,
                        help='validation labels path (default: None)')

    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--test_data_path', default='', type=str,
                        help='test dataset path (default: None)')
    parser.add_argument('--test_labels_path', default='', type=str,
                        help='test labels path (default: None)')
    parser.add_argument('--test_labels_mask_path', default='', type=str,
                        help='test labels path (default: None)')

    parser.add_argument('--lower_bnd', type=int, default=0, metavar='N',
                        help='lower_bnd')
    parser.add_argument('--upper_bnd', type=int, default=0, metavar='N',
                        help='upper_bnd')

    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--pos_label', default=0, type=int,
                        help='classification type with the smallest count')

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
    
    parser.add_argument('--plot_attention_map', action='store_true', default=False)
    parser.add_argument('--plot_embeddings', action='store_true', default=False)
    parser.add_argument('--save_embeddings', action='store_true', default=False,
                        help='save model embeddings')
    parser.add_argument('--save_logits', action='store_true', default=False,
                        help='save model logits')

    # Token L2-norm tracking (à la "Vision Transformers Need Registers" Fig. 4a)
    parser.add_argument('--track_token_norms', action='store_true', default=False,
                        help='Track per-layer L2 norm of patch tokens (encoder only) '
                             'on a fixed subset of training samples and log to wandb')
    parser.add_argument('--track_token_norms_samples', type=int, default=4096,
                        help='Number of random samples used for the token-norm diagnostic '
                             '(default: 4096)')
    parser.add_argument('--track_token_norms_freq', type=int, default=1,
                        help='How often (in epochs) to run the token-norm diagnostic '
                             '(default: 1, every epoch)')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--extract_embeddings', action='store_true', default=False,
                        help='Perform embeddings extraction only')
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
    # checkpoints already in output_dir belong to an earlier run: never prune or test them
    args.preexisting_ckpts = misc.list_checkpoints(args.output_dir)
    args.input_size = (args.input_channels, args.input_variates, args.time_steps)
    args.patch_size = (args.patch_height, args.patch_width)

    print(f"cuda devices: {torch.cuda.device_count()}")
    misc.init_distributed_mode(args)
    # args.distributed = False

    # wandb logging
    if args.wandb == True and misc.is_main_process():
        config = vars(args)
        if args.wandb_id:
            wandb.init(project=args.wandb_project, id=args.wandb_id, config=config, entity=args.wandb_entity)
        else:
            wandb.init(project=args.wandb_project, config=config, entity=args.wandb_entity)

        args.__dict__ = wandb.config.as_dict()

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    print(f"rank: {misc.get_rank()}")
    torch.manual_seed(seed)
    np.random.seed(seed)

    cudnn.benchmark = True
    
    dataset_train = TimeSeriesDataset(data_path=args.data_path, 
                                      labels_path=args.labels_path, 
                                      labels_mask_path=args.labels_mask_path, 
                                      downstream_task=args.downstream_task, 
                                      weighted_loss=args.weighted_loss,
                                      univariate=args.univariate,
                                      train=True, 
                                      N_val=1,
                                      args=args)
    dataset_val = TimeSeriesDataset(data_path=args.val_data_path,
                                    labels_path=args.val_labels_path,
                                    labels_mask_path=args.val_labels_mask_path,
                                    downstream_task=args.downstream_task,
                                    univariate=args.univariate,
                                    train=False,
                                    N_val=1,
                                    args=args)
    if args.test:
        dataset_test = TimeSeriesDataset(data_path=args.test_data_path,
                                        labels_path=args.test_labels_path,
                                        labels_mask_path=args.test_labels_mask_path,
                                        downstream_task=args.downstream_task,
                                        univariate=args.univariate,
                                        train=False,
                                        test=True,
                                        N_val=1,
                                        args=args)

    print("Training set size: ", len(dataset_train))
    print("Validation set size: ", len(dataset_val))
    if args.test:
        print("Test set size: ", len(dataset_test))

    if True:  # args.distributed:
        num_tasks = misc.get_world_size()
        print(f"num_tasks: {num_tasks}")
        global_rank = misc.get_rank()
        print(f"global_rank: {global_rank}")
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
        # print("Sampler_train = %s" % str(sampler_train))

        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)  # shuffle=True to reduce monitor bias
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        # print("Sampler_val = %s" % str(sampler_val))

        if args.test:
            if args.dist_eval:
                if len(dataset_test) % num_tasks != 0:
                    print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                        'This will slightly alter validation results as extra duplicate entries are added to achieve '
                        'equal num of samples per-process.')
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)  # the test pass runs on the main process only
            else:
                sampler_test = torch.utils.data.SequentialSampler(dataset_test)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
        if args.test:
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    # tensorboard logging
    if False: #global_rank == 0 and args.log_dir and not args.eval:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    elif False: #args.log_dir and args.eval and "checkpoint" not in args.resume.split("/")[-1]:
        log_writer = SummaryWriter(log_dir=args.log_dir + "/eval")
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        sampler=sampler_train,
        # shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_train.collate_fn_ft,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        sampler=sampler_val,
        # shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=dataset_val.collate_fn_ft,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    if args.test:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, 
            sampler=sampler_test,
            # shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=dataset_test.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False
        )

    # Token-L2-norm tracking: build a fixed subset dataloader on the main
    # process so the per-layer norm diagnostic is computed on the same set
    # of samples each epoch (and only once per training run, not per rank).
    token_norm_loader = None
    norm_track_n = 0
    if args.track_token_norms and misc.is_main_process():
        norm_track_n = min(args.track_token_norms_samples, len(dataset_train))
        norm_track_rng = np.random.default_rng(seed=42)
        norm_track_indices = norm_track_rng.choice(
            len(dataset_train), size=norm_track_n, replace=False).tolist()
        # clean views (train=False: first time_steps window, no augmentation), so
        # the epoch-to-epoch norm curves show training drift only; the shallow
        # copy shares the data tensors
        import copy as _copy
        dataset_train_track = _copy.copy(dataset_train)
        dataset_train_track.train = False
        token_norm_subset = torch.utils.data.Subset(dataset_train_track, norm_track_indices)
        token_norm_loader = torch.utils.data.DataLoader(
            token_norm_subset,
            batch_size=args.batch_size,
            num_workers=min(4, args.num_workers),
            collate_fn=dataset_train.collate_fn_ft,
            pin_memory=args.pin_mem,
            drop_last=False,
            shuffle=False,
            persistent_workers=True,
        )
        print(f"Token-L2-norm tracking enabled: {norm_track_n} fixed samples, "
              f"every {args.track_token_norms_freq} epoch(s)")

    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        print("Mixup is activated!")
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)
    
    # Take the encoder architecture flags (attention_sink, qk_norm, rope_theta,
    # num_register_tokens) from the pre-trained checkpoint; the parser defaults
    # do not match the release checkpoints.
    pretrained_ckpt = None
    if args.finetune and not args.eval:
        pretrained_ckpt = torch.load(args.finetune, map_location='cpu', weights_only=False)
        misc.sync_arch_args(args, pretrained_ckpt, get_args_parser())

    model = models_encoder.__dict__[args.model](
        input_size=args.input_size,
        patch_size=args.patch_size,
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        attention_pool=args.attention_pool,
        masking_blockwise=args.masking_blockwise,
        mask_ratio=args.mask_ratio,
        mask_c_ratio=args.mask_c_ratio,
        mask_t_ratio=args.mask_t_ratio,
        rope_theta=args.rope_theta,
        qk_norm=args.qk_norm,
        attention_sink=args.attention_sink,
        num_register_tokens=args.num_register_tokens
    )

    new_patch_size = False
    if args.finetune and not args.eval:
        checkpoint = pretrained_ckpt

        print("Load pretrained checkpoint from: %s" % args.finetune)
        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # check if new and old patch_size match
        nb_channels_ckpt = checkpoint_model['patch_embed.proj.weight'].shape[-3]
        nb_channels_model = args.input_size[0]

        checkpoint_patch_size = checkpoint_model['patch_embed.proj.weight'].shape[-2:]
        patch_height_ckpt, patch_width_ckpt = checkpoint_patch_size[0], checkpoint_patch_size[1]
        patch_height_model, patch_width_model = args.patch_size[0], args.patch_size[1]

        if nb_channels_ckpt != nb_channels_model or patch_height_ckpt != patch_height_model or patch_width_ckpt != patch_width_model:
            new_patch_size = True
            # initialise new patch_embed
            for key in ["patch_embed.proj.weight", "patch_embed.proj.bias", 
                        "patch_embed.norm.weight", "patch_embed.norm.bias"]:
                if key in checkpoint_model:
                    print(f"Removing key {key} from pretrained checkpoint")
                    del checkpoint_model[key]
            print("Initialising new patch_embed")

        # Drop any legacy domain_token keys (no longer part of the architecture)
        checkpoint_model.pop('domain_token', None)

        # handle register_tokens shape mismatch
        if 'register_tokens' in checkpoint_model:
            ckpt_shape = checkpoint_model['register_tokens'].shape
            model_shape = model.register_tokens.shape
            if ckpt_shape != model_shape:
                raise RuntimeError(f"register_tokens shape {list(ckpt_shape)} in the checkpoint != "
                                   f"{list(model_shape)} in the model; pass a matching --num_register_tokens")

        # load pretrained model
        msg = model.load_state_dict(checkpoint_model, strict=False)
        print(msg)
        misc.check_encoder_load(msg, model)

        if args.global_pool:
            assert {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}.issubset(set(msg.missing_keys))
        elif args.attention_pool:
            assert {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}.issubset(set(msg.missing_keys))
        else:
            assert {'head.weight', 'head.bias'}.issubset(set(msg.missing_keys))

        # manually initialise fc layer
        trunc_normal_(model.head.weight, std=0.01)#2e-5)

    if args.eval:
        sub_strings = args.resume.split("/")
        # --resume is one checkpoint file, or the index N of a series checkpoint-0.pth ... checkpoint-N.pth
        single_ckpt = os.path.isfile(args.resume) or "checkpoint" in sub_strings[-1]
        if single_ckpt:
            nb_ckpts = 1
        else:
            nb_ckpts = int(sub_strings[-1])+1

        if not single_ckpt:
            args.resume = "/".join(sub_strings[:-1]) + "/checkpoint-" + str(0) + ".pth"
    
    if args.compile:
        # model = torch.compile(model, backend="inductor", mode="reduce-overhead")
        model.forward = torch.compile(model.forward)
    model.to(device, non_blocking=True)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_encoder_prefixes = ("head", "fc_norm", "projector", "predictor", "decoder", "attention_pool")
    n_params_encoder = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and not any(n.startswith(k) for k in non_encoder_prefixes))
    n_params_head = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and (n.startswith("head") or n.startswith("fc_norm")))

    print("Model = %s" % str(model_without_ddp))
    print("Number of params (M): %.2f" % (n_parameters / 1.e6))
    print("Number of encoder params (M): %.2f" % (n_params_encoder / 1.e6))
    print("Number of head params (M): %.2f" % (n_params_head / 1.e6))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 32

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

    # build optimizer with layer-wise lr decay (lrd)
    param_groups = lrd.param_groups_lrd(model_without_ddp, args.weight_decay,
        no_weight_decay_list=model_without_ddp.no_weight_decay(), layer_decay=args.layer_decay,
        use_muon=args.use_muon)
    optimizer = misc.build_optimizer(param_groups, args)
    print(optimizer)
    loss_scaler = NativeScaler()

    class_weights = None
    print(dataset_train.class_weights)
    if dataset_train.class_weights is not None:
        class_weights = dataset_train.class_weights.to(device=device, non_blocking=True)
    
    if args.downstream_task == 'regression':
        criterion = torch.nn.MSELoss()
    elif mixup_fn is not None:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.:
        # criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.smoothing)
        # criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        # criterion = torch.nn.CrossEntropyLoss()

    print("criterion = %s" % str(criterion))

    if not args.eval:
        misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if args.extract_embeddings:
        extract_embeddings(data_loader_val, model_without_ddp, device, args=args)
        exit(0)

    if args.eval:
        sub_strings = args.resume.split("/")
        # --resume is one checkpoint file, or the index N of a series checkpoint-0.pth ... checkpoint-N.pth
        single_ckpt = os.path.isfile(args.resume) or "checkpoint" in sub_strings[-1]
        if single_ckpt:
            nb_ckpts = 1
        else:
            nb_ckpts = int(sub_strings[-1])+1

        for epoch in range(0, nb_ckpts):
            if not single_ckpt:
                args.resume = "/".join(sub_strings[:-1]) + "/checkpoint-" + str(epoch) + ".pth"

            misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

            test_stats, test_history = evaluate(data_loader_val, model_without_ddp, device, epoch, log_writer=log_writer, args=args)
            if args.downstream_task == 'classification':
                print(f"Accuracy / Accuracy (balanced) / Precision / Recall / F1 / AUROC / AUPRC / Cohen's Kappa of the network on {len(dataset_val)} test samples: ",
                      f"{test_stats['acc']:.2f}% / {test_stats['acc_balanced']:.2f}% / {test_stats['precision']:.2f}% / {test_stats['recall']:.2f}% / ",
                      f"{test_stats['f1']:.2f}% / {test_stats['auroc']:.2f}% / {test_stats['auprc']:.2f}% / {test_stats['cohen']:.4f}")
            elif args.downstream_task == 'regression':
                print(f"Root Mean Squared Error (RMSE) / Mean Absolute Error (MAE) / Pearson Correlation Coefficient (PCC) / R Squared (R2) ",
                      f"of the network on {len(dataset_val)} test samples: {test_stats['rmse']:.4f} / {test_stats['mae']:.4f} / ",
                      f"{test_stats['pcc']:.4f} / {test_stats['r2']:.4f}")
        
            if args.wandb and misc.is_main_process():
                wandb.log(test_history)

        exit(0)

    # Define callbacks
    early_stop = EarlyStop(patience=args.patience, max_delta=args.max_delta)
    
    print(f"Start training for {args.epochs} epochs")
    
    best_stats = {'loss':np.inf, 'acc':0.0, 'acc_balanced':0.0, 'precision':0.0, 'recall':0.0, 'f1':0.0, 'auroc':0.0, 'auprc':0.0, 'cohen':-np.inf,
                  'avg':-np.inf, 'epoch':0, 'rmse':np.inf, 'mae':np.inf, 'pcc':-np.inf, 'r2':-np.inf}  # signed metrics start at -inf
    best_eval_scores = {'count':1, 'nb_ckpts_max':1, 'eval_criterion':[best_stats[args.eval_criterion]]}
    for epoch in range(args.start_epoch, args.epochs):
        start_time = time.time()

        if True: #args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        
        train_stats, train_history = train_one_epoch(model, criterion, data_loader_train, optimizer, device, epoch, 
                                                     loss_scaler, args.clip_grad, mixup_fn, log_writer=log_writer, args=args)

        test_stats, test_history = evaluate(data_loader_val, model_without_ddp, device, epoch, log_writer=log_writer, args=args)

        if args.eval_criterion == "epoch":
            if args.downstream_task == 'classification':
                test_stats['avg'] = (test_stats['auroc'] + test_stats['acc_balanced'] + test_stats['f1'] + test_stats['cohen']) / 4
            elif args.downstream_task == 'regression':
                test_stats['avg'] = (test_stats['pcc'] + test_stats['r2']) / 2
            if args.output_dir:
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(epoch)

                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, nb_ckpts_max=best_eval_scores['nb_ckpts_max'])
        elif args.eval_criterion in ["loss", "rmse", "mae"]:
            if args.eval_criterion != "loss":
                test_stats['avg'] = (test_stats['rmse'] + test_stats['mae']) / 2
            else: 
                test_stats['avg'] = test_stats['loss']

            if misc.broadcast_flag(early_stop.evaluate_decreasing_metric(val_metric=test_stats[args.eval_criterion])):
                print("Early stopping the training")
                break

            if args.output_dir and test_stats[args.eval_criterion] <= max(best_eval_scores['eval_criterion']):
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'])
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(test_stats[args.eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats,
                    evaluation_criterion=args.eval_criterion, nb_ckpts_max=best_eval_scores['nb_ckpts_max'],
                    mode="decreasing")
        else:
            if args.downstream_task == 'classification':
                test_stats['avg'] = (test_stats['auroc'] + test_stats['acc_balanced'] + test_stats['f1'] + test_stats['cohen']) / 4
            elif args.downstream_task == 'regression':
                test_stats['avg'] = (test_stats['pcc'] + test_stats['r2']) / 2
            
            if misc.broadcast_flag(early_stop.evaluate_increasing_metric(val_metric=test_stats[args.eval_criterion])):
                print("Early stopping the training")
                break

            if args.output_dir and test_stats[args.eval_criterion] >= min(best_eval_scores['eval_criterion']):
                # save the best nb_ckpts_max checkpoints
                if best_eval_scores['count'] < best_eval_scores['nb_ckpts_max']:
                    best_eval_scores['count'] += 1
                else:
                    best_eval_scores['eval_criterion'] = sorted(best_eval_scores['eval_criterion'], reverse=True)
                    best_eval_scores['eval_criterion'].pop()
                best_eval_scores['eval_criterion'].append(test_stats[args.eval_criterion])

                misc.save_best_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch, test_stats=test_stats,
                    evaluation_criterion=args.eval_criterion, nb_ckpts_max=best_eval_scores['nb_ckpts_max'],
                    mode="increasing")

        test_history['test_avg'] = test_stats['avg']

        best_stats['loss'] = min(best_stats['loss'], test_stats['loss'])
        
        if args.downstream_task == 'classification':
            # update best stats
            best_stats['f1'] = max(best_stats['f1'], test_stats['f1'])
            best_stats['precision'] = max(best_stats['precision'], test_stats['precision'])
            best_stats['recall'] = max(best_stats['recall'], test_stats['recall'])
            best_stats['acc'] = max(best_stats['acc'], test_stats["acc"])
            best_stats['acc_balanced'] = max(best_stats['acc_balanced'], test_stats["acc_balanced"])
            best_stats['auroc'] = max(best_stats['auroc'], test_stats['auroc'])
            best_stats['auprc'] = max(best_stats['auprc'], test_stats['auprc'])
            best_stats['cohen'] = max(best_stats['cohen'], test_stats['cohen'])

            if test_stats['avg'] >= best_stats['avg']:
                best_stats['epoch'] = epoch
            best_stats['avg'] = max(best_stats['avg'], test_stats['avg'])

            print(f"Accuracy / Accuracy (balanced) / Precision / Recall / F1 / AUROC / AUPRC / Cohen's Kappa of the network on {len(dataset_val)} test samples: ",
                  f"{test_stats['acc']:.2f}% / {test_stats['acc_balanced']:.2f}% / {test_stats['precision']:.2f}% / {test_stats['recall']:.2f}% / ",
                  f"{test_stats['f1']:.2f}% / {test_stats['auroc']:.2f}% / {test_stats['auprc']:.2f}% / {test_stats['cohen']:.4f}")
            print(f'Max Accuracy / Accuracy (balanced) / Precision / Recall / F1 / AUROC / AUPRC: {best_stats["acc"]:.2f}% / ',
                  f'{best_stats["acc_balanced"]:.2f}% / {best_stats["precision"]:.2f}% / {best_stats["recall"]:.2f}% / {best_stats["f1"]:.2f}% / ',
                  f'{best_stats["auroc"]:.2f}% / {best_stats["auprc"]:.2f}% / {best_stats["cohen"]:.4f}\n')

        elif args.downstream_task == 'regression':
            # update best stats
            best_stats['rmse'] = min(best_stats['rmse'], test_stats['rmse'])
            best_stats['mae'] = min(best_stats['mae'], test_stats['mae'])
            best_stats['pcc'] = max(best_stats['pcc'], test_stats['pcc'])
            best_stats['r2'] = max(best_stats['r2'], test_stats['r2'])

            if args.eval_criterion in ["loss", "rmse", "mae"]:
                if test_stats['avg'] <= best_stats['avg']:
                    best_stats['epoch'] = epoch
                best_stats['avg'] = min(best_stats['avg'], test_stats['avg'])
            else:
                if test_stats['avg'] >= best_stats['avg']:
                    best_stats['epoch'] = epoch
                best_stats['avg'] = max(best_stats['avg'], test_stats['avg'])

            print(f"Root Mean Squared Error (RMSE) / Mean Absolute Error (MAE) / Pearson Correlation Coefficient (PCC) / R Squared (R2) ",
                  f"of the network on {len(dataset_val)} test samples: {test_stats['rmse']:.4f} / {test_stats['mae']:.4f} / ",
                  f"{test_stats['pcc']:.4f} / {test_stats['r2']:.4f}")
            print(f'Min Root Mean Squared Error (RMSE) / Min Mean Absolute Error (MAE) / Max Pearson Correlation Coefficient (PCC) / ',
                  f'Max R Squared (R2): {best_stats["rmse"]:.4f} / {best_stats["mae"]:.4f} / {best_stats["pcc"]:.4f} / {best_stats["r2"]:.4f}\n')
        
        total_time = time.time() - start_time
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 
                     **{f'test_{k}': v for k, v in test_stats.items()}, 
                     'n_parameters': n_parameters,
                     'epoch': epoch, 
                     'time_per_epoch' : total_time}
        
        if args.output_dir and misc.is_main_process():
            if log_writer:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # Token-L2-norm diagnostic on a fixed subset of training samples.
        # ``token_norm_loader`` is only built on the main process, so this
        # block is implicitly rank-0 only.
        token_norm_history = {}
        if (token_norm_loader is not None
                and (epoch % args.track_token_norms_freq == 0 or epoch == args.epochs - 1)):
            enc_norms, enc_dim = track_token_l2_norms_encoder(
                model_without_ddp, token_norm_loader, device,
                num_samples=norm_track_n)
            if args.wandb:
                fig_img = plot_token_l2_norms(
                    encoder_norms=enc_norms, encoder_embed_dim=enc_dim)
                if fig_img is not None:
                    token_norm_history["Token L2 Norms"] = fig_img

        if args.wandb and misc.is_main_process():
            wandb.log(train_history | test_history | token_norm_history
                      | {"Time per epoch [sec]": total_time})


    if args.test and misc.is_main_process():
        args.resume = misc.get_best_ckpt(args.output_dir, eval_criterion=args.eval_criterion,
                                          exclude=args.preexisting_ckpts)
        
        checkpoint = torch.load(args.resume, weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Run test data on checkpoint model %s" % args.resume)
        
        test_stats, test_history = evaluate(data_loader_test, model_without_ddp, device, epoch=-1, log_writer=log_writer, args=args)
        
        actual_test_history = {}
        for k,v in test_history.items():
            key = k
            if 'est' in k:
                key = 'actual_' + key
                actual_test_history[key] = v 

        print(actual_test_history)

        if args.wandb and misc.is_main_process():
            wandb.log(actual_test_history)


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