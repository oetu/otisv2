# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# timm: https://github.com/huggingface/pytorch-image-models/blob/main/timm/optim/_param_groups.py
# --------------------------------------------------------

from typing import Dict

import builtins
import datetime
import math
import os
import time
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.distributed as dist

import numpy as np

import re


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        # args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        # args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        # args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        
        args.rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID')))
        args.world_size = int(os.environ.get('WORLD_SIZE', os.environ.get('SLURM_NTASKS')))
        args.gpu = int(os.environ.get('LOCAL_RANK', os.environ.get('SLURM_LOCALID')))

        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}, gpu {args.gpu}', flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self, amp_dtype: str = 'bf16'):
        # bf16 autocast doesn't need loss scaling; disabled scaler is a no-op
        # that still exposes scale/unscale/step for API compatibility.
        # fp16 autocast DOES need scaling; enable when --amp_dtype fp16.
        self._scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == 'fp16'))

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            if not self._scaler.is_enabled() and not torch.isfinite(norm):
                # bf16: the scaler is disabled and would step on inf/nan grads.
                # DDP has synced the grads, so all ranks skip this step together.
                print(f"Non-finite grad norm ({norm.item()}), skipping optimiser step")
            else:
                self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, nb_ckpts_max:int=1):
    output_dir = Path(args.output_dir)
    epoch_str = str(epoch)

    file_names = os.listdir(output_dir)
    # remove directories from list
    file_names = [file for file in file_names if os.path.isfile(os.path.join(args.output_dir, file))]
    # ignore exceptions from list
    exceptions = ["log.txt"]
    file_names = [file for file in file_names if file not in exceptions and "epoch" in file]

    # save the best nb_ckpts_max performing models
    if len(file_names) >= nb_ckpts_max and is_main_process():
        file_names = sorted(file_names, key=lambda x: float(x.split(".pth")[0].split("-")[-1]), reverse=True)
        os.remove(os.path.join(output_dir, file_names[-1]))

    if loss_scaler is not None:
        checkpoint_paths = [output_dir / (f"checkpoint-epoch-{epoch_str}.pth")]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir,
                              tag=f"checkpoint-epoch-{epoch_str}.pth",
                              client_state=client_state)


def save_best_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, test_stats,
                    evaluation_criterion, nb_ckpts_max:int=1, mode="increasing"):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    file_names = os.listdir(output_dir)
    # remove directories from list
    file_names = [file for file in file_names if os.path.isfile(os.path.join(args.output_dir, file))]
    # ignore exceptions from list
    exceptions = ["log.txt"]
    file_names = [file for file in file_names if file not in exceptions and evaluation_criterion in file]
    # never prune checkpoints of an earlier run in the same directory
    preexisting = set(getattr(args, "preexisting_ckpts", ()) or ())
    file_names = [file for file in file_names if file not in preexisting]

    # save the best nb_ckpts_max performing models
    if len(file_names) >= nb_ckpts_max and is_main_process():
        # file_names = sorted(file_names, key=lambda str: int(re.search(r'\d+', str).group()))
        if mode == "increasing":
            file_names = sorted(file_names, key=_ckpt_score, reverse=True)
        else: # decreasing
            file_names = sorted(file_names, key=_ckpt_score)
        os.remove(os.path.join(output_dir, file_names[-1]))

    if loss_scaler is not None:
        checkpoint_paths = [output_dir / (f"checkpoint-{epoch_name}-{evaluation_criterion}-{test_stats[evaluation_criterion]:.4f}.pth")]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir,
                              tag=f"checkpoint-{epoch_name}-{evaluation_criterion}-{test_stats[evaluation_criterion]:.4f}.pth",
                              client_state=client_state)


def load_model(args, model_without_ddp, optimizer, loss_scaler):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model_without_ddp.load_state_dict(checkpoint['model'])
        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def get_best_ckpt(data_path, eval_criterion, exclude=()):
    print(f"Searching for the checkpoint file with the best evaluation score in: {data_path}")

    # Regex pattern to match the filenames and extract the scores.
    # The score may be negative (r2 and pcc are both signed, and save_best_model
    # writes e.g. "checkpoint-16-r2--0.0000.pth"), so the sign is part of the
    # capture group — without it a run whose best epoch scored below zero matches
    # nothing and best_file stays None.
    pattern = re.compile(r'checkpoint-\d+'+f'-{eval_criterion}-'+r'(-?\d+\.\d+)\.pth')

    # Variables to keep track of the best score and corresponding filename

    incr_criterions = ["epoch", "acc", "acc_balanced", "precision", "recall", "f1", "cohen",
                       "auroc", "auprc", "avg", "pcc", "r2"]

    incr_metric = eval_criterion in incr_criterions

    # Seed with +/-inf rather than -1 / 1e9: for signed criteria (r2, pcc) a best
    # score of -1 is attainable, and seeding at -1 would reject it.
    best_score = float('-inf') if incr_metric else float('inf')
    best_file = None

    # Iterate through all files in the directory
    for filename in os.listdir(data_path):
        print(f"Is it this one?: {filename}")
        # checkpoints of an earlier run in the same directory are not candidates
        if filename in exclude:
            continue
        # Check if the filename matches the pattern
        match = pattern.match(filename)
        if match:
            # Extract the score from the filename
            score = float(match.group(1))
            # Update the best score and filename if current score is better
            if incr_metric and score > best_score:
                best_score = score
                best_file = filename
            
            if not incr_metric and score < best_score:
                best_score = score
                best_file = filename

    # Print the filename with the highest score
    print(f"The checkpoint file with the best evaluation score is: {best_file}")

    return os.path.join(data_path, best_file)


# Encoder architecture flags that must match between pre-training and downstream use
ARCH_ARG_KEYS = ("attention_sink", "qk_norm", "rope_theta", "num_register_tokens")


def checkpoint_arch_args(checkpoint) -> dict:
    """Encoder architecture flags stored in a checkpoint.

    Reads ``checkpoint['args']`` (full pre-training or downstream checkpoints)
    and ``checkpoint['release_info']['pretrain']`` (trimmed release checkpoint,
    which keeps attention_sink / qk_norm there). Returns only the keys found.
    """
    found = {}
    src = checkpoint.get("args")
    if src is not None:
        src = src if isinstance(src, dict) else vars(src)
        found.update({k: src[k] for k in ARCH_ARG_KEYS if k in src})
    info = (checkpoint.get("release_info") or {}).get("pretrain", {})
    found.update({k: info[k] for k in ARCH_ARG_KEYS if k in info and k not in found})
    return found


def sync_arch_args(args, checkpoint, parser):
    """Take the encoder architecture flags from the pre-trained checkpoint.

    A flag left at its parser default is set to the checkpoint value. A flag set
    on the command line to a value that contradicts the checkpoint raises an
    error. Flags the checkpoint does not store keep the command-line value.
    """
    stored = checkpoint_arch_args(checkpoint)
    for k, v in stored.items():
        cur, default = getattr(args, k), parser.get_default(k)
        if cur == v:
            continue
        if cur != default:
            raise ValueError(f"--{k} {cur} contradicts the checkpoint ({k}={v}). "
                             f"Drop the flag or use a matching checkpoint.")
        print(f"Architecture flag from checkpoint: {k} = {v} (parser default {cur})")
        setattr(args, k, v)
    missing = [k for k in ARCH_ARG_KEYS if k not in stored]
    if missing:
        print(f"WARNING: checkpoint stores no {missing}; using the command-line values")


def check_encoder_load(msg, model):
    """Fail if a non-strict load dropped or missed transformer-block weights.

    Unexpected ``blocks.*`` / ``decoder_blocks.*`` keys (e.g. ``attn.sink_logit``)
    of a module the model HAS, or missing ``blocks.*`` keys, mean that the
    model's architecture differs from the checkpoint's, which ``strict=False``
    would otherwise hide. Keys of modules the model lacks (e.g. the decoder for
    the encoder-only Transformer) are ignored.
    """
    bad_unexpected = [k for k in msg.unexpected_keys
                      if k.startswith(("blocks.", "decoder_blocks.")) and hasattr(model, k.split(".")[0])]
    bad_missing = [k for k in msg.missing_keys if k.startswith("blocks.")]
    if bad_unexpected or bad_missing:
        raise RuntimeError("Architecture mismatch between model and checkpoint: "
                           f"unexpected {bad_unexpected[:4]}, missing {bad_missing[:4]}")


def broadcast_flag(flag: bool) -> bool:
    """Rank 0's boolean decision on every rank (collective: call it on all ranks).

    Use it for loop exits such as early stopping, so that no rank leaves the
    loop while the others wait in the next collective.
    """
    if not is_dist_avail_and_initialized():
        return bool(flag)
    t = torch.tensor([1.0 if flag else 0.0], device=torch.cuda.current_device())
    dist.broadcast(t, src=0)
    return t.item() > 0


def gather_eval_tensor(t, data_loader):
    """Concatenate an evaluation tensor over ranks when the loader shards the data.

    With ``--dist_eval`` the loader uses a DistributedSampler, so each rank only
    sees one shard; metrics must be computed on all shards. The sampler pads the
    shards to equal size with duplicate samples, which stay in (as its warning
    says). Without a DistributedSampler every rank already has the full set.
    """
    if not (is_dist_avail_and_initialized()
            and isinstance(getattr(data_loader, "sampler", None), torch.utils.data.DistributedSampler)):
        return t
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, t.cpu())
    return torch.cat(parts, dim=0)


def list_checkpoints(output_dir) -> list:
    """Checkpoint files currently in ``output_dir`` (empty if it does not exist).

    Downstream runs record this at start-up as ``args.preexisting_ckpts``, so
    that checkpoints of an earlier run in the same directory are never pruned
    and never picked as this run's best checkpoint.
    """
    if not output_dir or not os.path.isdir(output_dir):
        return []
    return sorted(f for f in os.listdir(output_dir) if f.startswith("checkpoint") and f.endswith(".pth"))


# score at the end of "checkpoint-<epoch>-<criterion>-<score>.pth"; the score can
# be negative ("...-r2--0.0500.pth"), so the sign belongs to the capture group
_CKPT_SCORE = re.compile(r"-(-?\d+(?:\.\d+)?)\.pth$")


def _ckpt_score(file_name: str) -> float:
    return float(_CKPT_SCORE.search(file_name).group(1))


def checkpoint_scores(output_dir, eval_criterion) -> list:
    """Scores of the "checkpoint-<epoch>-<criterion>-<score>.pth" files in ``output_dir``."""
    pattern = re.compile(r"checkpoint-\d+-" + re.escape(eval_criterion) + r"-(-?\d+(?:\.\d+)?)\.pth$")
    return [float(m.group(1)) for f in list_checkpoints(output_dir) if (m := pattern.match(f))]


def all_ranks_finite(value: float) -> bool:
    """True only if ``value`` is finite on EVERY rank.

    Collective: call it on all ranks, so that all ranks take the same branch
    (an exit on one rank alone leaves the others hanging in the next all-reduce).
    """
    flag = torch.tensor([0.0 if math.isfinite(value) else 1.0],
                        device=torch.cuda.current_device() if torch.cuda.is_available() else "cpu")
    if is_dist_avail_and_initialized():
        dist.all_reduce(flag)
    return flag.item() == 0


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
    

def _is_muon_param(name, param):
    """Determine if a parameter should use the MUON optimizer.

    MUON is applied to 2D+ hidden weight matrices (encoder blocks, decoder blocks,
    patch_embed conv, decoder_embed linear, projector/predictor).

    Excluded: learnable tokens, norms, biases, output heads.
    """
    if param.ndim < 2:
        return False
    if name.endswith(".bias"):
        return False
    # Learnable tokens
    # Learnable tokens / queries (any *_token(s), e.g. mask_token_encoder, and
    # the attention-pool query q) are not weight matrices: use AdamW
    leaf = name.split(".")[-1]
    if leaf.endswith("_token") or leaf.endswith("_tokens") or leaf.startswith("mask_token") or name == "q":
        return False
    # DINO / iBOT prototype layers (``*.last_layer.*``): AdamW, as in DINOv2/v3
    if ".last_layer." in name:
        return False
    # Norm layers
    if "norm" in name:
        return False
    # Output heads / prediction layers
    if name.startswith("decoder_pred") or name.startswith("mlp_pred") or name.startswith("head"):
        return False
    return True


def add_weight_decay_timm_lrd(model, weight_decay=1e-5, skip_list=(), layer_decay=.95,
                               lr_scale_overrides=None, patch_embed_lr_mult=1.0,
                               student_head_wd_mult=1.0, use_muon=False):
    """
    with layer-wise lr decay (LRD) for ViTs, following BEiT

    lr_scale_overrides: optional dict mapping parameter names to lr_scale multipliers,
        e.g. {"register_tokens": 10} to boost register_tokens lr by 10x.
    patch_embed_lr_mult: multiplier applied (on top of layer-wise decay at layer_id=0) to
        any parameter whose name starts with ``patch_embed``. Default 1.0 (no change).
        Values < 1.0 slow the patch projection further, following DINOv3 / BEiT — drift
        in the patch embed propagates to every token, so slowing it stabilises long
        pre-training runs.
    student_head_wd_mult: multiplier applied to ``weight_decay`` (not lr_scale) for any
        parameter whose name starts with ``student_head.`` — i.e. the CLS / DINO branch
        of the iBOT projection head. Default 1.0 (no change). Higher values implement
        ViT-2022-style high-WD-on-head pressure for transfer. Has no effect on
        ``student_head.last_layer.weight_v`` (the layer is weight-normalised; WD on
        weight_v is inert).
    use_muon: if True, tag each group with ``use_muon`` based on parameter properties.
    """
    if lr_scale_overrides is None:
        lr_scale_overrides = {}

    num_layers = len(model.blocks) + 1
    layer_scales = list(layer_decay ** (num_layers - i) for i in range(num_layers + 1))

    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        # Determine weight decay
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay

        # Determine layer ID
        layer_id = get_layer_id_for_vit(name, num_layers)
        this_scale = layer_scales[layer_id]

        # Apply patch_embed multiplier on top of layer-wise decay
        is_patch_embed_override = (
            name.startswith('patch_embed') and patch_embed_lr_mult != 1.0
        )
        if is_patch_embed_override:
            this_scale = this_scale * patch_embed_lr_mult

        # Apply student_head WD multiplier (CLS / DINO branch only — does NOT match
        # student_mim_head, which has a different prefix). Applied to weight_decay, not
        # lr_scale. Skip params already in the no_decay group (norms, biases) so the
        # multiplier doesn't accidentally turn them on.
        is_student_head_override = (
            name.startswith('student_head.') and student_head_wd_mult != 1.0
            and this_decay > 0
        )
        if is_student_head_override:
            this_decay = this_decay * student_head_wd_mult

        # Determine MUON eligibility
        muon_flag = use_muon and _is_muon_param(name, param)
        muon_suffix = "_muon" if muon_flag else "_adam"

        # Apply lr_scale override if specified
        if name in lr_scale_overrides:
            this_scale = this_scale * lr_scale_overrides[name]
            group_name = "%s_%s%s" % (name, g_decay, muon_suffix)
        elif is_patch_embed_override:
            group_name = "patch_embed_%s%s" % (g_decay, muon_suffix)
        elif is_student_head_override:
            group_name = "student_head_%s%s" % (g_decay, muon_suffix)
        else:
            group_name = "layer_%d_%s%s" % (layer_id, g_decay, muon_suffix)

        if group_name not in param_groups:
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            if use_muon:
                param_groups[group_name]["use_muon"] = muon_flag

        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())


def build_optimizer(param_groups, args):
    """Build optimizer, optionally with MUON for 2D hidden weight matrices."""
    if getattr(args, "use_muon", False):
        from util.muon import MuonAdamW

        muon_wd = getattr(args, "muon_weight_decay", 0.0)
        for g in param_groups:
            scale = g.get("lr_scale", 1.0)
            if g.get("use_muon", False):
                g["base_lr"] = args.muon_lr
                g["momentum"] = getattr(args, "muon_momentum", 0.95)
                g["lr"] = args.muon_lr * scale
                # Override weight decay for MUON groups (only those that had decay)
                if g["weight_decay"] > 0:
                    g["weight_decay"] = muon_wd
            else:
                g["base_lr"] = args.lr
                g.setdefault("betas", (0.9, 0.95))
                g.setdefault("eps", 1e-8)
                g["lr"] = args.lr * scale
        return MuonAdamW(param_groups)
    else:
        return torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))


def get_layer_id_for_vit(name, num_layers):
    """
    Assign a parameter with its layer id
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'register_tokens']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers