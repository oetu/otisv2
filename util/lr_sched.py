# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# MAE:  https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math

def adjust_learning_rate(optimizer, epoch, args, warmup_fraction=0.1):
    """Decay the learning rate with half-cycle cosine after warmup.

    Supports per-group ``base_lr`` for hybrid optimizers (e.g. MUON + AdamW).
    """
    if args.min_lr == args.blr:
        # constant-LR switch (min_lr set to blr): peak LR for the whole run,
        # with the per-group layer-decay scale still applied
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group.get("base_lr", args.lr) * param_group.get("lr_scale", 1.0)
        return args.lr
    # Sub-epoch (fractional) warmup when requested — see adjust_learning_rate_wsd.
    warmup_epochs = (args.epochs * warmup_fraction
                     if getattr(args, 'subepoch_lr_schedule', False)
                     else int(args.epochs * warmup_fraction))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        multiplier = epoch / warmup_epochs
    else:
        multiplier = (args.min_lr / args.lr) + (1.0 - args.min_lr / args.lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (args.epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        base = param_group.get("base_lr", args.lr)
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = base * multiplier * scale
    return args.lr * multiplier

def adjust_learning_rate_wsd(optimizer, epoch, args, warmup_fraction=0.1, decay_fraction=0.1):
    """
    Warmup-Stable-Decay (WSD) learning rate schedule with optional re-warmup
    for the annealing phase.

    - Warmup: linearly increase the learning rate from 0 to the peak learning rate over the warmup epochs.
    - Stable: keep the learning rate constant at the peak learning rate for the epochs between warmup and decay.
    - Decay (annealing): if annealing_rewarmup_epochs > 0, a brief re-warmup ramps from min_lr to peak LR
      before cosine decay continues; otherwise cosine annealing from peak LR to min LR.

    Supports per-group ``base_lr`` for hybrid optimizers (e.g. MUON + AdamW).
    When ``base_lr`` is present in a param group, the schedule is computed
    relative to that value instead of ``args.lr``.

    Args:
        warmup_fraction: Fraction of total epochs to use for the warmup phase (default: 0.1).
        decay_fraction: Fraction of total epochs to use for the decay phase (default: 0.1).
                        For example, with 100 total epochs and decay_fraction=0.1, the last 10 epochs will be decay.
    """
    if args.min_lr == args.blr:
        # constant-LR switch (min_lr set to blr): peak LR for the whole run,
        # with the per-group layer-decay scale still applied
        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group.get("base_lr", args.lr) * param_group.get("lr_scale", 1.0)
        return args.lr

    # Calculate the epoch when decay phase starts.
    # By default warmup/decay lengths are whole epochs — but int(epochs·frac)
    # rounds to 0 for epochs ≤ 9 (with frac=0.1), collapsing the schedule to a
    # constant peak LR with no warmup and, critically, no final anneal. That
    # inflates the loss of short / low-repetition runs.
    # ``--subepoch_lr_schedule`` keeps the
    # warmup/decay fractions sub-epoch so every run — even a 1-epoch run — warms
    # up and anneals to min_lr proportionally (the scheduler receives a
    # continuous fractional ``epoch``, so this is well-defined).
    if getattr(args, 'subepoch_lr_schedule', False):
        warmup_epochs = args.epochs * warmup_fraction
        decay_epochs = args.epochs * decay_fraction
    else:
        warmup_epochs = int(args.epochs * warmup_fraction)
        decay_epochs = int(args.epochs * decay_fraction)
    decay_start_epoch = args.epochs - decay_epochs

    rewarmup_epochs = getattr(args, 'annealing_rewarmup_epochs', 0)

    # Compute schedule multiplier: 0 at start of warmup, 1 at peak, decays during decay phase
    min_lr_ratio = args.min_lr / args.lr if args.lr > 0 else 0

    if warmup_epochs > 0 and epoch < warmup_epochs:
        # Warmup phase: linear ramp from 0 to 1
        multiplier = epoch / warmup_epochs
    elif epoch < decay_start_epoch:
        # Stable phase: constant at peak
        multiplier = 1.0
    elif rewarmup_epochs > 0 and epoch < decay_start_epoch + rewarmup_epochs:
        # Re-warmup at the start of annealing: ramp from min_lr_ratio to 1
        rewarmup_progress = (epoch - decay_start_epoch) / rewarmup_epochs
        multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * rewarmup_progress
    else:
        # Decay phase: cosine annealing from 1 to min_lr_ratio.
        # Floor the denominator to avoid div-by-zero; under sub-epoch scheduling
        # the decay window can be < 1 epoch, so use a tiny floor there instead of
        # 1 (which would stretch the cosine over a full epoch and prevent the
        # short run from ever reaching min_lr).
        effective_decay_start = decay_start_epoch + rewarmup_epochs
        _decay_floor = 1e-6 if getattr(args, 'subepoch_lr_schedule', False) else 1
        effective_decay_epochs = max(decay_epochs - rewarmup_epochs, _decay_floor)
        progress = (epoch - effective_decay_start) / effective_decay_epochs
        multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        base = param_group.get("base_lr", args.lr)
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = base * multiplier * scale

    # Return the default (AdamW) lr for logging
    return args.lr * multiplier