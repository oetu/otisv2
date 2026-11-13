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
import inspect
import sys
import time
from typing import Iterable, Optional

import torch

from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import r_regression
from sklearn.decomposition import PCA

import wandb

import matplotlib
matplotlib.use('Agg')           # prevents tkinter error
import matplotlib.pyplot as plt

import numpy as np

from timm.data.mixup import Mixup

import util.misc as misc
import util.lr_sched as lr_sched
import util.plot as plot


def _mask_kwargs(fn, attn_mask):
    """``attn_mask`` kwarg for padded batches, if ``fn`` accepts it (baseline models do not)."""
    if attn_mask is None or bool(attn_mask.all()):
        return {}
    if "attn_mask" not in inspect.signature(fn).parameters:
        return {}
    return {"attn_mask": attn_mask}


def _ovr_mean(score_fn, labels, probs, nb_classes):
    """Macro one-vs-rest score over the classes present in ``labels``.

    Equals sklearn's macro OvR AUROC / AUPRC when every class is present; a class
    that is missing from this split (no positives) is skipped instead of crashing.
    """
    y = np.asarray(labels)
    scores = [score_fn(y == c, probs[:, c]) for c in range(nb_classes) if 0 < (y == c).sum() < len(y)]
    return float(np.mean(scores)) if scores else float("nan")


def _compute_classification_metrics(logits, labels, probs, nb_classes):
    """Compute classification metrics, returning a dict of metric values."""
    preds = logits.argmax(dim=-1)

    metrics = {
        "f1": 100 * f1_score(y_true=labels, y_pred=preds, average="macro"),
        "precision": 100 * precision_score(y_true=labels, y_pred=preds, average="macro"),
        "recall": 100 * recall_score(y_true=labels, y_pred=preds, average="macro"),
        "acc": 100 * accuracy_score(y_true=labels, y_pred=preds),
        "acc_balanced": 100 * balanced_accuracy_score(y_true=labels, y_pred=preds),
        "cohen": 100 * cohen_kappa_score(y1=labels, y2=preds),
    }

    probs = np.asarray(probs)
    if nb_classes > 2:
        metrics["auroc"] = 100 * _ovr_mean(roc_auc_score, labels, probs, nb_classes)
        metrics["auprc"] = 100 * _ovr_mean(average_precision_score, labels, probs, nb_classes)
    else:
        metrics["auroc"] = 100 * roc_auc_score(y_true=labels, y_score=probs[:, 1], average="macro")
        metrics["auprc"] = 100 * average_precision_score(y_true=labels, y_score=probs[:, 1])

    return metrics


def _masked_mse(outputs, targets, targets_mask):
    """MSE over labelled entries only (mask = 1); equals the plain mean when nothing is masked."""
    sq = (outputs.float() - targets.float()) ** 2 * targets_mask
    return sq.sum() / targets_mask.sum().clamp(min=1)


def _compute_regression_metrics(logits, labels, masks=None):
    """Compute regression metrics, returning a dict with mean values and raw per-target arrays.

    ``masks`` (same shape as ``labels``, 1 = labelled) restricts every metric of
    target i to the rows where that target is labelled. Without it, masked
    entries would count as (0, 0) pairs and shrink RMSE / MAE.
    """
    if masks is None:
        masks = torch.ones_like(labels)
    masks = masks.bool()
    rmse, mae, pcc, r2 = [], [], [], []
    for i in range(labels.shape[-1]):
        m = masks[:, i]
        y, p = labels[m, i], logits[m, i]
        if y.numel() < 2:
            rmse.append(np.nan); mae.append(np.nan); pcc.append(np.nan); r2.append(np.nan)
            continue
        rmse.append(root_mean_squared_error(y, p))
        mae.append(mean_absolute_error(y, p))
        pcc.append(r_regression(p.view(-1, 1), y)[0])
        r2.append(r2_score(y, p))
    rmse, mae, pcc, r2 = (np.asarray(v, dtype=np.float64) for v in (rmse, mae, pcc, r2))

    # nanmean: a target without >= 2 labelled rows in this split is skipped
    metrics = {
        "rmse": np.nanmean(rmse, axis=-1),
        "mae": np.nanmean(mae, axis=-1),
        "pcc": np.nanmean(pcc, axis=-1),
        "r2": np.nanmean(r2, axis=-1),
    }
    metrics["_raw"] = {"rmse": rmse, "mae": mae, "pcc": pcc, "r2": r2}
    return metrics


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    training_history = {}
    
    # required for metrics calculation
    logits, labels, masks = [], [], []

    for data_iter_step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples, targets, targets_mask = batch[0], batch[1], batch[2]
        # patch-level padding mask (collate_fn_ft); None for older 3-tuple loaders
        attn_mask = batch[3] if len(batch) > 3 else None
        start_time = time.time()

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate_wsd(optimizer, data_iter_step / len(data_loader) + epoch, args,
                                                warmup_fraction=getattr(args, 'warmup_fraction', 0.1),
                                                decay_fraction=getattr(args, 'decay_fraction', 0.1))

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_mask = targets_mask.to(device, non_blocking=True)
        targets = targets * targets_mask

        if args.downstream_task == 'classification':
            targets_mask = targets_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)

        targets_metric = targets    # hard labels for the metrics (mixup makes targets soft)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device, non_blocking=True)
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            if attn_mask is not None:
                # timm Mixup blends each sample with its flip(0) partner
                attn_mask = torch.maximum(attn_mask, attn_mask.flip(0))

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            fwd = model.module.forward if hasattr(model, "module") else model.forward
            outputs = model(samples, **_mask_kwargs(fwd, attn_mask)) * targets_mask
            if args.downstream_task == 'regression':
                # mean over labelled entries only
                loss = _masked_mse(outputs, targets, targets_mask)
            else:
                loss = criterion(outputs, targets)

        loss_value = loss.item()

        # check on all ranks, so every rank exits together
        if not misc.all_ranks_finite(loss_value):
            print("Loss is {} on some rank, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        total_time = time.time() - start_time

        logits.append(outputs)
        labels.append(targets_metric)
        masks.append(targets_mask)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

        if args.wandb == True and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            if misc.is_main_process():
                wandb.log({"epoch_1000x": epoch_1000x,
                           "time_per_step[sec]": total_time,
                           "loss": loss_value_reduce,
                           "lr": max_lr}, step=epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    logits = torch.cat(logits, dim=0).to(device="cpu", dtype=torch.float32).detach()    # (B, num_classes)
    probs = torch.nn.functional.softmax(logits, dim=-1)                                 # (B, num_classes)
    labels = torch.cat(labels, dim=0).to(device="cpu").detach()                         # (B, )
    masks = torch.cat(masks, dim=0).to(device="cpu").detach()
    
    training_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.downstream_task == 'classification':
        cls_metrics = _compute_classification_metrics(logits, labels, probs, args.nb_classes)
        training_stats.update(cls_metrics)
    elif args.downstream_task == 'regression':
        reg_metrics = _compute_regression_metrics(logits, labels, masks)
        raw = reg_metrics.pop("_raw")
        training_stats.update(reg_metrics)

    # tensorboard
    if log_writer is not None:
        for key in (cls_metrics if args.downstream_task == 'classification' else reg_metrics):
            log_writer.add_scalar(f'perf/train_{key}', training_stats[key], epoch)

    # wandb
    if args.wandb == True:
        training_history['epoch'] = epoch
        if args.downstream_task == 'classification':
            training_history.update(cls_metrics)
        elif args.downstream_task == 'regression':
            training_history.update(reg_metrics)
            if targets.shape[-1] > 1:
                for i in range(targets.shape[-1]):
                    for name in ("rmse", "mae", "pcc", "r2"):
                        training_history[f'Train/{name.upper()}/{i}'] = raw[name][i]

    return training_stats, training_history


@torch.no_grad()
def evaluate(data_loader, model, device, epoch, log_writer=None, args=None):
    # switch to evaluation mode
    model.eval()

    if args.downstream_task == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.downstream_task == 'regression':
        criterion = torch.nn.MSELoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
    
    test_history = {}
    
    # required for metrics calculation
    embeddings, logits, labels, masks = [], [], [], []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        images = images.to(device, non_blocking=True)

        target = batch[1]
        target = target.to(device, non_blocking=True)

        target_mask = batch[2]
        target_mask = target_mask.to(device, non_blocking=True)
        target = target * target_mask

        if args.downstream_task == 'classification':
            target_mask = target_mask.unsqueeze(dim=-1).repeat(1, args.nb_classes)

        # compute output
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            attn_mask = batch[3].to(device, non_blocking=True) if len(batch) > 3 else None
            embedding = model.forward_features(images, **_mask_kwargs(model.forward_features, attn_mask))
            output = model.forward_head(embedding)
            output = output * target_mask
            if args.downstream_task == 'regression':
                # mean over labelled entries only
                loss = _masked_mse(output, target, target_mask)
            else:
                loss = criterion(output, target)

        if args.save_embeddings:
            embeddings.append(embedding)
        logits.append(output)
        labels.append(target)
        masks.append(target_mask)

        metric_logger.update(loss=loss.item())

    if args.wandb and args.plot_attention_map:
        attention_map = model.blocks[-1].attn.attn_map
        idx = 1 if args.batch_size > 1 else 0
        test_history["Attention"] = plot.plot_attention(images, attention_map, idx)

    if args.save_embeddings and misc.is_main_process():
        embeddings = torch.cat(embeddings, dim=0).to(device="cpu", dtype=torch.float32).detach() # (B, D)
        embeddings_path = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        
        file_name = f"embeddings_test.pt" if args.eval else f"embeddings_{epoch}.pt"
        torch.save(embeddings, os.path.join(embeddings_path, file_name))

    # gather the stats from all processes
    if epoch != -1:
        metric_logger.synchronize_between_processes()

    logits = torch.cat(logits, dim=0).to(device="cpu", dtype=torch.float32).detach()    # (B, num_classes)
    probs = torch.nn.functional.softmax(logits, dim=-1)                                 # (B, num_classes)
    labels = torch.cat(labels, dim=0).to(device="cpu").detach()                         # (B, 1)
    masks = torch.cat(masks, dim=0).to(device="cpu").detach()
    # --dist_eval: every rank saw one shard of the eval set; compute metrics on all
    logits = misc.gather_eval_tensor(logits, data_loader)
    labels = misc.gather_eval_tensor(labels, data_loader)
    masks = misc.gather_eval_tensor(masks, data_loader)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    if args.save_logits and misc.is_main_process():
        logits_path = os.path.join(args.output_dir, "logits")
        if not os.path.exists(logits_path):
            os.makedirs(logits_path)
        
        file_name = f"logits_test.pt" if args.eval else f"logits_{epoch}.pt"
        torch.save(logits, os.path.join(logits_path, file_name))

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if args.downstream_task == 'classification':
        cls_metrics = _compute_classification_metrics(logits, labels, probs, args.nb_classes)
        test_stats.update(cls_metrics)
    elif args.downstream_task == 'regression':
        reg_metrics = _compute_regression_metrics(logits, labels, masks)
        raw = reg_metrics.pop("_raw")
        test_stats.update(reg_metrics)

    if args.downstream_task == 'classification':
        print('* Acc@1 {acc:.2f} Acc@1 (balanced) {acc_balanced:.2f} Precision {precision:.2f} Recall {recall:.2f} F1 {f1:.2f} AUROC {auroc} AUPRC {auprc} loss {loss:.3f}'
              .format(**cls_metrics, loss=test_stats["loss"]))
    elif args.downstream_task == 'regression':
        print('* RMSE {rmse:.3f} MAE {mae:.3f} PCC {pcc:.3f} R2 {r2:.3f} loss {loss:.3f}'
              .format(**reg_metrics, loss=test_stats["loss"]))

    # tensorboard
    if log_writer is not None:
        for key in (cls_metrics if args.downstream_task == 'classification' else reg_metrics):
            log_writer.add_scalar(f'perf/test_{key}', test_stats[key], epoch)
        log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)

    # wandb
    if args.wandb == True:
        test_history = {'epoch': epoch, 'test_loss': test_stats['loss']}
        if args.downstream_task == 'classification':
            test_history.update({f'test_{k}': v for k, v in cls_metrics.items()})
        elif args.downstream_task == 'regression':
            test_history.update({f'test_{k}': v for k, v in reg_metrics.items()})
            if target.shape[-1] > 1:
                for i in range(target.shape[-1]):
                    for name in ("rmse", "mae", "pcc", "r2"):
                        test_history[f'Test/{name.upper()}/{i}'] = raw[name][i]

        if args.plot_embeddings and epoch % 1 == 0:
            font_size = 17
            plt.rcParams.update({'font.size': font_size})

            e = embeddings[::5]
            labels = labels[::5]

            # Perform PCA to reduce embeddings to 2D
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(e)

            # Calculate centroids for each label
            unique_labels = np.unique(labels)
            centroids = np.array([
                reduced_embeddings[labels.flatten() == label].mean(axis=0)
                for label in unique_labels
            ])

            # Plot the embeddings with labels
            colors = plt.cm.viridis(np.linspace(0, 1, len(unique_labels)))

            fig, axs = plt.subplots(1, 1, figsize=(8, 8))

            for label, color in zip(unique_labels, colors):
                mask = labels.flatten() == label
                alpha = 0.75
                axs.scatter(
                    reduced_embeddings[mask, 0],
                    reduced_embeddings[mask, 1],
                    color=color,
                    label=f"class {label}",
                    alpha=alpha
                )

            # Plot centroids
            axs.scatter(
                centroids[:, 0], 
                centroids[:, 1],
                c='red', marker='X', s=50, label='centroid'
            )

            axs.spines['top'].set_visible(False)
            axs.spines['right'].set_visible(False)

            # Add text for the number of samples in the upper left
            n_samples = len(labels)
            axs.text(
                0.03, 1.0, f"n = {n_samples:,}", 
                transform=axs.transAxes, ha="left", va="top", fontsize=font_size, 
            )

            axs.text(
                0.03, 0.94, f"bACC = {cls_metrics['acc_balanced']/100:.2%}",
                # transform=axs.transAxes, ha="left", va="top", fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
                transform=axs.transAxes, ha="left", va="top", fontsize=font_size, 
            )

            # Calculate explained variance
            explained_variance = pca.explained_variance_ratio_
            total_explained_variance = np.sum(explained_variance)
            axs.text(
                0.03, 0.88, f"v = {total_explained_variance:.2%}", 
                transform=axs.transAxes, ha="left", va="top", fontsize=font_size, 
            )

            plt.xticks([])
            plt.yticks([])

            plt.legend(loc="lower left")
            plt.tight_layout()
            test_history["PCA of val embeddings"] = wandb.Image(fig)
            plt.close('all')
    
    return test_stats, test_history


@torch.no_grad()
def extract_embeddings(data_loader, model, device, args=None):
    # switch to evaluation mode
    model.eval()
    
    # required for metrics calculation
    embeddings = []

    start_time = time.time()
    for idx, batch in enumerate(data_loader):
        if idx % 10 == 0:
            end_time = time.time()
            print(idx*args.batch_size, end_time-start_time)

        samples = batch[0]
        samples = samples.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            embedding = model.forward_features(samples)

        embeddings.append(embedding.detach().to(device="cpu", dtype=torch.float32))

    if misc.is_main_process():
        embeddings = torch.cat(embeddings, dim=0) # (B, D)
        embeddings_path = os.path.join(args.output_dir, "embeddings")
        if not os.path.exists(embeddings_path):
            os.makedirs(embeddings_path)
        
        file_name = f"embeddings_eval.pt"
        torch.save(embeddings, os.path.join(embeddings_path, file_name))