# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
"""Frozen evaluation of a pre-trained OTISv2 encoder.

The encoder stays frozen. Each data set is embedded once ([CLS] token and the
average of the patch tokens) and scored with three protocols:
- k-NN: the k nearest training samples by cosine similarity vote with the
  weights exp(similarity / temperature)
- few-shot: class prototypes from k_shots random training samples per class,
  cosine similarity to the prototypes, mean over n_episodes
- linear: multinomial logistic regression with an L2 penalty on standardised
  embeddings; the penalty strength is selected on a hold-out part of the
  training split
All scores are balanced accuracies.

Data, either or both:
- the UEA / UCR benchmark (--uea_data_base, --ucr_data_base), one folder per
  data set with train.pt, train_labels.pt, test.pt and test_labels.pt
- your own data (--data_path, --labels_path, --val_data_path, --val_labels_path)
The data files hold a tensor of the shape (N, V, T) or (N, C, V, T), or a list
of samples (see util/dataset.py). The label files hold a tensor of one-hot rows
or of class indices. All samples of a data set have one shape.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

import models_otis
import util.misc as misc
from util.benchmarks import UEA_DATASETS, UCR_DATASETS
from util.dataset import load_samples

POOLS = ('cls', 'gap')
C_GRID = [1e-2, 1e-1, 1.0, 10.0, 100.0]


def get_args_parser():
    parser = argparse.ArgumentParser('OTISv2 frozen evaluation', add_help=False)
    # Model parameters
    parser.add_argument('--finetune', required=True, type=str,
                        help='Pre-trained checkpoint')
    parser.add_argument('--model', default='', type=str,
                        help='Name of the pre-training model, e.g. otis_base_dec160d4b '
                             '(default: read from the checkpoint)')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size of the embedding pass')
    parser.add_argument('--precision', default='bf16', choices=['bf16', 'fp32'],
                        help='Precision of the embedding pass: bf16 autocast (default) or fp32 (TF32 off)')

    # Dataset parameters
    parser.add_argument('--uea_data_base', default='', type=str,
                        help='Folder with one sub-folder per UEA data set')
    parser.add_argument('--ucr_data_base', default='', type=str,
                        help='Folder with one sub-folder per UCR data set')
    parser.add_argument('--data_path', default='', type=str,
                        help='Own data: training split (labelled reference samples)')
    parser.add_argument('--labels_path', default='', type=str,
                        help='Own data: labels of the training split')
    parser.add_argument('--val_data_path', default='', type=str,
                        help='Own data: split to evaluate')
    parser.add_argument('--val_labels_path', default='', type=str,
                        help='Own data: labels of the split to evaluate')

    # Protocol parameters
    parser.add_argument('--knn_k', default=5, type=int,
                        help='Number of neighbours of the k-NN classifier')
    parser.add_argument('--knn_temperature', default=0.07, type=float,
                        help='Temperature of the k-NN vote weights')
    parser.add_argument('--k_shots', default=5, type=int,
                        help='Training samples per class of the few-shot classifier')
    parser.add_argument('--n_episodes', default=3, type=int,
                        help='Number of few-shot episodes (random draws of the training samples)')
    parser.add_argument('--no_linear', action='store_true', default=False,
                        help='Skip the linear classifier')

    parser.add_argument('--output_dir', default='', type=str,
                        help='Folder for the results (<family>.jsonl, summary.json)')
    parser.add_argument('--save_embeddings', action='store_true', default=False,
                        help='Also save the embeddings (needs --output_dir)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', default=0, type=int,
                        help='Seed of the few-shot draws')
    return parser


def build_encoder(checkpoint, model_name=''):
    """Build the pre-training model of a checkpoint and load its weights.

    Works for the released weights and for checkpoints of main_pretrain.py.
    Returns the model and the patch size.
    """
    stored = checkpoint.get('args') or {}
    stored = stored if isinstance(stored, dict) else vars(stored)
    info = checkpoint.get('release_info') or {}
    pre = info.get('pretrain', {})

    model_name = model_name or info.get('model') or stored.get('model')
    if not model_name:
        raise ValueError('The checkpoint does not name its model; pass --model.')
    if 'patch_size' in pre:
        patch_size = tuple(pre['patch_size'])
    else:
        patch_size = (stored.get('patch_height', 1), stored.get('patch_width', 24))

    kwargs = misc.checkpoint_arch_args(checkpoint)
    if 'swiglu' in stored:
        kwargs['swiglu'] = bool(stored['swiglu'])
    model = models_otis.__dict__[model_name](
        input_channels=pre.get('input_channels', stored.get('input_channels', 1)),
        time_steps=pre.get('time_steps', stored.get('time_steps', 1008)),
        patch_size=patch_size,
        drop_path=0.0, drop_path_decoder=0.0,
        **kwargs)

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    misc.check_encoder_load(msg, model)
    missing = [k for k in msg.missing_keys if k.startswith(('patch_embed', 'cls_token', 'register_tokens', 'norm.'))]
    if missing:
        raise RuntimeError(f'Encoder weights missing in the checkpoint: {missing[:4]}')
    print(f'Model = {model_name}, patch size {patch_size}, {kwargs}')
    return model, patch_size


def load_split(data_path, labels_path):
    """Samples (N, C, V, T) and class indices (N,) of one split."""
    labels = torch.load(labels_path, map_location='cpu', weights_only=False)

    samples = [sample if sample.dim() == 3 else sample.unsqueeze(0) for sample in load_samples(data_path)]
    if len({tuple(sample.shape) for sample in samples}) != 1:
        raise ValueError(f'{data_path}: all samples of a data set must have the same shape.')
    # missing values are set to 0
    X = torch.nan_to_num(torch.stack(samples, dim=0).float(), nan=0.0)
    y = labels.argmax(dim=-1).long() if labels.ndim == 2 else labels.long()
    return X, y


def load_dataset(data_base, name):
    folder = os.path.join(data_base, name)
    X_train, y_train = load_split(os.path.join(folder, 'train.pt'), os.path.join(folder, 'train_labels.pt'))
    X_test, y_test = load_split(os.path.join(folder, 'test.pt'), os.path.join(folder, 'test_labels.pt'))
    return X_train, y_train, X_test, y_test


@torch.no_grad()
def extract_embeddings(model, X, device, batch_size=64, precision='bf16'):
    """[CLS] embeddings and averaged patch embeddings (without register tokens), both (N, D)."""
    S = 1 + model.num_register_tokens  # cls + register
    cls_embs, gap_embs = [], []
    for i in range(0, len(X), batch_size):
        batch = X[i:i + batch_size].to(device, non_blocking=True)
        # (B, 1+R+N, D) = [CLS, register, patches]
        if precision == 'bf16':
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                emb = model.forward_encoder_all_patches(batch)
        else:
            emb = model.forward_encoder_all_patches(batch)
        emb = emb.float()
        cls_embs.append(emb[:, 0, :].cpu())
        gap_embs.append(emb[:, S:, :].mean(dim=1).cpu())
    return torch.cat(cls_embs, dim=0), torch.cat(gap_embs, dim=0)


def _l2_normalise(x):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm == 0, 1.0, norm)


def knn_acc(train_emb, train_y, test_emb, test_y, k=5, temperature=0.07):
    """Weighted k-NN on L2-normalised embeddings; k is capped at the number of training samples."""
    nb_classes = int(max(train_y.max(), test_y.max())) + 1
    train = _l2_normalise(train_emb.astype(np.float32))
    test = _l2_normalise(test_emb.astype(np.float32))
    k = min(k, train.shape[0])

    sim = test @ train.T
    idx = np.argpartition(-sim, k - 1, axis=1)[:, :k]
    rows = np.arange(sim.shape[0])[:, None]
    weights = np.exp(sim[rows, idx] / temperature)
    neighbour_y = train_y[idx]

    scores = np.zeros((sim.shape[0], nb_classes), dtype=np.float64)
    for c in range(nb_classes):
        scores[:, c] = (weights * (neighbour_y == c)).sum(axis=1)
    return float(balanced_accuracy_score(test_y, scores.argmax(axis=1)))


def fewshot_acc(train_emb, train_y, test_emb, test_y, nb_classes, k_shots, n_episodes, rng):
    """Prototypical classification from k_shots training samples per class, mean over the episodes."""
    train_emb = torch.from_numpy(train_emb.astype(np.float32))
    test_emb = torch.from_numpy(test_emb.astype(np.float32))
    train_y = torch.from_numpy(train_y.astype(np.int64))
    test_y = test_y.astype(int)

    accs = []
    for _ in range(n_episodes):
        support_idx = []
        for c in range(nb_classes):
            class_idx = (train_y == c).nonzero(as_tuple=True)[0].tolist()
            if len(class_idx) == 0:
                continue
            if len(class_idx) <= k_shots:
                support_idx.extend(class_idx)
            else:
                support_idx.extend(rng.choice(class_idx, size=k_shots, replace=False).tolist())
        support_y = train_y[support_idx]
        support_emb = train_emb[support_idx]

        prototypes = torch.zeros(nb_classes, support_emb.shape[1])
        for c in range(nb_classes):
            mask = support_y == c
            if mask.any():
                prototypes[c] = support_emb[mask].mean(dim=0)

        # cosine similarity classification
        prototypes = F.normalize(prototypes, dim=1)
        queries = F.normalize(test_emb, dim=1)
        preds = (queries @ prototypes.T).argmax(dim=1).numpy()
        accs.append(balanced_accuracy_score(test_y, preds))
    return float(np.mean(accs))


def _fit_logreg(X, y, C, max_iter=2000, tol=1e-6):
    """min_{W,b} 0.5 * ||W||^2 + C * sum_i CE(W x_i + b, y_i), solved with L-BFGS."""
    nb_classes = int(y.max().item()) + 1
    W = torch.zeros(X.shape[1], nb_classes, device=X.device, dtype=torch.float32, requires_grad=True)
    b = torch.zeros(nb_classes, device=X.device, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([W, b], lr=1.0, max_iter=max_iter,
                                  tolerance_grad=tol, tolerance_change=tol * 1e-2,
                                  history_size=20, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = 0.5 * (W * W).sum() + C * F.cross_entropy(X @ W + b, y, reduction='sum')
        loss.backward()
        return loss

    optimizer.step(closure)
    return W.detach(), b.detach()


def linear_acc(train_emb, train_y, test_emb, test_y, device):
    """Linear classifier on standardised embeddings. Returns the test score and the selected C.

    C is selected on a stratified 20% hold-out of the training split, or with a
    stratified k-fold when the training split is small. The final classifier is
    fitted on the full training split.
    """
    scaler = StandardScaler().fit(train_emb)
    train_emb, test_emb = scaler.transform(train_emb), scaler.transform(test_emb)
    nb_classes = int(np.unique(train_y).size)

    X_train = torch.from_numpy(train_emb).float().to(device)
    X_test = torch.from_numpy(test_emb).float().to(device)
    y_train = torch.from_numpy(train_y.astype(np.int64)).to(device)

    def val_score(train_idx, val_idx, C):
        train_idx_t, val_idx_t = torch.from_numpy(train_idx).to(device), torch.from_numpy(val_idx).to(device)
        W, b = _fit_logreg(X_train[train_idx_t], y_train[train_idx_t], C)
        preds = (X_train[val_idx_t] @ W + b).argmax(dim=1).cpu().numpy()
        return balanced_accuracy_score(train_y[val_idx], preds)

    n_per_class = np.min(np.bincount(train_y.astype(int), minlength=nb_classes))
    if n_per_class >= 5 and len(train_y) >= 50:
        splits = list(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(train_emb, train_y))
    elif n_per_class >= 2:
        splits = list(StratifiedKFold(n_splits=min(5, int(n_per_class)), shuffle=True,
                                      random_state=0).split(train_emb, train_y))
    else:
        splits = []

    best_C, best_score = 1.0, -1.0
    if splits:
        for C in C_GRID:
            score = float(np.mean([val_score(train_idx, val_idx, C) for train_idx, val_idx in splits]))
            if score > best_score:
                best_score, best_C = score, C

    W, b = _fit_logreg(X_train, y_train, best_C, max_iter=4000)
    preds = (X_test @ W + b).argmax(dim=1).cpu().numpy()
    return float(balanced_accuracy_score(test_y.astype(int), preds)), float(best_C)


def evaluate_dataset(model, patch_size, device, X_train, y_train, X_test, y_test, rng, args):
    """Embed one data set and score it. Returns the scores and the embeddings."""
    patch_width = patch_size[-1]
    if X_train.shape[-1] < patch_width:
        # shorter than one patch: zero pad to one patch
        pad = patch_width - X_train.shape[-1]
        X_train, X_test = F.pad(X_train, (0, pad)), F.pad(X_test, (0, pad))

    embs = {}
    embs['cls_train'], embs['gap_train'] = extract_embeddings(model, X_train, device, args.batch_size, args.precision)
    embs['cls_test'], embs['gap_test'] = extract_embeddings(model, X_test, device, args.batch_size, args.precision)
    embs = {k: v.numpy() for k, v in embs.items()}
    y_train, y_test = y_train.numpy(), y_test.numpy()
    nb_classes = int(max(y_train.max(), y_test.max())) + 1

    scores = {'n_train': int(y_train.size), 'n_test': int(y_test.size), 'nb_classes': nb_classes}
    for pool in POOLS:
        train_emb, test_emb = embs[f'{pool}_train'], embs[f'{pool}_test']
        scores[f'{pool}_knn'] = knn_acc(train_emb, y_train, test_emb, y_test, args.knn_k, args.knn_temperature)
        scores[f'{pool}_fewshot'] = fewshot_acc(train_emb, y_train, test_emb, y_test,
                                                nb_classes, args.k_shots, args.n_episodes, rng)
        if not args.no_linear:
            scores[f'{pool}_linear'], scores[f'{pool}_linear_C'] = linear_acc(train_emb, y_train, test_emb, y_test, device)
    embs['y_train'], embs['y_test'] = y_train, y_test
    return scores, embs


def main(args):
    print(f"job dir: {os.path.dirname(os.path.realpath(__file__))}")
    print("{}".format(args).replace(', ', ',\n'))

    if args.precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if args.save_embeddings and not args.output_dir:
        raise ValueError('--save_embeddings needs --output_dir')
    own = [args.data_path, args.labels_path, args.val_data_path, args.val_labels_path]
    if any(own) and not all(own):
        raise ValueError('Own data needs --data_path, --labels_path, --val_data_path and --val_labels_path')
    if not (args.uea_data_base or args.ucr_data_base or all(own)):
        raise ValueError('No data: pass --uea_data_base, --ucr_data_base or the four paths of your own data')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.finetune, map_location='cpu', weights_only=False)
    print("Load pre-trained checkpoint from: %s" % args.finetune)
    model, patch_size = build_encoder(checkpoint, args.model)
    model.eval().to(device)

    # (family, data sets), each data set a (name, loader) pair
    families = []
    if args.uea_data_base:
        families.append(('uea', [(entry[0], lambda name=entry[0]: load_dataset(args.uea_data_base, name))
                                 for entry in UEA_DATASETS]))
    if args.ucr_data_base:
        families.append(('ucr', [(entry[0], lambda name=entry[0]: load_dataset(args.ucr_data_base, name))
                                 for entry in UCR_DATASETS]))
    if all(own):
        families.append(('own', [('own', lambda: load_split(args.data_path, args.labels_path)
                                  + load_split(args.val_data_path, args.val_labels_path))]))

    protocols = ('fewshot', 'knn') if args.no_linear else ('fewshot', 'knn', 'linear')
    summary = {'checkpoint': os.path.basename(args.finetune), 'precision': args.precision,
               'knn_k': args.knn_k, 'knn_temperature': args.knn_temperature,
               'k_shots': args.k_shots, 'n_episodes': args.n_episodes, 'seed': args.seed}
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    start_time = time.time()
    for family, datasets in families:
        # one random state per family, used by the data sets in their list order
        rng = np.random.RandomState(args.seed)
        rows = []
        for name, loader in datasets:
            try:
                X_train, y_train, X_test, y_test = loader()
            except FileNotFoundError:
                print(f"[{family}] SKIP {name} (data not found)")
                continue
            scores, embs = evaluate_dataset(model, patch_size, device, X_train, y_train, X_test, y_test, rng, args)
            rows.append({'family': family, 'dataset': name, **scores})
            print(f"[{family}] {name:30s} " + " | ".join(
                f"{pool}: " + " ".join(f"{p} {scores[f'{pool}_{p}']:.3f}" for p in protocols) for pool in POOLS))

            if args.output_dir:
                with open(os.path.join(args.output_dir, f'{family}.jsonl'), 'a' if len(rows) > 1 else 'w') as f:
                    f.write(json.dumps(rows[-1]) + '\n')
            if args.save_embeddings:
                folder = os.path.join(args.output_dir, 'embeddings', family, name)
                os.makedirs(folder, exist_ok=True)
                np.savez(os.path.join(folder, 'train.npz'), y=embs['y_train'], cls=embs['cls_train'], gap=embs['gap_train'])
                np.savez(os.path.join(folder, 'test.npz'), y=embs['y_test'], cls=embs['cls_test'], gap=embs['gap_test'])

        # mean balanced accuracy over the data sets, in percent
        summary[family] = {'nb_datasets': len(rows)}
        for pool in POOLS:
            for p in protocols:
                summary[family][f'{pool}_{p}'] = float(np.mean([row[f'{pool}_{p}'] for row in rows])) * 100 if rows else None
        if rows:
            print(f"[{family}] mean balanced accuracy over {len(rows)} data set(s): " + " | ".join(
                f"{pool}: " + " ".join(f"{p} {summary[family][f'{pool}_{p}']:.2f}%" for p in protocols) for pool in POOLS))

    if args.output_dir:
        with open(os.path.join(args.output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
    print(f"Evaluation time {time.time() - start_time:.0f} s")


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
