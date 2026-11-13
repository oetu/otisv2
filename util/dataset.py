# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
from typing import Any, Tuple

import os

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

import util.augmentations as augmentations
import util.transformations as transformations


def load_samples(data_path:str) -> list:
    """
    Load the samples of a data file as a list of tensors, each of the shape (V, T) or (C, V, T).

    The file holds one of the following:
    - a tensor of the shape (N, V, T) or (N, C, V, T)
    - a list with one tensor per sample
    - a list of (name, sample) tuples; the name is not used
    """
    raw = torch.load(data_path, map_location=torch.device('cpu'), weights_only=False)

    if isinstance(raw, np.ndarray):
        raw = torch.from_numpy(raw)
    if isinstance(raw, torch.Tensor):
        if raw.dim() not in (3, 4):
            raise ValueError(f"{data_path}: expected a tensor of the shape (N, V, T) or (N, C, V, T), got {tuple(raw.shape)}.")
        return list(raw)

    return [sample[1] if isinstance(sample, (tuple, list)) else sample for sample in raw]


class TimeSeriesDataset(Dataset):
    """
    Dataset for any-domain time series analysis.
    """
    def __init__(self, data_path:str, labels_path:str=None, labels_mask_path:str=None,
                 downstream_task:str=None, weighted_loss:bool=False,
                 univariate:bool=False,
                 train:bool=False,
                 test:bool=False,
                 N_val:int=1,
                 args=None,
                 **kwargs) -> None: # Added kwargs to safely ignore legacy args if passed
        """
            labels_path: path to labels (finetuning / online evaluation)
            labels_mask_path: path to labels masks (finetuning / online evaluation)
            downstream_task: downstream task (finetuning / online evaluation)
            univariate: analyse each variate independently
            train: training or validation / test
            N_val: nb of chunks to validate / test
        """
        # one tensor per sample, of the shape (C, V, T) or (V, T)
        samples = load_samples(data_path)

        if samples[0].dim() == 3:
            data = [sample[None, ...] for sample in samples]  # data_shape : (1, C, V, T)
        else:
            # [None, ] to add auxiliary channel (C) dimension
            data = [sample[None, None, ...] for sample in samples]  # data_shape : (1, C, V, T)

        self.univariate = univariate

        self.data = data

        # one-hot-encdoded labels defined as torch.tensor()
        if labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'), weights_only=False) # load to ram
        else:
            self.labels = torch.zeros(size=(len(self.data), ))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu'), weights_only=False) # load to ram
        else:
            self.labels_mask = torch.ones_like(self.labels)

        self.downstream_task = downstream_task
        
        self.weighted_loss = weighted_loss
        self.class_weights = None
        if self.downstream_task == "classification" and weighted_loss == True:
            self.class_weights = torch.sqrt(len(self.labels) / self.labels.argmax(-1).unique(return_counts=True)[-1])

        self.train = train 
        self.test = test
        self.N_val = N_val
        self.args = args

    def __len__(self) -> int:
        """return the number of samples in the dataset"""
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        """return a sample from the dataset at index idx"""
        # (1, C, V, T)
        data = self.data[idx]
        if self.univariate:
            # transform a multivariate time series with V variates into V univariate time series
            # (V, C, 1, T)
            data = data.permute(2, 1, 0, 3)
        
        # number of samples to process
        N = data.shape[0]

        # (N=N_val,   C, V, T) if multivariate analysis, 
        # (N=N_val*V, C, 1, T) else
        if self.train == False:
            # validate / test on more than one chunk per sample
            # Calculate number of overlapping chunks
            stride = 32 # ili : 4, traffic : 32
            if self.test == True:
                stride = 8
            
            N_val = max(int(((data.shape[-1] - self.args.time_steps) / stride) + 1), 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            # Create overlapping chunks
            data_chunks = [ data[..., i*stride:(i*stride+self.args.time_steps)] for i in range(N_val) ]

            # Concatenate chunks
            # (N, C, x, T), where x either V or 1
            data = torch.cat(data_chunks, dim=0)
        else:
            N_val = max(data.shape[-1] - self.args.time_steps + 1, 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            transform = transforms.Compose([
                augmentations.CropResizing(fixed_resize_len=self.args.time_steps,
                                           lower_bnd=self.args.crop_lower_bnd,
                                           upper_bnd=self.args.crop_upper_bnd,
                                           resize=True),
                # transformations.Normalisation(),
                augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                augmentations.Jitter(sigma=self.args.jitter_sigma),
                augmentations.Rescaling(sigma=self.args.rescaling_sigma),
            ])

            # Create random chunks
            data_chunks = [ transform(data) for i in range(N_val) ]

            # Concatenate chunks
            # (N, C, x, T), where x either V or 1
            data = torch.cat(data_chunks, dim=0)

        if self.downstream_task == 'regression':
            label = self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd]
            label_mask = self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            label = self.labels[idx].type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        # Expand labels to match N chunks
        label = [label for i in range(N)]
        label_mask = [label_mask for i in range(N)]

        return data, label, label_mask, self.args.patch_size, self.args.time_steps, self.univariate

    @staticmethod
    def collate_fn(batch):
        # (p, q)
        patch_size = batch[0][3]
        # Patch grid per sample with floor: a partial last patch is dropped, the
        # same as the downstream PatchEmbed conv does. The grid is decided per
        # sample, never by the batch-mates (the former global "+1" marked an
        # all-zero patch as real, or masked a real partial patch, depending on
        # whether some sample in the batch reached time_steps).
        cap = max(sample[4] for sample in batch)    # crop cap along time (time_steps)
        assert cap % patch_size[-1] == 0, f"time_steps {cap} must be a multiple of the patch width {patch_size[-1]}"
        grid_width = torch.tensor([max(min(sample[0].shape[-1], cap) // patch_size[-1], 1) for sample in batch])
        grid_height = torch.tensor([sample[0].shape[-2] // patch_size[-2] for sample in batch])

        # Determine the largest shape in the batch
        shape = [data.shape for sample in batch for data in sample[0]]
        max_values = [max(x) for x in zip(*shape)]
        max_variates = max_values[-2]
        max_timesteps = int(grid_width.max()) * patch_size[-1]    # multiple of q; longer data is cropped by the pad

        # Zero pad the input data to the largest shape
        # (B, C, V_max, T_max)
        data = [torch.nn.functional.pad(data.unsqueeze(0),
                                        pad=(0, int(max_timesteps - data.shape[-1]), 0, int(max_variates - data.shape[-2])),
                                        mode="constant", value=0) for sample in batch for data in sample[0]]
        data = torch.cat(data, dim=0)

        # Create the attention mask
        # (B, V'_max, T'_max), with V'_max=V_max/p, T'_max=T_max/p
        attn_mask = [torch.nn.functional.pad(torch.ones(size=(grid_height[idx], grid_width[idx])),
                                             pad=(0, int(grid_width.max() - grid_width[idx]), 0, int(grid_height.max() - grid_height[idx])),
                                             mode="constant", value=0) for idx, sample in enumerate(batch) for i in range(len(sample[0]))]
        attn_mask = torch.stack(attn_mask, dim=0)

        return data, attn_mask
    
    @staticmethod
    def collate_fn_ft(batch):
        # (B, C, V, T) — right-zero-pad along the time axis when samples have
        # different lengths (e.g. CropResizing with resize=False).
        items = [data for sample in batch for data in sample[0]]
        lengths = [d.shape[-1] for d in items]
        max_T = max(lengths)
        if any(t != max_T for t in lengths):
            items = [torch.nn.functional.pad(d, (0, max_T - d.shape[-1])) for d in items]
        data = torch.cat([d.unsqueeze(0) for d in items], dim=0)
        # (B, 1)
        label = torch.stack([label for sample in batch for label in sample[1]], dim=0)
        # (B, 1)
        label_mask = torch.stack([label_mask for sample in batch for label_mask in sample[2]], dim=0)
        # (B, V', T') patch-level padding mask, 1 = real; floor per sample, as the
        # PatchEmbed conv does. The encoder excludes padding from attention/pooling.
        attn_mask = TimeSeriesDataset._ft_patch_mask(data, lengths, batch[0][3])

        return data, label, label_mask, attn_mask

    @staticmethod
    def _ft_patch_mask(data, lengths, patch_size):
        p, q = patch_size[-2], patch_size[-1]
        attn_mask = torch.zeros(data.shape[0], data.shape[-2] // p, data.shape[-1] // q)
        for i, t in enumerate(lengths):
            attn_mask[i, :, :max(t // q, 1)] = 1
        return attn_mask


class TimeSeriesDatasetMmap(Dataset):
    """
    Memory-mapped dataset for any-domain time series analysis.
    Drop-in replacement for TimeSeriesDataset that loads data from
    numpy mmap files instead of a single .pt file.

    Expected files in data_path directory:
      - data.bin         — flat float32 binary with all tensor values
      - index.npy        — (N, 3) int64 array: [offset, dim0, dim1] per sample

    Any sibling ``domains.npy`` / ``domain_names.npy`` files (legacy artefacts
    from the multi-domain era) are ignored — the model is fully
    domain-agnostic now.
    """
    def __init__(self, data_path: str, labels_path: str = None, labels_mask_path: str = None,
                 downstream_task: str = None, weighted_loss: bool = False,
                 univariate: bool = False,
                 train: bool = False,
                 test: bool = False,
                 N_val: int = 1,
                 args=None,
                 subset_tokens: int = None,
                 subset_seed: int = 2026,
                 **kwargs) -> None:
        # Load metadata (tiny, instant)
        self.index = np.load(os.path.join(data_path, "index.npy"))         # (N, 3)

        # Open memory-mapped data (no actual loading)
        self.fp = np.memmap(os.path.join(data_path, "data.bin"), dtype=np.float32, mode="r")

        N = len(self.index)
        self.univariate = univariate

        # Labels
        if labels_path:
            self.labels = torch.load(labels_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            self.labels = torch.zeros(size=(N,))

        if labels_mask_path:
            self.labels_mask = torch.load(labels_mask_path, map_location=torch.device('cpu'), weights_only=False)
        else:
            self.labels_mask = torch.ones_like(self.labels)

        self.downstream_task = downstream_task

        self.weighted_loss = weighted_loss
        self.class_weights = None
        if self.downstream_task == "classification" and weighted_loss:
            self.class_weights = torch.sqrt(len(self.labels) / self.labels.argmax(-1).unique(return_counts=True)[-1])

        self.train = train
        self.test = test
        self.N_val = N_val
        self.args = args

        # Optional token-budget subset, stratified by domain. The selection is
        # deterministic and *nested*: any larger subset_tokens budget is a
        # strict superset of every smaller one (same seed, same data path).
        if subset_tokens is not None and train:
            self._restrict_to_token_budget(data_path, subset_tokens, subset_seed)

    # ------------------------------------------------------------------
    # Token-budget subsetting
    # ------------------------------------------------------------------
    def _restrict_to_token_budget(self, data_path, subset_tokens, seed):
        """Filter ``self.index`` (and labels) down to the smallest stratified
        subset whose tokens-per-epoch sum reaches ``subset_tokens``.

        Tokens per sample = ``V × (T // P)`` with ``T = args.time_steps`` and
        ``P = args.patch_size``; this matches the actual per-epoch token count
        the model sees during pre-training (one fixed-length T crop per
        sample). Stratification preserves each domain's per-epoch token
        share. Within each domain, samples are picked in a fixed seeded
        permutation, so a 4M-token subset is a strict prefix of an 8M-token
        subset (for any K1 ≤ K2 the per-domain quota is non-decreasing and
        we accumulate samples in the *same* deterministic order).

        Falls back to the full dataset if ``subset_tokens`` exceeds the
        total tokens available.
        """
        if self.args is None or not hasattr(self.args, "time_steps") or not hasattr(self.args, "patch_width"):
            raise ValueError("subset_tokens requires args.time_steps and args.patch_width to be set")
        T = int(self.args.time_steps)
        P = int(self.args.patch_width)  # patch size along time axis
        if T <= 0 or P <= 0 or T % P != 0:
            raise ValueError(f"need T % P == 0; got T={T}, P={P}")
        patches_per_sample_time = T // P

        V = self.index[:, 1].astype(np.int64)
        # real patches per sample: a sample shorter than time_steps yields
        # min(T, time_steps) // P patches, not T // P (e.g. T=720 -> 30, not 42)
        T_native = self.index[:, 2].astype(np.int64)
        tokens_per_sample = V * (np.minimum(T_native, T) // P)   # (N,)

        # Stratify by domain when domains.npy is available; otherwise treat
        # the entire dataset as a single stratum.
        dom_path = os.path.join(data_path, "domains.npy")
        if os.path.exists(dom_path):
            domains = np.load(dom_path)
        else:
            domains = np.zeros(len(self.index), dtype=np.int64)

        total_tokens = int(tokens_per_sample.sum())
        if subset_tokens >= total_tokens:
            print(f"[subset_tokens] requested {subset_tokens:,} >= corpus total "
                  f"{total_tokens:,}; using full dataset")
            return

        rng = np.random.default_rng(seed)
        selected_global = []
        cumulative_tokens = 0

        unique_domains = np.unique(domains)
        for d in unique_domains:
            cand = np.flatnonzero(domains == d)
            # Fixed permutation per domain. Same seed -> same permutation
            # regardless of subset_tokens, so prefixes nest.
            perm = rng.permutation(cand)

            # Domain quota proportional to its share of the corpus' tokens.
            d_total = int(tokens_per_sample[cand].sum())
            quota = int(round(subset_tokens * (d_total / total_tokens)))
            if quota <= 0:
                continue

            # Greedy prefix that meets the quota (or exhausts the domain).
            csum = np.cumsum(tokens_per_sample[perm])
            n_take = int(np.searchsorted(csum, quota, side="left")) + 1
            n_take = min(n_take, len(perm))
            chosen = perm[:n_take]
            selected_global.append(chosen)
            cumulative_tokens += int(tokens_per_sample[chosen].sum())

        sel = np.sort(np.concatenate(selected_global)) if selected_global \
            else np.empty(0, dtype=np.int64)

        kept_tokens = int(tokens_per_sample[sel].sum())
        print(f"[subset_tokens] target={subset_tokens:,} -> kept {len(sel):,}/"
              f"{len(self.index):,} samples = {kept_tokens:,} tokens "
              f"({kept_tokens/total_tokens*100:.2f}% of corpus); "
              f"per-domain shares preserved, seed={seed}")

        # Apply the filter. Labels / labels_mask are kept aligned when their
        # length matches the original sample count.
        N_full = len(self.index)
        self.index = self.index[sel]
        if isinstance(self.labels, torch.Tensor) and self.labels.shape[0] == N_full:
            self.labels = self.labels[sel]
        if isinstance(self.labels_mask, torch.Tensor) and self.labels_mask.shape[0] == N_full:
            self.labels_mask = self.labels_mask[sel]

    def _load_sample(self, idx):
        """Load a single sample from mmap as a torch tensor with shape (1, 1, V, T)."""
        o, d0, d1 = int(self.index[idx, 0]), int(self.index[idx, 1]), int(self.index[idx, 2])
        arr = self.fp[o: o + d0 * d1].reshape(d0, d1)
        # .copy() to get a writable tensor (mmap is read-only)
        # unsqueeze twice: batch dim + channel dim -> (1, 1, V, T)
        return torch.from_numpy(arr.copy()).unsqueeze(0).unsqueeze(0)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx) -> Tuple[Any, Any]:
        # (1, C, V, T) where C=1
        data = self._load_sample(idx)

        if self.univariate:
            # (V, C, 1, T)
            data = data.permute(2, 1, 0, 3)

        N = data.shape[0]

        if not self.train:
            stride = 32
            if self.test:
                stride = 8

            N_val = max(int(((data.shape[-1] - self.args.time_steps) / stride) + 1), 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            data_chunks = [data[..., i * stride:(i * stride + self.args.time_steps)] for i in range(N_val)]
            data = torch.cat(data_chunks, dim=0)
        else:
            N_val = max(data.shape[-1] - self.args.time_steps + 1, 1)
            if self.N_val != -1:
                N_val = min(N_val, self.N_val)
            N *= N_val

            transform = transforms.Compose([
                augmentations.CropResizing(fixed_resize_len=self.args.time_steps,
                                           lower_bnd=self.args.crop_lower_bnd,
                                           upper_bnd=self.args.crop_upper_bnd,
                                           resize=True),
                augmentations.FTSurrogate(phase_noise_magnitude=self.args.ft_surr_phase_noise, prob=0.5),
                augmentations.Jitter(sigma=self.args.jitter_sigma),
                augmentations.Rescaling(sigma=self.args.rescaling_sigma),
            ])

            data_chunks = [transform(data) for i in range(N_val)]
            data = torch.cat(data_chunks, dim=0)

        if self.downstream_task == 'regression':
            label = self.labels[idx][..., self.args.lower_bnd:self.args.upper_bnd]
            label_mask = self.labels_mask[idx][..., self.args.lower_bnd:self.args.upper_bnd]
        else:
            label = self.labels[idx].type(torch.LongTensor).argmax(dim=-1)
            label_mask = torch.ones_like(label)

        label = [label for i in range(N)]
        label_mask = [label_mask for i in range(N)]

        return data, label, label_mask, self.args.patch_size, self.args.time_steps, self.univariate

    @staticmethod
    def collate_fn(batch):
        # (p, q)
        patch_size = batch[0][3]
        # Patch grid per sample with floor: a partial last patch is dropped, the
        # same as the downstream PatchEmbed conv does. The grid is decided per
        # sample, never by the batch-mates (the former global "+1" marked an
        # all-zero patch as real, or masked a real partial patch, depending on
        # whether some sample in the batch reached time_steps).
        cap = max(sample[4] for sample in batch)    # crop cap along time (time_steps)
        assert cap % patch_size[-1] == 0, f"time_steps {cap} must be a multiple of the patch width {patch_size[-1]}"
        grid_width = torch.tensor([max(min(sample[0].shape[-1], cap) // patch_size[-1], 1) for sample in batch])
        grid_height = torch.tensor([sample[0].shape[-2] // patch_size[-2] for sample in batch])

        # Determine the largest shape in the batch
        shape = [data.shape for sample in batch for data in sample[0]]
        max_values = [max(x) for x in zip(*shape)]
        max_variates = max_values[-2]
        max_timesteps = int(grid_width.max()) * patch_size[-1]    # multiple of q; longer data is cropped by the pad

        # Zero pad the input data to the largest shape
        # (B, C, V_max, T_max)
        data = [torch.nn.functional.pad(data.unsqueeze(0),
                                        pad=(0, int(max_timesteps - data.shape[-1]), 0, int(max_variates - data.shape[-2])),
                                        mode="constant", value=0) for sample in batch for data in sample[0]]
        data = torch.cat(data, dim=0)

        # Create the attention mask
        # (B, V'_max, T'_max)
        attn_mask = [torch.nn.functional.pad(torch.ones(size=(grid_height[idx], grid_width[idx])),
                                             pad=(0, int(grid_width.max() - grid_width[idx]), 0, int(grid_height.max() - grid_height[idx])),
                                             mode="constant", value=0) for idx, sample in enumerate(batch) for i in range(len(sample[0]))]
        attn_mask = torch.stack(attn_mask, dim=0)

        return data, attn_mask

    @staticmethod
    def collate_fn_ft(batch):
        # (B, C, V, T)
        data = torch.cat([data.unsqueeze(0) for sample in batch for data in sample[0]], dim=0)
        # (B, 1)
        label = torch.stack([label for sample in batch for label in sample[1]], dim=0)
        # (B, 1)
        label_mask = torch.stack([label_mask for sample in batch for label_mask in sample[2]], dim=0)
        # (B, V', T') padding mask: samples are concatenated, so all have the same length
        attn_mask = TimeSeriesDataset._ft_patch_mask(data, [data.shape[-1]] * data.shape[0], batch[0][3])

        return data, label, label_mask, attn_mask
