# Copyright (c) Oezguen Turgut.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Build the UEA / UCR benchmark files of the frozen evaluation from the archive files.
#
# Reads <data_base>/<name>/<name>_TRAIN.ts and <name>_TEST.ts and writes train.pt,
# train_labels.pt, test.pt and test_labels.pt to <output_base>/<name>/ (default: the
# same folder). The data sets are listed in benchmarks.py.
#
# Steps per data set:
#   1. Read both splits. Shorter series of a split are zero padded to its longest series.
#   2. Data sets with missing values: the missing values of each series are filled by
#      linear interpolation in time (the first / last valid value is held at the edges),
#      and shorter series are extended in the same way.
#   3. Both splits are normalised with the mean and standard deviation of the training
#      split (over all its values, outliers ignored) and clamped to [-3, 3].
#   4. The class labels are encoded as one-hot rows.
#
# Reads the archive files with the package aeon (part of the environment).
#
# Usage:
#   python3 util/prepare_uea_ucr.py --archive uea --data_base /path/to/UEA
#   python3 util/prepare_uea_ucr.py --archive ucr --data_base /path/to/UCR
# --------------------------------------------------------
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmarks import UEA_DATASETS, UCR_DATASETS
from preprocess import compute_stats, preprocess


def to_array(samples, fill_value: float) -> np.ndarray:
    """Return a split as an array (N, V, T); shorter series are padded with fill_value."""
    if not isinstance(samples, list):
        return np.asarray(samples, dtype=np.float64)
    max_t = max(s.shape[-1] for s in samples)
    data = np.full((len(samples), samples[0].shape[0], max_t), fill_value, dtype=np.float64)
    for i, s in enumerate(samples):
        data[i, :, :s.shape[-1]] = s
    return data


def has_missing(samples) -> bool:
    if isinstance(samples, list):
        return any(np.isnan(s).any() for s in samples)
    return bool(np.isnan(samples).any())


def interpolate_missing(data: np.ndarray) -> np.ndarray:
    """Fill NaN entries of each series by linear interpolation in time; edge values are held."""
    data = data.copy()
    for i in range(data.shape[0]):
        for v in range(data.shape[1]):
            x = data[i, v]
            missing = np.isnan(x)
            if not missing.any():
                continue
            if missing.all():
                data[i, v] = 0.0
                continue
            t = np.arange(x.size)
            x[missing] = np.interp(t[missing], t[~missing], x[~missing])
    return data


def labels_to_onehot(labels, classes) -> torch.Tensor:
    """One-hot rows (N, number of classes); classes is the sorted list of all class labels."""
    label_map = {label: i for i, label in enumerate(classes)}
    indices = np.array([label_map[label] for label in labels])
    onehot = np.zeros((len(indices), len(classes)), dtype=np.int32)
    onehot[np.arange(len(indices)), indices] = 1
    return torch.tensor(onehot, dtype=torch.int32)


def process_dataset(name: str, data_base: str, output_base: str) -> bool:
    from aeon.datasets import load_from_ts_file

    train_path = os.path.join(data_base, name, f"{name}_TRAIN.ts")
    test_path = os.path.join(data_base, name, f"{name}_TEST.ts")
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"SKIP {name}: {name}_TRAIN.ts / {name}_TEST.ts not found in {os.path.join(data_base, name)}")
        return False

    X_train, y_train = load_from_ts_file(train_path)
    X_test, y_test = load_from_ts_file(test_path)

    missing = has_missing(X_train) or has_missing(X_test)
    fill_value = np.nan if missing else 0.0
    X_train, X_test = to_array(X_train, fill_value), to_array(X_test, fill_value)
    if missing:
        X_train, X_test = interpolate_missing(X_train), interpolate_missing(X_test)

    # statistics of the training split, applied to both splits
    stats = compute_stats(X_train, mode="sample", ignore_outliers=True, outlier_threshold=3.0, global_stats=True)
    X_train, X_test = preprocess(X_train, stats=stats), preprocess(X_test, stats=stats)

    classes = sorted(set(y_train) | set(y_test))
    out_dir = os.path.join(output_base, name)
    os.makedirs(out_dir, exist_ok=True)
    for split, X, y in (("train", X_train, y_train), ("test", X_test, y_test)):
        samples = [(name, torch.tensor(X[i], dtype=torch.float32)) for i in range(X.shape[0])]
        torch.save(samples, os.path.join(out_dir, f"{split}.pt"))
        torch.save(labels_to_onehot(y, classes), os.path.join(out_dir, f"{split}_labels.pt"))

    print(f"{name}: train {X_train.shape}, test {X_test.shape}, {len(classes)} classes"
          + (", missing values interpolated" if missing else ""))
    return True


def main():
    parser = argparse.ArgumentParser(description="Build the UEA / UCR benchmark files of the frozen evaluation.")
    parser.add_argument("--archive", type=str, required=True, choices=["uea", "ucr"],
                        help="Archive to process.")
    parser.add_argument("--data_base", type=str, required=True,
                        help="Folder with one sub-folder per data set, each with <name>_TRAIN.ts and <name>_TEST.ts.")
    parser.add_argument("--output_base", type=str, default=None,
                        help="Folder for the processed files (default: --data_base).")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Process only these data sets (default: all data sets of the archive list).")
    args = parser.parse_args()

    names = [entry[0] for entry in (UEA_DATASETS if args.archive == "uea" else UCR_DATASETS)]
    if args.datasets:
        unknown = sorted(set(args.datasets) - set(names))
        if unknown:
            raise ValueError(f"Not in the {args.archive} list: {unknown}")
        names = [name for name in names if name in args.datasets]

    done = [process_dataset(name, args.data_base, args.output_base or args.data_base) for name in names]
    print(f"Done: {sum(done)} of {len(names)} data sets.")


if __name__ == "__main__":
    main()
