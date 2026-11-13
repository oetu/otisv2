# Copyright (c) Oezguen Turgut.
# All rights reserved.
"""
Track patch-level cosine similarity on synthetic sine waves during pre-training.

Inspired by Figure 6 of the DINOv3 paper, which visualises how the similarity
between a single reference patch and all other patches evolves over training.
For each of two sine waves with distinct frequency, we produce two subplots:

  1. Cosine similarity of every patch to a single reference patch placed on an
     arbitrary peak of the signal (matches the DINOv3 figure).
  2. Cosine similarity of every patch to a *mean peak prototype*. The set of
     "peak patches" is selected via a frequency-adjusted percentile so that
     roughly one peak patch is picked per cycle, regardless of the sine wave's
     period.

Why the percentile must depend on frequency
--------------------------------------------
With ``T`` time steps, patch width ``W`` and dominant period ``P``, there are
``N = T / W`` patches and roughly ``N_cycles = T / P`` peaks. A fixed 75th
percentile (top 25%) selects ``N/4`` patches, i.e. ``N / (4 * N_cycles)``
patches per cycle. For a fast sine (e.g. ``P = 160``, ``W = 24`` → 6.67
patches/period) this is ~1.67 patches/peak — fine. For a slow sine
(e.g. ``P = 480`` → 20 patches/period) it is 5 patches/peak, which spans the
entire rising/falling slope and dilutes the prototype.

Picking ``n_peak = N_cycles`` (one peak patch per cycle) keeps the prototype
concentrated on the actual maxima for *both* fast and slow sine waves. The
corresponding percentile is ``100 * (1 - n_peak / N)``.
"""

import os

import numpy as np
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")  # prevents tkinter error
import matplotlib.pyplot as plt

import wandb


# Paths of the synthetic sine data files. Set the environment variables
# OTIS_SINE_PATH and OTIS_SINE_MOD_PATH, or pass ``sine_path`` and
# ``sine_mod_path`` (or custom ``sources``) to ``plot_sine_patch_similarity``.
# Without a path the figure is skipped.
SINE_DATA_PATH = os.environ.get("OTIS_SINE_PATH", "")
SINE_MOD_DATA_PATH = os.environ.get("OTIS_SINE_MOD_PATH", "")

# Three reference signals:
#   1. slow pure sine (sample 10, period 480, 5 cycles in the default
#      2400-timestep window) analysed with a peak prototype,
#   2. custom-shaped sine_mod with 6 big peaks (sample 30) analysed with a
#      peak prototype,
#   3. custom-shaped sine_mod with 12 big peaks (sample 1667) analysed with a
#      *flat-patch* prototype: the reference patch covers a constant-valued
#      flattened bump, and the right-hand prototype averages every patch
#      whose raw value matches that flat level.
# Each entry is ``(label, file_path, sample_idx, mode)`` with mode in
# {"peak", "flat"}.
DEFAULT_SOURCES = (
    ("sine",     SINE_DATA_PATH,     10,   "peak"),
    ("sine_mod", SINE_MOD_DATA_PATH, 30,   "peak"),
    ("sine_mod", SINE_MOD_DATA_PATH, 1667, "flat"),
)

_SAMPLE_CACHE = {}


def _estimate_period(signal: np.ndarray) -> float:
    """Return the dominant period (in time steps) via large-peak spacing.

    The modified sine waves have arbitrary DC offset and small intra-cycle
    bumps, so plain FFT can lock onto the envelope or sub-harmonics. Using
    the spacing of prominent peaks is robust for both pure and modified
    sines; FFT is kept as a fallback.
    """
    from scipy.signal import find_peaks
    prom = max(1e-6, 0.5 * (signal.max() - signal.min()))
    peaks, _ = find_peaks(signal, prominence=prom)
    if len(peaks) >= 2:
        return float(np.mean(np.diff(peaks)))
    spectrum = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs = np.fft.rfftfreq(len(signal))
    dom = int(np.argmax(spectrum[1:])) + 1
    return float(1.0 / freqs[dom])


def _normalise_sources(sources):
    """Expand legacy 3-tuple sources to the canonical 4-tuple form."""
    out = []
    for entry in sources:
        if len(entry) == 3:
            label, path, idx = entry
            out.append((label, path, idx, "peak"))
        else:
            out.append(tuple(entry))
    return tuple(out)


def _load_samples(time_steps: int, sources=DEFAULT_SOURCES):
    """Load and cache reference signals truncated to ``time_steps``.

    Returns a tuple ``(samples, periods, labels, modes)`` where ``samples``
    has shape ``(K, 1, 1, T)``, ``periods`` is a list of ``K`` floats,
    ``labels`` is a list of ``K`` short identifiers and ``modes`` selects
    which reference/prototype strategy to use per row.
    """
    sources = _normalise_sources(sources)
    key = (time_steps, sources)
    if key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[key]

    # missing files → return Nones so the tracker silently skips
    for _, path, _, _ in sources:
        if not os.path.exists(path):
            _SAMPLE_CACHE[key] = (None, None, None, None)
            return _SAMPLE_CACHE[key]

    raw_cache = {}
    tensors, periods, labels, modes = [], [], [], []
    for label, path, idx, mode in sources:
        if path not in raw_cache:
            raw_cache[path] = torch.load(path, map_location="cpu", weights_only=False)
        _, s = raw_cache[path][idx]
        s = s.float()[..., :time_steps]  # (1, 1, T)
        tensors.append(s)
        periods.append(_estimate_period(s[0, 0].numpy()))
        labels.append(label)
        modes.append(mode)

    _SAMPLE_CACHE[key] = (torch.stack(tensors, dim=0), periods, labels, modes)
    return _SAMPLE_CACHE[key]


def _patch_means(signal: np.ndarray, patch_width: int) -> np.ndarray:
    n_patches = len(signal) // patch_width
    return np.array([
        signal[i * patch_width : (i + 1) * patch_width].mean()
        for i in range(n_patches)
    ])


def _patch_stds(signal: np.ndarray, patch_width: int) -> np.ndarray:
    n_patches = len(signal) // patch_width
    return np.array([
        signal[i * patch_width : (i + 1) * patch_width].std()
        for i in range(n_patches)
    ])


def _draw_signal_with_patch_colors(ax, signal, sims, patch_width, cmap, norm,
                                   highlight_idx=None):
    ax.plot(np.arange(len(signal)), signal, color="black", linewidth=0.6, alpha=0.7)
    n_patches = len(signal) // patch_width
    for i in range(n_patches):
        start = i * patch_width
        ax.axvspan(start, start + patch_width, alpha=0.55,
                   color=cmap(norm(sims[i])), linewidth=0)
    if highlight_idx is not None:
        ax.axvspan(highlight_idx * patch_width, (highlight_idx + 1) * patch_width,
                   facecolor="none", edgecolor="black", linewidth=2.0)
    ax.set_xlim(0, len(signal))
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Amplitude")


@torch.no_grad()
def plot_sine_patch_similarity(model, *, time_steps: int = 2400,
                               patch_width: int = 24,
                               sources=DEFAULT_SOURCES,
                               device=None,
                               sine_path: str = "",
                               sine_mod_path: str = ""):
    """Build a Kx2 figure of patch cosine similarity for K reference signals.

    ``sine_path`` / ``sine_mod_path`` are the files of the "sine" / "sine_mod"
    entries in ``sources`` that carry no path of their own.

    Returns a ``wandb.Image`` (or ``None`` if any source file is missing).
    The model is temporarily switched to eval mode and restored afterwards.
    """
    if sine_path or sine_mod_path:
        paths = {"sine": sine_path, "sine_mod": sine_mod_path}
        sources = tuple((label, path or paths.get(label, ""), idx, mode)
                        for label, path, idx, mode in _normalise_sources(sources))
    samples, periods, labels, modes = _load_samples(time_steps, sources)
    if samples is None:
        return None

    if device is None:
        device = next(model.parameters()).device
    samples_dev = samples.to(device, non_blocking=True)

    R = getattr(model, "num_register_tokens", 0)
    S = 1 + R
    was_training = model.training
    model.eval()
    try:
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            emb = model.forward_encoder_all_patches(samples_dev)  # (K, 1+R+N, D)
        # drop CLS and register tokens, L2-normalise patches
        patch_tokens = F.normalize(emb[:, S:, :].float(), dim=-1).cpu()  # (K, N, D)
    finally:
        if was_training:
            model.train()

    n_rows = len(samples)
    n_patches = time_steps // patch_width
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)

    plt.close("all")
    fig, axes = plt.subplots(n_rows, 2, figsize=(15, 4.0 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for row in range(n_rows):
        label = labels[row]
        mode = modes[row]
        signal = samples[row, 0, 0].numpy()
        period = periods[row]
        n_cycles = max(1, int(round(time_steps / period)))
        patches = patch_tokens[row]  # (N, D)
        amps = _patch_means(signal, patch_width)

        if mode == "peak":
            # Frequency-adjusted peak set — one peak patch per cycle,
            # regardless of period. Non-peak patches form the "other" set
            # used for the peak↔other contrast metric.
            n_peak = min(n_cycles, n_patches - 1)
            peak_pct = 100.0 * (1.0 - n_peak / n_patches)
            peak_idx = np.where(amps >= np.percentile(amps, peak_pct))[0]
            if len(peak_idx) == 0:
                peak_idx = np.array([int(np.argmax(amps))])
            other_idx = np.array(
                [i for i in range(n_patches) if i not in set(peak_idx.tolist())]
            )

            # -------- Subplot 1: reference peak patch (DINOv3-style) --------
            ref_idx = int(np.argmax(amps))
            ref = patches[ref_idx : ref_idx + 1]  # (1, D)
            sim_to_ref = (patches @ ref.T).squeeze(-1).numpy()
            p2p_ref = float(sim_to_ref[peak_idx].mean())
            p2o_ref = float(sim_to_ref[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 0]
            _draw_signal_with_patch_colors(ax, signal, sim_to_ref, patch_width,
                                           cmap, norm, highlight_idx=ref_idx)
            ax.set_title(
                f"{label} period {period:.0f} ({n_cycles} cycles): "
                f"cos-sim to reference peak patch (idx={ref_idx})\n"
                f"avg peak→peak={p2p_ref:.3f}, avg peak→other={p2o_ref:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

            # -------- Subplot 2: mean peak prototype ------------------------
            proto = F.normalize(patches[peak_idx].mean(dim=0, keepdim=True), dim=-1)
            sim_to_proto = (patches @ proto.T).squeeze(-1).numpy()
            p2p_proto = float(sim_to_proto[peak_idx].mean())
            p2o_proto = float(sim_to_proto[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 1]
            _draw_signal_with_patch_colors(ax, signal, sim_to_proto, patch_width,
                                           cmap, norm)
            ax.set_title(
                f"{label} period {period:.0f}: cos-sim to mean peak proto "
                f"(n_peak={len(peak_idx)}, pct={peak_pct:.0f})\n"
                f"avg peak→peak={p2p_proto:.3f}, avg peak→other={p2o_proto:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

        elif mode == "flat":
            # A "flat" patch has ~zero intra-patch std (e.g. a flattened bump
            # in the sine_mod generator). Pick the first such patch as the
            # reference; its prototype averages every other flat patch whose
            # raw value matches the reference's (within a tight tolerance).
            stds = _patch_stds(signal, patch_width)
            std_tol = 1e-3
            flat_all = np.where(stds < std_tol)[0]
            if len(flat_all) == 0:
                # Fall back to the flattest patch we can find.
                flat_all = np.array([int(np.argmin(stds))])

            ref_idx = int(flat_all[0])
            ref_val = float(amps[ref_idx])
            # Patches that share the same flat value (same sign & magnitude).
            val_tol = max(1e-3, 0.01 * abs(ref_val))
            match_mask = (stds < std_tol) & (np.abs(amps - ref_val) < val_tol)
            match_idx = np.where(match_mask)[0]
            if len(match_idx) == 0:
                match_idx = np.array([ref_idx])
            other_idx = np.array(
                [i for i in range(n_patches) if i not in set(match_idx.tolist())]
            )

            # -------- Subplot 1: reference flat patch -----------------------
            ref = patches[ref_idx : ref_idx + 1]  # (1, D)
            sim_to_ref = (patches @ ref.T).squeeze(-1).numpy()
            f2f_ref = float(sim_to_ref[match_idx].mean())
            f2o_ref = float(sim_to_ref[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 0]
            _draw_signal_with_patch_colors(ax, signal, sim_to_ref, patch_width,
                                           cmap, norm, highlight_idx=ref_idx)
            ax.set_title(
                f"{label} period {period:.0f} ({n_cycles} cycles): "
                f"cos-sim to reference flat patch (idx={ref_idx}, val={ref_val:.2f})\n"
                f"avg flat→flat={f2f_ref:.3f}, avg flat→other={f2o_ref:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

            # -------- Subplot 2: mean flat prototype ------------------------
            proto = F.normalize(patches[match_idx].mean(dim=0, keepdim=True), dim=-1)
            sim_to_proto = (patches @ proto.T).squeeze(-1).numpy()
            f2f_proto = float(sim_to_proto[match_idx].mean())
            f2o_proto = float(sim_to_proto[other_idx].mean()) if len(other_idx) else float("nan")

            ax = axes[row, 1]
            _draw_signal_with_patch_colors(ax, signal, sim_to_proto, patch_width,
                                           cmap, norm)
            ax.set_title(
                f"{label} period {period:.0f}: cos-sim to mean flat proto "
                f"(n_flat={len(match_idx)}, val={ref_val:.2f})\n"
                f"avg flat→flat={f2f_proto:.3f}, avg flat→other={f2o_proto:.3f}"
            )
            plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm),
                         ax=ax, label="cos sim", shrink=0.85)

        else:
            raise ValueError(f"Unknown sine_patch_sim mode: {mode!r}")

    plt.tight_layout()
    img = wandb.Image(plt)
    plt.close(fig)
    return img
