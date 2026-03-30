"""
src/utils/visualization.py

Fonctions de visualisation pour l'EDA et l'analyse des cartes magnetiques.

Auteur(s) : MAKAMTA Linda, GILHAS Radia
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from typing import Optional, Dict
import warnings

warnings.filterwarnings("ignore")

CHANNEL_NAMES = ["Bx", "By", "Bz", "Norm"]
CHANNEL_CMAPS = ["RdBu_r", "RdBu_r", "RdBu_r", "viridis"]

STYLE = {
    ("real",  "pipe"):    {"color": "#e63946", "ls": "-",  "lw": 2,   "marker": "o", "label": "Reel — PIPE"},
    ("real",  "no_pipe"): {"color": "#457b9d", "ls": "-",  "lw": 2,   "marker": "o", "label": "Reel — NO-PIPE"},
    ("synth", "pipe"):    {"color": "#ff8c69", "ls": "--", "lw": 1.5, "marker": "s", "label": "Synth — PIPE"},
    ("synth", "no_pipe"): {"color": "#a8dadc", "ls": "--", "lw": 1.5, "marker": "s", "label": "Synth — NO-PIPE"},
}


def plot_channels(arr, title="", patch_size=300, save_path=None):
    h, w   = arr.shape[:2]
    r0, r1 = max(0, h//2-patch_size//2), min(h, h//2+patch_size//2)
    c0, c1 = max(0, w//2-patch_size//2), min(w, w//2+patch_size//2)
    patch  = arr[r0:r1, c0:c1, :]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title or "Canaux magnetiques (Bx, By, Bz, Norm)", fontsize=13, fontweight="bold")
    for col, (ch_name, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_CMAPS)):
        ax = axes[col]; ch = patch[:, :, col]
        im = ax.imshow(ch, cmap=cmap, vmin=np.nanpercentile(ch,2), vmax=np.nanpercentile(ch,98), aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{ch_name}", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_profile_1d(arr, row=None, title="", pixel_size_m=0.2, save_path=None):
    h, w = arr.shape[:2]
    if row is None: row = h // 2
    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    fig.suptitle(title or f"Profil 1D transversal (ligne {row}/{h})", fontsize=12, fontweight="bold")
    for col, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[col]; profile = arr[row, :, col]; valid = np.isfinite(profile)
        x  = np.arange(len(profile)) * pixel_size_m
        ax.plot(x[valid], profile[valid], color="#2a9d8f", lw=1.5)
        ax.fill_between(x[valid], 0, profile[valid], alpha=0.15, color="#2a9d8f")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(ch_name, fontsize=10, fontweight="bold"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_distribution(arr, title="", n_bins=80, max_samples=50000, save_path=None):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title or "Distributions des canaux magnetiques", fontsize=12, fontweight="bold")
    results = {}
    for col, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[col]; ch = arr[:, :, col].flatten(); valid = ch[np.isfinite(ch)]
        if len(valid) > max_samples: valid = np.random.choice(valid, max_samples, replace=False)
        mu_raw, sigma_raw = valid.mean(), valid.std()
        valid_n = (valid - mu_raw) / (sigma_raw + 1e-8)
        ax.hist(valid_n, bins=n_bins, density=True, color="#457b9d", alpha=0.7)
        mu_f, sg_f = stats.norm.fit(valid_n)
        x = np.linspace(valid_n.min(), valid_n.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu_f, sg_f), "r-", lw=2)
        sk = stats.skew(valid_n); ku = stats.kurtosis(valid_n)
        ks_stat, ks_p = stats.kstest(valid_n, "norm", args=(mu_f, sg_f))
        ax.set_title(f"{ch_name}\nskew={sk:.2f} kurt={ku:.2f} | {'approx Gauss' if ks_p>0.05 else 'Non-Gauss'}", fontsize=8)
        ax.grid(True, alpha=0.3)
        results[ch_name] = {"mu_raw": mu_raw, "sigma_raw": sigma_raw, "skewness": sk, "kurtosis": ku, "ks_p": ks_p}
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return results


def plot_distribution_comparison(datasets, channel="Bz", max_samples=30000, save_path=None):
    ch_idx = CHANNEL_NAMES.index(channel)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Domain Gap — Canal {channel} (normalise par image)", fontsize=12, fontweight="bold")
    for name, info in datasets.items():
        style   = STYLE.get((info.get("origin","synth"), info.get("label","no_pipe")), {"color":"gray","label":name})
        ch      = info["data"][:,:,ch_idx].flatten(); valid = ch[np.isfinite(ch)]
        if len(valid) > max_samples: valid = np.random.choice(valid, max_samples, replace=False)
        valid_n = (valid - valid.mean()) / (valid.std() + 1e-8)
        ax.hist(valid_n, bins=60, density=True, color=style["color"], alpha=0.4, label=style["label"])
    ax.set_xlabel(f"{channel} normalise", fontsize=10); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


def plot_correlations_and_nan(arr, title="", patch_size=400, save_path=None):
    h, w = arr.shape[:2]
    r0, r1 = max(0,h//2-patch_size//2), min(h,h//2+patch_size//2)
    c0, c1 = max(0,w//2-patch_size//2), min(w,w//2+patch_size//2)
    patch  = arr[r0:r1, c0:c1, :]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title or "Correlations inter-canaux + Masque NaN", fontsize=12, fontweight="bold")
    flat        = patch.reshape(-1,4); valid_rows = np.all(np.isfinite(flat), axis=1)
    flat_valid  = flat[valid_rows]
    if len(flat_valid) > 100:
        corr = np.corrcoef(flat_valid.T)
        im   = ax1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax1.set_xticks(range(4)); ax1.set_xticklabels(CHANNEL_NAMES)
        ax1.set_yticks(range(4)); ax1.set_yticklabels(CHANNEL_NAMES)
        for i in range(4):
            for j in range(4):
                ax1.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                         color="white" if abs(corr[i,j])>0.5 else "black")
        plt.colorbar(im, ax=ax1, fraction=0.046)
    ax1.set_title(f"Correlations ({valid_rows.sum()} pixels valides)", fontsize=10)
    nan_mask = np.isnan(patch[:,:,0]).astype(float)
    ax2.imshow(nan_mask, cmap="Reds", aspect="auto")
    ax2.set_title(f"Masque NaN — {nan_mask.mean()*100:.1f}% hors trajectoire", fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
