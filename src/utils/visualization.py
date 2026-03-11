"""
src/utils/visualization.py
Fonctions de visualisation pour l'EDA et l'analyse des cartes magnétiques.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from scipy import stats
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import warnings

warnings.filterwarnings("ignore")

CHANNEL_NAMES = ["Bx", "By", "Bz", "Norm"]
CHANNEL_CMAPS = ["RdBu_r", "RdBu_r", "RdBu_r", "viridis"]

STYLE = {
    ("real", "pipe"):     {"color": "#e63946", "ls": "-",  "lw": 2,   "marker": "o", "label": "Réel — PIPE"},
    ("real", "no_pipe"):  {"color": "#457b9d", "ls": "-",  "lw": 2,   "marker": "o", "label": "Réel — NO-PIPE"},
    ("synth", "pipe"):    {"color": "#ff8c69", "ls": "--", "lw": 1.5, "marker": "s", "label": "Synth — PIPE"},
    ("synth", "no_pipe"): {"color": "#a8dadc", "ls": "--", "lw": 1.5, "marker": "s", "label": "Synth — NO-PIPE"},
}


# ─────────────────────────────────────────────
# 1. Visualisation des 4 canaux d'une image
# ─────────────────────────────────────────────
def plot_channels(
    arr: np.ndarray,
    title: str = "",
    patch_size: int = 300,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Affiche les 4 canaux (Bx, By, Bz, Norm) d'une image magnétique.
    Coupe un patch central si l'image est très grande.
    """
    h, w = arr.shape[:2]
    r0 = max(0, h // 2 - patch_size // 2)
    r1 = min(h, r0 + patch_size)
    c0 = max(0, w // 2 - patch_size // 2)
    c1 = min(w, c0 + patch_size)
    patch = arr[r0:r1, c0:c1, :]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title or "Canaux magnétiques (Bx, By, Bz, Norm)", fontsize=13, fontweight="bold")

    for col, (ch_name, cmap) in enumerate(zip(CHANNEL_NAMES, CHANNEL_CMAPS)):
        ax = axes[col]
        ch = patch[:, :, col]
        vmin = np.nanpercentile(ch, 2)
        vmax = np.nanpercentile(ch, 98)
        im = ax.imshow(ch, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{ch_name}\n[{vmin:.1f}, {vmax:.1f}] nT", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 2. Profil 1D transversal
# ─────────────────────────────────────────────
def plot_profile_1d(
    arr: np.ndarray,
    row: Optional[int] = None,
    title: str = "",
    pixel_size_m: float = 0.2,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Trace le profil 1D (coupe transversale) au rang `row` (défaut : milieu).
    Révèle la signature dipôle caractéristique d'un pipe.
    """
    h, w = arr.shape[:2]
    if row is None:
        row = h // 2

    fig, axes = plt.subplots(1, 4, figsize=(18, 3.5))
    fig.suptitle(
        title or f"Profil 1D transversal (ligne {row} / {h})",
        fontsize=12, fontweight="bold"
    )

    for col, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[col]
        profile = arr[row, :, col]
        valid = np.isfinite(profile)
        x = np.arange(len(profile)) * pixel_size_m

        ax.plot(x[valid], profile[valid], color="#2a9d8f", lw=1.5)
        ax.fill_between(x[valid], 0, profile[valid], alpha=0.15, color="#2a9d8f")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(ch_name, fontsize=10, fontweight="bold")
        ax.set_xlabel("Position (m)", fontsize=9)
        ax.set_ylabel("nT", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 3. Distribution + test KS + fit gaussien
# ─────────────────────────────────────────────
def plot_distribution(
    arr: np.ndarray,
    title: str = "",
    n_bins: int = 80,
    max_samples: int = 50000,
    save_path: Optional[str] = None,
) -> Dict:
    """
    Histogramme + fit gaussien + test KS pour chaque canal.
    Retourne les résultats statistiques.
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(title or "Distributions des canaux magnétiques", fontsize=12, fontweight="bold")

    results = {}
    for col, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[col]
        ch = arr[:, :, col].flatten()
        valid = ch[np.isfinite(ch)]
        if len(valid) > max_samples:
            valid = np.random.choice(valid, max_samples, replace=False)

        # Normalisation individuelle
        mu_raw, sigma_raw = valid.mean(), valid.std()
        valid_norm = (valid - mu_raw) / (sigma_raw + 1e-8)

        ax.hist(valid_norm, bins=n_bins, density=True, color="#457b9d", alpha=0.7, label="Données")

        # Fit gaussien
        mu_fit, sigma_fit = stats.norm.fit(valid_norm)
        x = np.linspace(valid_norm.min(), valid_norm.max(), 300)
        ax.plot(x, stats.norm.pdf(x, mu_fit, sigma_fit), "r-", lw=2, label="Gauss fit")

        # Stats
        skewness = stats.skew(valid_norm)
        kurtosis = stats.kurtosis(valid_norm)
        ks_stat, ks_p = stats.kstest(valid_norm, "norm", args=(mu_fit, sigma_fit))
        verdict = "≈ Gauss" if ks_p > 0.05 else "Non-Gauss"

        ax.set_title(
            f"{ch_name} (normalisé)\nμ={mu_fit:.2f} σ={sigma_fit:.2f}\n"
            f"skew={skewness:.2f} kurt={kurtosis:.2f} | {verdict}",
            fontsize=8,
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Valeur normalisée", fontsize=8)

        results[ch_name] = {
            "mu_raw": mu_raw, "sigma_raw": sigma_raw,
            "skewness": skewness, "kurtosis": kurtosis,
            "ks_stat": ks_stat, "ks_p": ks_p, "gaussian": ks_p > 0.05,
        }

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return results


# ─────────────────────────────────────────────
# 4. Comparaison multi-images (overlay distributions)
# ─────────────────────────────────────────────
def plot_distribution_comparison(
    datasets: Dict[str, Dict],
    channel: str = "Bz",
    max_samples: int = 30000,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Compare les distributions d'un canal entre plusieurs datasets.

    Args:
        datasets : dict { 'nom': {'data': np.ndarray, 'origin': str, 'label': str} }
        channel  : 'Bx' | 'By' | 'Bz' | 'Norm'
    """
    ch_idx = CHANNEL_NAMES.index(channel)
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"Comparaison distributions — Canal {channel} (normalisé par image)", fontsize=12, fontweight="bold")

    for name, info in datasets.items():
        arr = info["data"]
        origin = info.get("origin", "synth")
        label = info.get("label", "no_pipe")
        style = STYLE.get((origin, label), {"color": "gray", "ls": "-", "lw": 1.5, "label": name})

        ch = arr[:, :, ch_idx].flatten()
        valid = ch[np.isfinite(ch)]
        if len(valid) > max_samples:
            valid = np.random.choice(valid, max_samples, replace=False)
        valid_norm = (valid - valid.mean()) / (valid.std() + 1e-8)

        ax.hist(valid_norm, bins=60, density=True, color=style["color"],
                alpha=0.4, label=f"{style['label']} ({name})")

    ax.set_xlabel(f"{channel} normalisé", fontsize=10)
    ax.set_ylabel("Densité", fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig


# ─────────────────────────────────────────────
# 5. Corrélations inter-canaux + masque NaN
# ─────────────────────────────────────────────
def plot_correlations_and_nan(
    arr: np.ndarray,
    title: str = "",
    patch_size: int = 400,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Affiche la matrice de corrélation inter-canaux et le masque NaN.
    """
    h, w = arr.shape[:2]
    r0 = max(0, h // 2 - patch_size // 2)
    r1 = min(h, r0 + patch_size)
    c0 = max(0, w // 2 - patch_size // 2)
    c1 = min(w, c0 + patch_size)
    patch = arr[r0:r1, c0:c1, :]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title or "Corrélations inter-canaux + Masque NaN", fontsize=12, fontweight="bold")

    # Corrélation
    flat = patch.reshape(-1, 4)
    valid_rows = np.all(np.isfinite(flat), axis=1)
    flat_valid = flat[valid_rows]

    if len(flat_valid) > 100:
        corr = np.corrcoef(flat_valid.T)
        im = ax1.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        ax1.set_xticks(range(4)); ax1.set_xticklabels(CHANNEL_NAMES)
        ax1.set_yticks(range(4)); ax1.set_yticklabels(CHANNEL_NAMES)
        for i in range(4):
            for j in range(4):
                ax1.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                         fontsize=11, fontweight="bold",
                         color="white" if abs(corr[i, j]) > 0.5 else "black")
        plt.colorbar(im, ax=ax1, fraction=0.046)
    ax1.set_title(f"Corrélations (sur {valid_rows.sum()} pixels valides)", fontsize=10)

    # Masque NaN
    nan_mask = np.isnan(patch[:, :, 0]).astype(float)
    ax2.imshow(nan_mask, cmap="Reds", aspect="auto")
    pct_nan = nan_mask.mean() * 100
    ax2.set_title(f"Masque NaN — {pct_nan:.1f}% hors trajectoire drone", fontsize=10)
    ax2.set_xticks([]); ax2.set_yticks([])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig
