"""
src/preprocessing/features.py

Extraction de features statistiques pour la baseline ML (Tache 1 — pipe_present).

39 features par image : 8 statistiques sur 4 canaux + 6 correlations inter-canaux
+ 1 pourcentage de NaN global.

Features par canal (apres normalisation zscore individuelle) :
    mean, std, skewness, kurtosis, percentile 5, percentile 95,
    range P95-P5, fraction de pixels a plus de 2 ecarts-types

La skewness et la kurtosis sont les features les plus discriminantes d'apres l'EDA.
Les correlations inter-canaux capturent la coherence du signal dipole caracteristique
des conduites.

Auteur(s) : TUEKAM Ludovic
"""

import numpy as np
from pathlib import Path
from typing import Optional


FEATURE_NAMES = []
CHANNEL_NAMES = ["Bx", "By", "Bz", "Norm"]

_STAT_NAMES = ["mean", "std", "skew", "kurt", "p5", "p95", "range", "peak_pct"]
for ch in CHANNEL_NAMES:
    for s in _STAT_NAMES:
        FEATURE_NAMES.append(f"{ch}_{s}")

_CHAN_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
for i, j in _CHAN_PAIRS:
    FEATURE_NAMES.append(f"corr_{CHANNEL_NAMES[i]}_{CHANNEL_NAMES[j]}")

FEATURE_NAMES.append("nan_pct")

N_FEATURES = len(FEATURE_NAMES)  # 39


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    Extrait 39 features d'une carte magnetique (H, W, 4).

    Chaque canal est normalise individuellement (zscore sur les pixels valides)
    avant le calcul des statistiques. Les canaux sans pixels valides donnent
    des zeros pour les 8 features correspondantes.

    Args:
        arr : array float32 (H, W, 4), peut contenir des NaN

    Returns:
        vecteur float32 (39,)
    """
    from scipy import stats as sp_stats

    feats = []

    channels_valid = []
    for c in range(4):
        ch    = arr[:, :, c].flatten()
        valid = ch[np.isfinite(ch)]

        if len(valid) < 10:
            feats.extend([0.0] * 8)
            channels_valid.append(np.zeros(10))
            continue

        mu, sigma = valid.mean(), valid.std()
        if sigma < 1e-8:
            sigma = 1.0
        ch_n = (valid - mu) / sigma
        channels_valid.append(ch_n)

        p5  = float(np.percentile(ch_n, 5))
        p95 = float(np.percentile(ch_n, 95))

        feats.extend([
            float(ch_n.mean()),
            float(ch_n.std()),
            float(sp_stats.skew(ch_n)),
            float(sp_stats.kurtosis(ch_n)),
            p5,
            p95,
            p95 - p5,
            float(np.mean(np.abs(ch_n) > 2)),
        ])

    for i, j in _CHAN_PAIRS:
        ci = channels_valid[i]
        cj = channels_valid[j]
        n  = min(len(ci), len(cj))
        if n < 10:
            feats.append(0.0)
            continue
        if n > 50000:
            idx = np.random.choice(n, 50000, replace=False)
            ci, cj = ci[idx], cj[idx]
        corr = float(np.corrcoef(
            ci[:min(len(ci), len(cj))],
            cj[:min(len(ci), len(cj))],
        )[0, 1])
        feats.append(0.0 if np.isnan(corr) else corr)

    feats.append(float(np.isnan(arr).mean()))

    return np.array(feats, dtype=np.float32)


def extract_features_batch(
    paths: list,
    labels: list,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait les features pour une liste de fichiers .npz.

    Les fichiers illisibles sont ignores silencieusement (un avertissement
    est affiche si verbose=True).

    Args:
        paths   : liste de Path vers les fichiers .npz
        labels  : labels correspondants
        verbose : affiche la progression

    Returns:
        X : (N, 39) float32
        y : (N,) int64
    """
    from src.preprocessing.loader import load_npz

    X, y_out = [], []
    n = len(paths)

    for i, (path, label) in enumerate(zip(paths, labels)):
        if verbose and (i % 50 == 0 or i == n - 1):
            print(f"  [{i+1:4d}/{n}] {Path(path).stem[:50]}", flush=True)
        try:
            arr   = load_npz(path)
            feats = extract_features(arr)
            X.append(feats)
            y_out.append(label)
        except Exception as e:
            print(f"  erreur sur {path}: {e}")
            continue

    return np.array(X, dtype=np.float32), np.array(y_out, dtype=np.int64)
