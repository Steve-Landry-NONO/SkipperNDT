"""
src/preprocessing/features.py
Extraction de features statistiques pour la baseline ML (Tâche 1 — pipe_present).

Stratégie : 32 features × image (8 stats × 4 canaux), calculées sur les pixels valides
après normalisation individuelle par image (zscore).

Features par canal :
  mean, std, skewness, kurtosis,
  percentile 5, percentile 95, range P95-P5,
  % pixels > 2σ (indicateur de pic d'anomalie)

Pourquoi ces features ?
  - L'EDA a montré que la skewness et la kurtosis discriminent pipe/no_pipe
  - La corrélation inter-canaux est informative → on l'ajoute en bonus (+6 features = 38 total)
  - Normalisation individuelle obligatoire (ordres de grandeur variables)
"""

import numpy as np
from pathlib import Path
from typing import Optional


FEATURE_NAMES = []
CHANNEL_NAMES = ["Bx", "By", "Bz", "Norm"]

# 8 features × 4 canaux = 32
_STAT_NAMES = ["mean", "std", "skew", "kurt", "p5", "p95", "range", "peak_pct"]
for ch in CHANNEL_NAMES:
    for s in _STAT_NAMES:
        FEATURE_NAMES.append(f"{ch}_{s}")

# 6 corrélations inter-canaux (combinaisons 2 à 2)
_CHAN_PAIRS = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
for i, j in _CHAN_PAIRS:
    FEATURE_NAMES.append(f"corr_{CHANNEL_NAMES[i]}_{CHANNEL_NAMES[j]}")

# 1 feature globale : % NaN
FEATURE_NAMES.append("nan_pct")

N_FEATURES = len(FEATURE_NAMES)  # 39


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    Extrait 39 features d'une image magnétique (H, W, 4).

    Args:
        arr : array float32 (H, W, 4), peut contenir des NaN

    Returns:
        vecteur float32 (39,)
    """
    from scipy import stats as sp_stats

    feats = []

    # ── Statistiques par canal ────────────────────────────────
    channels_valid = []
    for c in range(4):
        ch = arr[:, :, c].flatten()
        valid = ch[np.isfinite(ch)]

        if len(valid) < 10:
            feats.extend([0.0] * 8)
            channels_valid.append(np.zeros(10))
            continue

        # Normalisation individuelle zscore
        mu, sigma = valid.mean(), valid.std()
        if sigma < 1e-8:
            sigma = 1.0
        ch_n = (valid - mu) / sigma
        channels_valid.append(ch_n)

        p5  = float(np.percentile(ch_n, 5))
        p95 = float(np.percentile(ch_n, 95))

        feats.extend([
            float(ch_n.mean()),                      # mean (≈0 après zscore)
            float(ch_n.std()),                       # std  (≈1 après zscore)
            float(sp_stats.skew(ch_n)),              # asymétrie
            float(sp_stats.kurtosis(ch_n)),          # aplatissement
            p5,                                      # percentile 5
            p95,                                     # percentile 95
            p95 - p5,                                # étendue interquantile
            float(np.mean(np.abs(ch_n) > 2)),        # % pixels > 2σ (pic)
        ])

    # ── Corrélations inter-canaux ─────────────────────────────
    for i, j in _CHAN_PAIRS:
        ci = channels_valid[i]
        cj = channels_valid[j]
        n = min(len(ci), len(cj))
        if n < 10:
            feats.append(0.0)
            continue
        # Sous-échantillonnage si trop long (corrélation coûteuse)
        if n > 50000:
            idx = np.random.choice(n, 50000, replace=False)
            ci, cj = ci[idx], cj[idx]
        corr = float(np.corrcoef(ci[:min(len(ci), len(cj))],
                                  cj[:min(len(ci), len(cj))])[0, 1])
        feats.append(0.0 if np.isnan(corr) else corr)

    # ── % NaN global ──────────────────────────────────────────
    feats.append(float(np.isnan(arr).mean()))

    return np.array(feats, dtype=np.float32)


def extract_features_batch(
    paths: list,
    labels: list,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extrait les features pour une liste de fichiers .npz.

    Args:
        paths  : liste de Path vers les fichiers .npz
        labels : liste d'entiers (labels correspondants)
        verbose: affiche la progression

    Returns:
        X : (N, 39) float32
        y : (N,) int
    """
    from src.preprocessing.loader import load_npz

    X, y_out = [], []
    n = len(paths)

    for i, (path, label) in enumerate(zip(paths, labels)):
        if verbose and (i % 50 == 0 or i == n - 1):
            print(f"  [{i+1:4d}/{n}] {Path(path).stem[:50]}", flush=True)
        try:
            arr = load_npz(path)
            feats = extract_features(arr)
            X.append(feats)
            y_out.append(label)
        except Exception as e:
            print(f"  [!] Erreur sur {path}: {e}")
            continue

    return np.array(X, dtype=np.float32), np.array(y_out, dtype=np.int64)
