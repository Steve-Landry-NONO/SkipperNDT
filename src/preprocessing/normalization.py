"""
src/preprocessing/normalization.py
Stratégies de normalisation des cartes magnétiques.

Pourquoi normalisation individuelle ?
  Chaque image provient d'un terrain/courant/profondeur différents.
  Les ordres de grandeur absolus varient de ~1 nT (synth no-pipe) à ~2000 nT (synth pipe).
  Une normalisation globale effacerait cette variabilité structurelle.
"""

import numpy as np
from typing import Literal


NormStrategy = Literal["zscore", "minmax", "robust", "none"]


def normalize_image(
    arr: np.ndarray,
    strategy: NormStrategy = "zscore",
    per_channel: bool = True,
) -> np.ndarray:
    """
    Normalise une image magnétique (H, W, 4) individuellement.

    Args:
        arr       : array (H, W, 4) float32 — peut contenir des NaN
        strategy  : 'zscore' | 'minmax' | 'robust' | 'none'
        per_channel : si True, normalise canal par canal (recommandé)

    Returns:
        array normalisé (H, W, 4) float32, NaN préservés
    """
    out = arr.copy()
    channels = range(arr.shape[-1]) if per_channel else [None]

    for c in channels:
        if c is not None:
            ch = out[:, :, c]
        else:
            ch = out

        valid = ch[np.isfinite(ch)]
        if len(valid) == 0:
            continue

        if strategy == "zscore":
            mu, sigma = valid.mean(), valid.std()
            if sigma < 1e-8:
                sigma = 1.0
            ch_norm = (ch - mu) / sigma

        elif strategy == "minmax":
            vmin, vmax = valid.min(), valid.max()
            rng = vmax - vmin if (vmax - vmin) > 1e-8 else 1.0
            ch_norm = (ch - vmin) / rng

        elif strategy == "robust":
            # Robuste aux outliers : utilise médiane + IQR
            med = np.nanmedian(ch)
            q25, q75 = np.nanpercentile(ch, 25), np.nanpercentile(ch, 75)
            iqr = q75 - q25 if (q75 - q25) > 1e-8 else 1.0
            ch_norm = (ch - med) / iqr

        elif strategy == "none":
            ch_norm = ch
        else:
            raise ValueError(f"Stratégie inconnue: {strategy}")

        if c is not None:
            out[:, :, c] = ch_norm
        else:
            out[:] = ch_norm

    return out


def apply_abs_offset(arr: np.ndarray) -> np.ndarray:
    """
    Astuce domain adaptation : prend la valeur absolue de chaque canal
    (sauf Norm qui est déjà positif) pour réduire le décalage DC entre
    données synthétiques (centrées sur 0) et réelles (offset positif).
    À combiner avec une normalisation ensuite.
    """
    out = arr.copy()
    for c in range(3):  # Bx, By, Bz seulement
        out[:, :, c] = np.abs(out[:, :, c])
    return out
