"""
src/preprocessing/normalization.py

Strategies de normalisation des cartes magnetiques.

La normalisation individuelle par image est indispensable : les ordres de grandeur
absolus varient de ~1 nT (synthetique no-pipe) a ~2000 nT (pipe fort courant).
Une normalisation globale effacerait cette variabilite structurelle exploitee
par les modeles pour distinguer pipe/no-pipe.

Auteur(s) : KENGNI Theophane
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
    Normalise une carte magnetique (H, W, 4) individuellement.

    Args:
        arr         : array (H, W, 4) float32, peut contenir des NaN
        strategy    : zscore | minmax | robust | none
        per_channel : si True, normalise canal par canal (recommande)

    Returns:
        array normalise (H, W, 4) float32, NaN preserves
    """
    out      = arr.copy()
    channels = range(arr.shape[-1]) if per_channel else [None]

    for c in channels:
        ch    = out[:, :, c] if c is not None else out
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
            med      = np.nanmedian(ch)
            q25, q75 = np.nanpercentile(ch, 25), np.nanpercentile(ch, 75)
            iqr      = q75 - q25 if (q75 - q25) > 1e-8 else 1.0
            ch_norm  = (ch - med) / iqr

        elif strategy == "none":
            ch_norm = ch
        else:
            raise ValueError(f"strategie inconnue: {strategy}")

        if c is not None:
            out[:, :, c] = ch_norm
        else:
            out[:] = ch_norm

    return out


def apply_abs_offset(arr: np.ndarray) -> np.ndarray:
    """
    Prend la valeur absolue des canaux Bx, By, Bz (pas Norm) pour attenuer
    l'offset DC positif des donnees reelles par rapport aux synthetiques
    (centrees sur 0 nT). A combiner avec une normalisation zscore ensuite.
    """
    out = arr.copy()
    for c in range(3):
        out[:, :, c] = np.abs(out[:, :, c])
    return out
