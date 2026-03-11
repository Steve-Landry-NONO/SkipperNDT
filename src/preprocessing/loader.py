"""
src/preprocessing/loader.py
Chargement et inspection des fichiers .npz / .tif (4 canaux magnétiques).
Canaux : [0]=Bx, [1]=By, [2]=Bz, [3]=Norm  — unité : nanoTesla (nT)
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict


CHANNEL_NAMES = ["Bx", "By", "Bz", "Norm"]


def load_npz(path: str | Path) -> np.ndarray:
    """
    Charge un fichier .npz et retourne l'array (H, W, 4) en float32.
    Les NaN représentent les zones hors trajectoire du drone.
    """
    data = np.load(path)
    # La clé est toujours 'data' dans ce dataset
    arr = data["data"].astype(np.float32)
    return arr


def get_valid_mask(arr: np.ndarray) -> np.ndarray:
    """
    Retourne un masque booléen (H, W) — True là où tous les canaux sont valides (non-NaN).
    """
    return np.all(np.isfinite(arr), axis=-1)


def image_stats(arr: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calcule des statistiques descriptives par canal sur les pixels valides.
    Retourne un dict { 'Bx': {'mean': ..., 'std': ..., ...}, ... }
    """
    stats = {}
    for i, name in enumerate(CHANNEL_NAMES):
        ch = arr[:, :, i].flatten()
        valid = ch[np.isfinite(ch)]
        if len(valid) == 0:
            stats[name] = {k: float("nan") for k in ["mean", "std", "min", "max", "p5", "p95", "nan_pct"]}
            continue
        stats[name] = {
            "mean":    float(valid.mean()),
            "std":     float(valid.std()),
            "min":     float(valid.min()),
            "max":     float(valid.max()),
            "p5":      float(np.percentile(valid, 5)),
            "p95":     float(np.percentile(valid, 95)),
            "nan_pct": float(np.isnan(ch).mean() * 100),
        }
    return stats


def print_summary(path: str | Path) -> None:
    """Affiche un résumé lisible d'un fichier .npz."""
    arr = load_npz(path)
    h, w, c = arr.shape
    valid_mask = get_valid_mask(arr)
    pct_valid = valid_mask.mean() * 100
    print(f"Fichier  : {Path(path).name}")
    print(f"Forme    : {h} x {w} x {c}  ({h*0.2:.0f}m x {w*0.2:.0f}m)")
    print(f"Dtype    : {arr.dtype}")
    print(f"Pixels valides : {pct_valid:.1f}%")
    print(f"{'Canal':<6} {'mean':>10} {'std':>10} {'min':>10} {'max':>10} {'p5':>10} {'p95':>10} {'NaN%':>7}")
    print("-" * 75)
    for name, s in image_stats(arr).items():
        print(f"{name:<6} {s['mean']:>10.2f} {s['std']:>10.2f} {s['min']:>10.2f} "
              f"{s['max']:>10.2f} {s['p5']:>10.2f} {s['p95']:>10.2f} {s['nan_pct']:>6.1f}%")
