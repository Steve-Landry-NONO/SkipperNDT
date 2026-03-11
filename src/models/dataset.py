"""
src/models/dataset.py
Dataset PyTorch pour les cartes magnétiques — Tâche 1 (pipe_present).

Stratégie de resize : Global Average Pooling spatial → patch fixe 128×128
  - Les images vont de 150×150 à 4000×3750 → impossible de passer tout en mémoire
  - On redimensionne à 128×128 (PIL BILINEAR, NaN ignorés)
  - Alternative pour grandes images : patch central + random crop en train

Normalisation : zscore individuelle par canal sur les pixels valides.
Les NaN sont remplacés par 0.0 après normalisation (valeur neutre).

Augmentation (train seulement) :
  - Flip horizontal / vertical aléatoire
  - Rotation 90° aléatoire
  - Pas de bruit ajouté ici (domain gap déjà important)
"""

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None
    Dataset = object
from pathlib import Path
from typing import Optional, Callable
import random


TARGET_SIZE = 128   # px — compromis mémoire / résolution


def resize_array(arr: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """
    Redimensionne une image (H, W, 4) à (size, size, 4) en ignorant les NaN.
    Utilise une interpolation bilinéaire via numpy (pas de dépendance PIL/cv2).
    """
    from scipy.ndimage import zoom

    h, w = arr.shape[:2]
    if h == size and w == size:
        return arr

    # Masquer les NaN temporairement
    out = np.zeros((size, size, 4), dtype=np.float32)
    for c in range(4):
        ch = arr[:, :, c].copy()
        nan_mask = np.isnan(ch)
        ch[nan_mask] = 0.0
        zoom_h = size / h
        zoom_w = size / w
        ch_resized = zoom(ch, (zoom_h, zoom_w), order=1)  # bilinear
        # Propager le masque NaN
        mask_resized = zoom(nan_mask.astype(np.float32), (zoom_h, zoom_w), order=0)
        ch_resized[mask_resized > 0.5] = np.nan
        out[:, :, c] = ch_resized

    return out


def normalize_channels(arr: np.ndarray) -> np.ndarray:
    """
    Normalisation zscore individuelle par canal. NaN → 0.0 après normalisation.
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(4):
        ch = arr[:, :, c]
        valid = ch[np.isfinite(ch)]
        if len(valid) < 10:
            out[:, :, c] = 0.0
            continue
        mu, sigma = valid.mean(), valid.std()
        if sigma < 1e-8:
            sigma = 1.0
        normalized = (ch - mu) / sigma
        normalized = np.where(np.isfinite(normalized), normalized, 0.0)
        out[:, :, c] = normalized
    return out


class MagneticMapDataset(Dataset):
    """
    Dataset PyTorch pour les cartes magnétiques 4 canaux.

    Args:
        paths     : liste de Path vers les fichiers .npz
        labels    : liste d'entiers (0=no_pipe, 1=pipe)
        augment   : si True, active les augmentations (train seulement)
        size      : taille cible après resize (défaut: 128)
        cache_size: nb d'images à garder en cache RAM (0 = pas de cache)
    """

    def __init__(
        self,
        paths:      list,
        labels:     list,
        augment:    bool = False,
        size:       int  = TARGET_SIZE,
        cache_size: int  = 0,
    ):
        self.paths      = [Path(p) for p in paths]
        self.labels     = labels
        self.augment    = augment
        self.size       = size
        self.cache_size = cache_size
        self._cache     = {}

        assert len(paths) == len(labels), "paths et labels doivent avoir la même longueur"

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, idx: int) -> np.ndarray:
        """Charge, redimensionne et normalise une image."""
        if idx in self._cache:
            return self._cache[idx]

        from src.preprocessing.loader import load_npz
        arr = load_npz(self.paths[idx])
        arr = resize_array(arr, self.size)
        arr = normalize_channels(arr)

        if self.cache_size > 0 and len(self._cache) < self.cache_size:
            self._cache[idx] = arr

        return arr

    def _augment(self, arr: np.ndarray) -> np.ndarray:
        """Augmentations géométriques (flip + rotation 90°)."""
        # Flip horizontal
        if random.random() > 0.5:
            arr = arr[:, ::-1, :].copy()
        # Flip vertical
        if random.random() > 0.5:
            arr = arr[::-1, :, :].copy()
        # Rotation 90° aléatoire
        k = random.randint(0, 3)
        if k > 0:
            arr = np.rot90(arr, k=k, axes=(0, 1)).copy()
        return arr

    def __getitem__(self, idx: int):
        arr = self._load(idx)

        if self.augment:
            arr = self._augment(arr)

        # (H, W, C) → (C, H, W) pour PyTorch
        import torch
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

    def class_weights(self):
        """
        Calcule les poids de classe inversement proportionnels à leur fréquence.
        Utile pour gérer le déséquilibre pipe/no_pipe.
        """
        counts = np.bincount(self.labels)
        total  = len(self.labels)
        weights = total / (len(counts) * counts.astype(np.float32))
        import torch
        return torch.tensor(weights, dtype=torch.float32)
