"""
src/models/dataset.py

Dataset PyTorch pour les cartes magnetiques 4 canaux — Tache 1 (pipe_present).

Les images varient de 150x150 a 4000x3750 pixels. Elles sont redimensionnees
a 128x128 (zoom scipy bilineaire) avant d'etre passees au reseau. Cette resolution
est un compromis entre cout memoire et preservation des structures dipole.

Normalisation : zscore individuelle par canal sur les pixels valides. Les NaN sont
remplaces par 0.0 apres normalisation — valeur neutre dans l'espace zscore.

Augmentation (train uniquement) : flip horizontal/vertical aleatoire + rotation 90.
Pas de bruit ajoute car le domain gap synth->reel est deja significatif.

Auteur(s) : MAKAMTA Linda, KENGNI Theophane, TUEKAM Ludovic, KOUOKAM NONO Steve Landry
"""

import numpy as np
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch   = None
    Dataset = object
from pathlib import Path
from typing import Optional
import random


TARGET_SIZE = 128


def resize_array(arr: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    """
    Redimensionne (H, W, 4) vers (size, size, 4) via interpolation bilineaire.

    Les NaN sont masques avant le zoom puis re-appliques apres pour ne pas
    propager de valeurs interpolees invalides.

    Args:
        arr  : array (H, W, 4) float32
        size : taille cible (carre)

    Returns:
        array (size, size, 4) float32
    """
    from scipy.ndimage import zoom

    h, w = arr.shape[:2]
    if h == size and w == size:
        return arr

    out = np.zeros((size, size, 4), dtype=np.float32)
    for c in range(4):
        ch       = arr[:, :, c].copy()
        nan_mask = np.isnan(ch)
        ch[nan_mask] = 0.0
        ch_resized   = zoom(ch, (size / h, size / w), order=1)
        mask_resized = zoom(nan_mask.astype(np.float32), (size / h, size / w), order=0)
        ch_resized[mask_resized > 0.5] = np.nan
        out[:, :, c] = ch_resized

    return out


def normalize_channels(arr: np.ndarray) -> np.ndarray:
    """
    Normalisation zscore individuelle par canal. Les NaN et les pixels nuls
    sont remplaces par 0.0 apres normalisation.

    Args:
        arr : array (H, W, 4) float32

    Returns:
        array normalise, meme shape
    """
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(4):
        ch    = arr[:, :, c]
        valid = ch[np.isfinite(ch)]
        if len(valid) < 10:
            out[:, :, c] = 0.0
            continue
        mu, sigma = valid.mean(), valid.std()
        if sigma < 1e-8:
            sigma = 1.0
        normalized = (ch - mu) / sigma
        out[:, :, c] = np.where(np.isfinite(normalized), normalized, 0.0)
    return out


class MagneticMapDataset(Dataset):
    """
    Dataset PyTorch pour les cartes magnetiques 4 canaux.

    Args:
        paths      : chemins vers les fichiers .npz
        labels     : 0 = no_pipe, 1 = pipe
        augment    : active les augmentations geometriques (train uniquement)
        size       : taille cible apres resize
        cache_size : nombre d'images a conserver en cache RAM (0 = desactive)
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

        assert len(paths) == len(labels)

    def __len__(self) -> int:
        return len(self.paths)

    def _load(self, idx: int) -> np.ndarray:
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
        if random.random() > 0.5:
            arr = arr[:, ::-1, :].copy()
        if random.random() > 0.5:
            arr = arr[::-1, :, :].copy()
        k = random.randint(0, 3)
        if k > 0:
            arr = np.rot90(arr, k=k, axes=(0, 1)).copy()
        return arr

    def __getitem__(self, idx: int):
        arr = self._load(idx)

        if self.augment:
            arr = self._augment(arr)

        import torch
        tensor = torch.from_numpy(arr.transpose(2, 0, 1))
        label  = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, label

    def class_weights(self):
        """
        Poids de classe inverses a la frequence — utile pour les datasets
        desequilibres (pipe/no_pipe dans T1, parallel/single dans T4).
        """
        counts  = np.bincount(self.labels)
        total   = len(self.labels)
        weights = total / (len(counts) * counts.astype(np.float32))
        import torch
        return torch.tensor(weights, dtype=torch.float32)
