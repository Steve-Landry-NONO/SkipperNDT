"""
src/models/dataset_regression.py
Dataset PyTorch pour la Tâche 2 : régression map_width.

Différences vs MagneticMapDataset (classification) :
    - Labels = float (log1p(width_m)) au lieu d'int
    - Même preprocessing image (resize 128×128, zscore par canal, NaN→0)
    - Même augmentation géométrique (flip H/V, rotation 90°)

Auteur(s) : MAKAMTA Linda, KENGNI Theophane, TUEKAM Ludovic, KOUOKAM NONO Steve Landry
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List

from src.preprocessing.loader import load_npz
from src.models.dataset import resize_array, normalize_channels


class RegressionDataset(Dataset):
    """
    Dataset pour la régression map_width.

    Retourne (image_tensor, log1p_width) à l'entraînement.
    image_tensor : (4, 128, 128) float32
    log1p_width  : float32 scalaire — log1p(width_m)

    Pour récupérer les mètres depuis la prédiction :
        width_m = torch.expm1(prediction)
    """

    def __init__(
        self,
        paths:   List[Path],
        widths:  List[float],   # largeurs en mètres
        augment: bool = False,
        size:    int  = 128,
    ):
        self.paths   = paths
        # Entraînement en log-space : distribution plus symétrique
        self.targets = [float(np.log1p(w)) for w in widths]
        self.augment = augment
        self.size    = size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        arr = load_npz(self.paths[idx])          # (H, W, 4)
        arr = resize_array(arr, size=self.size)   # (128, 128, 4)
        arr = normalize_channels(arr)             # zscore par canal, NaN→0

        if self.augment:
            # Flip horizontal
            if np.random.rand() > 0.5:
                arr = arr[:, ::-1, :].copy()
            # Flip vertical
            if np.random.rand() > 0.5:
                arr = arr[::-1, :, :].copy()
            # Rotation 90°
            k = np.random.randint(0, 4)
            if k > 0:
                arr = np.rot90(arr, k=k, axes=(0, 1)).copy()

        tensor = torch.from_numpy(arr.transpose(2, 0, 1))  # (4, 128, 128)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        return tensor, target

    def width_stats(self):
        """Retourne (mean_log, std_log) des targets pour info."""
        t = np.array(self.targets)
        return float(t.mean()), float(t.std())
