"""
src/models/cnn_task2.py
Architecture CNN de régression pour la Tâche 2 : prédiction de map_width (en mètres).

Différences vs cnn_task1.py :
    - Tête de régression (sortie linéaire 1 neurone) au lieu de classification (2 neurones)
    - Entraînement en log-space (log1p) pour gérer la distribution asymétrique (2m→155m)
    - Huber loss (robuste aux grands écarts) au lieu de CrossEntropy

Entrée  : (B, 4, 128, 128) float32
Sortie  : (B, 1) — log1p(width_m) en entraînement, converti en mètres en inférence

Auteur(s) : MAKAMTA Linda, KENGNI Theophane, TUEKAM Ludovic, KOUOKAM NONO Steve Landry
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MagCNNRegressor(nn.Module):
    """
    CNN léger de régression — même backbone que MagCNN, tête linéaire.
    ~420K paramètres.
    """

    def __init__(self, dropout: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            # Bloc 1 : 4 → 32
            nn.Conv2d(4, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 128→64

            # Bloc 2 : 32 → 64
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 64→32

            # Bloc 3 : 64 → 128
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                              # 32→16

            # Bloc 4 : 128 → 256
            nn.Conv2d(128, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            # Pas de MaxPool ici → préserver les features spatiales fines
        )

        self.gap = nn.AdaptiveAvgPool2d(1)   # Global Average Pooling → (B, 256, 1, 1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 1),                # sortie : log1p(width_m)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = self.regressor(x)
        return x.squeeze(1)   # (B,)


class MagDenseNetRegressor(nn.Module):
    """
    DenseNet de régression — même backbone que MagDenseNet, tête linéaire.
    ~1.8M paramètres.
    """

    def __init__(self, growth_rate: int = 16, dropout: float = 0.3):
        super().__init__()
        k = growth_rate

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(4, 2 * k, 3, padding=1, bias=False),
            nn.BatchNorm2d(2 * k), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),   # 128→64
        )

        # Blocs denses
        self.dense1 = self._dense_block(2 * k,          n_layers=6, k=k)
        n1 = 2 * k + 6 * k
        self.trans1 = self._transition(n1, n1 // 2)    # 64→32

        self.dense2 = self._dense_block(n1 // 2,        n_layers=6, k=k)
        n2 = n1 // 2 + 6 * k
        self.trans2 = self._transition(n2, n2 // 2)    # 32→16

        self.dense3 = self._dense_block(n2 // 2,        n_layers=6, k=k)
        n3 = n2 // 2 + 6 * k

        self.bn_final = nn.BatchNorm2d(n3)
        self.gap      = nn.AdaptiveAvgPool2d(1)

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n3, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    @staticmethod
    def _dense_block(in_ch: int, n_layers: int, k: int) -> nn.ModuleList:
        layers = nn.ModuleList()
        ch = in_ch
        for _ in range(n_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
                nn.Conv2d(ch, k, 3, padding=1, bias=False),
            ))
            ch += k
        return layers

    @staticmethod
    def _transition(in_ch: int, out_ch: int) -> nn.Sequential:
        return nn.Sequential(
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2),
        )

    def _forward_dense(self, x: torch.Tensor, block: nn.ModuleList) -> torch.Tensor:
        feats = [x]
        for layer in block:
            out = layer(torch.cat(feats, dim=1))
            feats.append(out)
        return torch.cat(feats, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self._forward_dense(x, self.dense1)
        x = self.trans1(x)
        x = self._forward_dense(x, self.dense2)
        x = self.trans2(x)
        x = self._forward_dense(x, self.dense3)
        x = F.relu(self.bn_final(x))
        x = self.gap(x)
        x = self.regressor(x)
        return x.squeeze(1)


def get_regressor(name: str = "cnn", dropout: float = 0.3) -> nn.Module:
    """Factory : 'cnn' | 'densenet'"""
    if name == "cnn":
        return MagCNNRegressor(dropout=dropout)
    elif name == "densenet":
        return MagDenseNetRegressor(dropout=dropout)
    else:
        raise ValueError(f"Modèle inconnu: {name}. Choix: cnn | densenet")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
