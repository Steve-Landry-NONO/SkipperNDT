# SkipperNDT — Identification Intelligente de Pipes par Apprentissage Automatique

Projet de collaboration **SKIPPER NDT × HETIC** — Détection automatique de conduites enterrées à partir de cartes magnétiques capturées par drone.

---

## Contexte

SKIPPER NDT utilise des drones équipés de capteurs magnétiques pour détecter des réseaux souterrains (pétrole, gaz, eau, câbles). Un champ magnétique est induit en électrifiant les sorties de pipes/forages. Les données brutes sont des **cartes 4 canaux** (Bx, By, Bz, Norme) au format `.npz` ou `.tif`, en nanoTesla (nT), avec une résolution spatiale de **0.2m/pixel**.

---

## Structure du Projet

```
SkipperNDT/
├── data/
│   ├── raw/                  # Données brutes (.npz / .tif) — non versionnées
│   └── processed/            # Données prétraitées
├── notebooks/                # Notebooks Jupyter d'exploration
├── src/
│   ├── preprocessing/
│   │   ├── loader.py         # Chargement et inspection des fichiers
│   │   └── normalization.py  # Stratégies de normalisation
│   ├── models/               # Architectures PyTorch
│   └── utils/
│       └── visualization.py  # Fonctions de visualisation EDA
├── scripts/
│   └── run_eda.py            # Script EDA complet (génère toutes les figures)
├── tests/
├── outputs/
│   └── figures/              # Figures générées
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/<username>/SkipperNDT.git
cd SkipperNDT
pip install -r requirements.txt
```

---

## Données

Placer les fichiers `.npz` dans `data/raw/`. Convention de nommage :

| Fichier | Origine | Label |
|---------|---------|-------|
| `real_data_*.npz` | Réel | Pipe |
| `real_data_no_pipe_*.npz` | Réel | No pipe |
| `sample_*_no_pipe_*.npz` | Synthétique | No pipe |
| `sample_*_(perfect\|offset\|curved)_*.npz` | Synthétique | Pipe |

**Format** : array `(H, W, 4)` float16/float32 — `data['data']`  
**Canaux** : `[0]=Bx`, `[1]=By`, `[2]=Bz`, `[3]=Norm`  
**NaN** : zones hors trajectoire du drone (à masquer, ne pas interpoler)

---

## Exploration des Données (EDA)

```bash
# Générer toutes les figures d'analyse
python scripts/run_eda.py --data_dir data/raw --output_dir outputs/figures

# Générer seulement certaines figures (ex: 1, 3 et 5)
python scripts/run_eda.py --figures 1,3,5
```

Figures générées :
- **fig1** — Visualisation des 4 canaux (4 cas types)
- **fig2** — Distributions + test KS + fit gaussien
- **fig3** — Séparabilité : PCA 2D, KNN, SVM
- **fig4** — Profils 1D transversaux (signature dipôle du pipe)
- **fig5** — Domain gap synthétique → réel

---

## Tâches ML

| # | Tâche | Type | Objectif |
|---|-------|------|----------|
| 1 | Présence de conduite | Classification binaire | Accuracy > 92%, Recall > 95% |
| 2 | Largeur de carte magnétique | Régression | MAE < 1 mètre |
| 3 | Suffisance du courant | Classification binaire | Accuracy > 90% |
| 4 | Conduites parallèles | Classification binaire avancée | F1 > 0.80 |

---

## Points Techniques Clés

- **Normalisation individuelle par image** : les ordres de grandeur absolus varient énormément selon le terrain/courant/profondeur
- **NaN = zones hors drone** : les exclure des calculs statistiques, ne pas les remplir brutalement
- **Domain gap synth → réel** : les données synthétiques no-pipe sont centrées sur ~0 nT, les réelles ont un offset DC important → stratégies : `apply_abs_offset()`, dégradation synthétique, data augmentation
- **Gestion dimensions variables** : de 150×150 à 4000×3750 px → utiliser Global Average Pooling ou patch-based approach
- **Framework** : PyTorch (obligatoire) — livrables : `train.py` + `inference.py` + poids `.pt/.pth`

---

## Observations EDA

- Toutes les distributions sont **non-gaussiennes** (test KS, p≈0) — la skewness et la kurtosis sont des features discriminantes
- **SVM linéaire ≈ 93%** sur features statistiques → séparabilité partielle confirmée
- La **signature spatiale** (profil dipôle sur Bx/By, bosse sur Bz) est le vrai discriminant, pas seulement l'intensité absolue
- **Forte corrélation inter-canaux** avec pipe (>0.8) vs sans pipe — feature utile en soi

---

## Livrables

```
task1/
├── train.py
├── inference.py    # Usage: python inference.py --input image.tif → {"pipeline_present": 1}
└── model.pt
```
