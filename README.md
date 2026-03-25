# SkipperNDT — Identification Intelligente de Pipes par Apprentissage Automatique

Projet de collaboration **SKIPPER NDT × HETIC** — Détection automatique de conduites enterrées à partir de cartes magnétiques capturées par drone.

---

## 👥 Équipe

| Nom | |
|-----|-|
| **GILHAS** Radia | HETIC 2025-2026 |
| **KENGNI** Théophane | HETIC 2025-2026 |
| **KOUOKAM NONO** Steve Landry | HETIC 2025-2026 |
| **MAKAMTA** Linda | HETIC 2025-2026 |
| **TUEKAM** Ludovic | HETIC 2025-2026 |

Partenaire industriel : **SKIPPER NDT** — [m.hu@skipperndt.com](mailto:m.hu@skipperndt.com)

---

## 🎯 Tableau de Bord des Résultats

| Tâche | Description | Objectif SKIPPER | Baseline ML | Résultat final | Statut |
|-------|-------------|-----------------|-------------|----------------|--------|
| **T1** | Présence de conduite | Acc > 92%, Recall > 95% | SVM_rbf : 99.5% | CNN **100%** sur 102 réels | ✅ |
| **T2** | Largeur carte magnétique | MAE < 1m | GradientBoosting : 10.64m | **Géométrique 0.61m** sur 51 réels | ✅ |
| **T3** | Suffisance du courant | Acc > 90% | RandomForest : 88.5% | CNN **94.9%** (score SKIPPER) | ✅ |
| **T4** | Conduites parallèles | F1 > 0.80 | SVM_rbf : F1=96.0% | CNN **F1=99.0%** | ✅ |

> **T2** : La largeur de carte n'est pas prédictible par deep learning (propriété géométrique, non statistique). L'algorithme géométrique `scripts/geometric_width.py` exploite les bords NaN de la trajectoire de vol — MAE = 0.61m, médiane = 0.21m, 92% des prédictions sous 1m.

---

## Contexte

SKIPPER NDT utilise des drones équipés de capteurs magnétiques pour détecter des réseaux souterrains (pétrole, gaz, eau, câbles). Un champ magnétique est induit en électrifiant les sorties de pipes/forages. Les données brutes sont des **cartes 4 canaux** (Bx, By, Bz, Norme) au format `.npz`, en nanoTesla (nT), avec une résolution spatiale de **0.2m/pixel**.

---

## Structure du Projet

```
SkipperNDT/
├── data/
│   ├── raw/                              # Données brutes (.npz) — non versionnées
│   │   ├── real_data/                    # 102 fichiers réels (51 pipe + 51 no_pipe)
│   │   └── Training_database_float16/    # 2833 fichiers synthétiques
│   └── pipe_presence_width_detection_label.csv  # Labels T2 (width_m)
│
├── notebooks/
│   └── task2_colab_gpu.ipynb             # Entraînement T2 sur GPU Colab
│
├── src/
│   ├── preprocessing/
│   │   ├── loader.py         # Chargement .npz → (H, W, 4) float32
│   │   ├── normalization.py  # Stratégies de normalisation (zscore, minmax, robust)
│   │   ├── catalog.py        # DatasetCatalog — index lazy, filtres multi-tâches
│   │   ├── labeling.py       # extract_labels() — labels T1/T3/T4 depuis nom fichier
│   │   └── features.py       # extract_features() — 39 features statistiques
│   ├── models/
│   │   ├── dataset.py              # MagneticMapDataset PyTorch (resize 128x128)
│   │   ├── dataset_regression.py   # RegressionDataset pour T2 (log-space)
│   │   ├── cnn_task1.py            # MagCNN + MagDenseNet (classification)
│   │   └── cnn_task2.py            # MagCNNRegressor + MagDenseNetRegressor
│   └── utils/
│       └── visualization.py  # Fonctions EDA
│
├── scripts/
│   ├── geometric_width.py    # ⭐ Algorithme géométrique T2 — MAE=0.61m
│   ├── analyze_t2_width.py   # Analyse et comparaison des approches T2
│   └── run_eda.py            # EDA complète → 5 figures
│
├── task1/                    # Tâche 1 : pipe_present (✅ CNN 100% sur réels)
│   ├── train.py
│   ├── inference.py
│   ├── checkpoints/          # best_model.pt (non versionné)
│   └── results/
│       ├── baseline_results.json
│       └── inference_real.json
│
├── task2/                    # Tâche 2 : map_width (✅ Géométrique MAE=0.61m)
│   ├── train.py              # CNN (GPU recommandé) — pour comparaison
│   ├── inference.py
│   └── checkpoints/          # best_model.pt (non versionné)
│
├── task3/                    # Tâche 3 : current_sufficient (✅ CNN 94.9%)
│   ├── train.py
│   ├── inference.py
│   ├── checkpoints/
│   │   └── metrics.json
│   └── results/
│       └── baseline_results.json
│
├── task4/                    # Tâche 4 : parallel_pipelines (✅ CNN F1=99%)
│   ├── train.py
│   ├── inference.py
│   ├── checkpoints/          # best_model.pt (non versionné)
│   └── results/
│       └── baseline_results.json
│
├── docs/
│   └── geometric_width_explainer.html  # Visualisation 3D interactive (7 étapes)
├── tests/
├── outputs/figures/
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/Steve-Landry-NONO/SkipperNDT.git
cd SkipperNDT
pip install -r requirements.txt
```

---

## Données

Placer les fichiers `.npz` dans `data/raw/` selon la structure :

```
data/raw/
├── real_data/                   # real_data_*.npz + real_data_no_pipe_*.npz
└── Training_database_float16/   # tous les fichiers synthétiques
```

Le CSV des labels T2 doit être placé dans `data/` :
```
data/pipe_presence_width_detection_label.csv
```

**Convention de nommage :**

| Fichier | Pipe | T3 | T4 |
|---------|------|----|----|
| `sample_{id}_perfect_straight_clean_field.npz` | ✓ | clean | single |
| `sample_{id}_offset_straight_noisy_field.npz` | ✓ | noisy | single |
| `sample_{id}_no_pipe_straight_clean_field.npz` | ✗ | clean | — |
| `parallel_{id}_straight_same_clean_field.npz` | ✓ | clean | parallel |
| `real_data_{id}.npz` | ✓ | inconnu | inconnu |
| `real_data_no_pipe_{id}.npz` | ✗ | inconnu | — |

**Format** : array `(H, W, 4)` float16/float32 — clé `data['data']`  
**Canaux** : `[0]=Bx`, `[1]=By`, `[2]=Bz`, `[3]=Norm` — unité : nanoTesla  
**NaN** : zones hors trajectoire drone — masquer, ne pas interpoler  
**Résolution** : 0.2m/pixel

---

## Utilisation

### ⭐ Tâche 2 — Algorithme géométrique (résultat principal)

```bash
# Tester sur un fichier unique
python scripts/geometric_width.py --file data/raw/real_data/real_data_00000.npz

# Évaluer sur les 51 données réelles labellisées
python scripts/geometric_width.py \
    --csv data/pipe_presence_width_detection_label.csv \
    --data_dir data/raw \
    --reals_only

# Évaluer sur 100 fichiers synthétiques
python scripts/geometric_width.py \
    --csv data/pipe_presence_width_detection_label.csv \
    --data_dir data/raw \
    --n_samples 100
```

**Résultats sur données réelles (51 fichiers) :**
```
MAE     : 0.61m   ✓ (objectif : < 1m)
Médiane : 0.21m
< 1m    : 92%
< 5m    : 98%
```

**Principe de l'algorithme (7 étapes) :**
1. Profil Bz par ligne → lissage gaussien (σ=3) → argmin = position du pipe
2. Trace brute du pipe (séquence de colonnes minimales)
3. Lissage de la trace (moyenne glissante, fenêtre=10)
4. Tangente locale : `a = Δrow/Δcol`
5. Perpendiculaire analytique : `a_perp = -1/a`
6. Marche pixel par pixel (pas=0.5px) jusqu'au premier NaN (bord de carte)
7. Largeur locale = `(d_gauche + d_droite) × 0.2m/px` → médiane finale

Visualisation interactive : `docs/geometric_width_explainer.html` (ouvrir dans un navigateur)

---

### Exploration des Données (EDA)

```bash
python scripts/run_eda.py --data_dir data/raw --output_dir outputs/figures
python scripts/run_eda.py --figures 1,3,5
```

Figures générées : visualisation 4 canaux, distributions + test KS, séparabilité PCA/KNN/SVM, profils 1D dipôle, domain gap synth→réel.

---

### Entraînement et Inférence

#### Tâche 1 — pipe_present ✅

```bash
python task1/train.py --mode baseline --data_dir data/raw
python task1/train.py --mode cnn --model cnn --epochs 20 --batch_size 32
python task1/inference.py --input data/raw/real_data/ --model task1/checkpoints/best_model.pt
# → {"pipeline_present": 1, "confidence": 0.9999, "model": "cnn"}
```

#### Tâche 2 — map_width ✅ (algorithme géométrique recommandé)

```bash
# Méthode recommandée : algorithme géométrique (sans entraînement)
python scripts/geometric_width.py --file image.npz

# Méthode alternative : CNN (GPU recommandé, MAE ~8-9m)
python task2/train.py --mode cnn --model cnn --epochs 50 --batch_size 32 \
    --csv data/pipe_presence_width_detection_label.csv --data_dir data/raw
# GPU recommandé : notebooks/task2_colab_gpu.ipynb
python task2/inference.py --input image.npz --model task2/checkpoints/best_model.pt
```

#### Tâche 3 — current_sufficient ✅

```bash
python task3/train.py --mode baseline --data_dir data/raw
python task3/train.py --mode cnn --model cnn --epochs 20 --batch_size 32
python task3/inference.py --input image.npz --model task3/checkpoints/best_model.pt
# → {"current_sufficient": 1, "label": "clean", "confidence": 0.97}
```

#### Tâche 4 — parallel_pipelines ✅

```bash
python task4/train.py --mode baseline --data_dir data/raw
python task4/train.py --mode cnn --model cnn --epochs 30 --batch_size 32
python task4/inference.py --input image.npz --model task4/checkpoints/best_model.pt
# → {"parallel_pipelines": 1, "label": "parallel", "confidence": 0.99}
```

---

## Points Techniques Clés

**Normalisation** : zscore individuelle par canal par image. Les ordres de grandeur absolus varient de ~1 nT (synthétique no-pipe) à ~2000 nT (pipe fort courant) — une normalisation globale effacerait cette variabilité structurelle.

**NaN** : zones hors trajectoire drone. Masqués avant normalisation, remplacés par 0.0 après. Ne pas interpoler. Ces NaN constituent les bornes naturelles utilisées par l'algorithme géométrique T2.

**Domain gap synth → réel** : les synthétiques no-pipe sont centrées sur ~0 nT, les réelles ont un offset DC positif. La normalisation individuelle atténue ce problème — validé : T1 CNN = 100% sur les 102 données réelles.

**Resize** : images de 150×150 à 4000×3750 px → resize scipy vers 128×128. Global Average Pooling en sortie → invariant à la taille d'entrée.

**Tâche 2 — pourquoi le deep learning échoue** : `map_width` est une propriété géométrique de la carte de vol (largeur du couloir survolé), non une caractéristique statistique du signal magnétique. Tous les modèles CNN/DenseNet plafonnent à ~8-9m MAE quelle que soit la durée d'entraînement. L'algorithme géométrique est 13× plus précis.

**Tâche 4 — correction label** : les images no-pipe ont `parallel_pipelines = None` (pas 0). Les exclure du dataset T4 est indispensable pour un entraînement correct.

**Framework** : PyTorch — livrables `train.py` + `inference.py` + poids `.pt` pour chaque tâche.

---

## Observations EDA

- Distributions **non-gaussiennes** (test KS, p≈0) — skewness et kurtosis sont des features discriminantes
- **Domain gap** visible sur canal Norm : synthétiques no-pipe ≈ 0 nT, réelles offset DC positif
- **Signature dipôle** sur Bx/By (profil antisymétrique) et Bz (creux) — le CNN apprend cette structure spatiale
- **Corrélation inter-canaux** élevée avec pipe (>0.8) vs sans pipe — feature utile pour la baseline ML
- **map_width** : distribution très asymétrique (2m → 155m, moyenne 36.9m) → entraînement en log-space indispensable pour T2 CNN
