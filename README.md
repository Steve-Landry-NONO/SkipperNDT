# SkipperNDT — Identification Intelligente de Pipes par Apprentissage Automatique

Projet de collaboration SKIPPER NDT x HETIC — detection automatique de conduites enterrees
a partir de cartes magnetiques capturees par drone.

---

## Equipe

| Nom | Promotion |
|-----|-----------|
| GILHAS Radia | HETIC 2025-2026 |
| KENGNI Theophane | HETIC 2025-2026 |
| KOUOKAM NONO Steve Landry | HETIC 2025-2026 |
| MAKAMTA Linda | HETIC 2025-2026 |
| TUEKAM Ludovic | HETIC 2025-2026 |

Partenaire industriel : SKIPPER NDT — [m.hu@skipperndt.com](mailto:m.hu@skipperndt.com)

---

## Tableau de Bord des Resultats

| Tache | Description | Objectif SKIPPER | Baseline ML | Resultat final | Statut |
|-------|-------------|-----------------|-------------|----------------|--------|
| T1 | Presence de conduite | Acc > 92%, Recall > 95% | SVM_rbf : 99.5% | CNN 100% sur 102 reels | OK |
| T2 | Largeur carte magnetique | MAE < 1m | GradientBoosting : 10.64m | Geometrique 0.61m sur 51 reels | OK |
| T3 | Suffisance du courant | Acc > 90% | RandomForest : 88.5% | CNN 94.9% (score SKIPPER) | OK |
| T4 | Conduites paralleles | F1 > 0.80 | SVM_rbf : F1=96.0% | CNN F1=99.0% | OK |

> T2 : La largeur de carte n'est pas predictible par deep learning (propriete geometrique,
> non statistique). L'algorithme geometrique `scripts/geometric_width.py` exploite les bords
> NaN de la trajectoire de vol — MAE = 0.61m, mediane = 0.21m, 92% des predictions sous 1m.

---

## Contexte

SKIPPER NDT utilise des drones equipes de capteurs magnetiques pour detecter des reseaux
souterrains (petrole, gaz, eau, cables). Un champ magnetique est induit en electrifiant les
sorties de pipes/forages. Les donnees brutes sont des cartes 4 canaux (Bx, By, Bz, Norme)
au format `.npz`, en nanoTesla (nT), avec une resolution spatiale de 0.2m/pixel.

---

## Structure du Projet

```
SkipperNDT/
├── data/
│   ├── raw/                              # Donnees brutes (.npz) — non versionnees
│   │   ├── real_data/                    # 102 fichiers reels (51 pipe + 51 no_pipe)
│   │   └── Training_database_float16/    # 2833 fichiers synthetiques
│   └── pipe_presence_width_detection_label.csv  # Labels T2 (width_m)
│
├── notebooks/
│   └── task2_colab_gpu.ipynb             # Entrainement T2 sur GPU Colab
│
├── src/
│   ├── preprocessing/
│   │   ├── loader.py         # Chargement .npz -> (H, W, 4) float32
│   │   ├── normalization.py  # Strategies de normalisation (zscore, minmax, robust)
│   │   ├── catalog.py        # DatasetCatalog — index lazy, filtres multi-taches
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
│   ├── geometric_width.py    # Algorithme geometrique T2 — MAE=0.61m
│   ├── analyze_t2_width.py   # Analyse et comparaison des approches T2
│   └── run_eda.py            # EDA complete -> 5 figures
│
├── task1/                    # Tache 1 : pipe_present (CNN 100% sur reels)
│   ├── train.py
│   ├── inference.py
│   ├── checkpoints/          # best_model.pt (non versionne)
│   └── results/
│       ├── baseline_results.json
│       └── inference_real.json
│
├── task2/                    # Tache 2 : map_width (Geometrique MAE=0.61m)
│   ├── train.py              # CNN (GPU recommande) — pour comparaison
│   ├── inference.py
│   └── checkpoints/          # best_model.pt (non versionne)
│
├── task3/                    # Tache 3 : current_sufficient (CNN 94.9%)
│   ├── train.py
│   ├── inference.py
│   ├── checkpoints/
│   │   └── metrics.json
│   └── results/
│       └── baseline_results.json
│
├── task4/                    # Tache 4 : parallel_pipelines (CNN F1=99%)
│   ├── train.py
│   ├── inference.py
│   ├── checkpoints/          # best_model.pt (non versionne)
│   └── results/
│       └── baseline_results.json
│
├── docs/
│   ├── geometric_width_explainer.html       # Visualisation 3D interactive (7 etapes)
│   ├── Article-Skipper-Groupe5-Fr.pdf       # Article academique (francais)
│   └── Article-Skipper-Groupe5-En.pdf       # Article academique (anglais)
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

## Donnees

Placer les fichiers `.npz` dans `data/raw/` selon la structure :

```
data/raw/
├── real_data/                   # real_data_*.npz + real_data_no_pipe_*.npz
└── Training_database_float16/   # tous les fichiers synthetiques
```

Le CSV des labels T2 doit etre place dans `data/` :

```
data/pipe_presence_width_detection_label.csv
```

**Convention de nommage :**

| Fichier | Pipe | T3 | T4 |
|---------|------|----|----|
| `sample_{id}_perfect_straight_clean_field.npz` | oui | clean | single |
| `sample_{id}_offset_straight_noisy_field.npz` | oui | noisy | single |
| `sample_{id}_no_pipe_straight_clean_field.npz` | non | clean | — |
| `parallel_{id}_straight_same_clean_field.npz` | oui | clean | parallel |
| `real_data_{id}.npz` | oui | inconnu | inconnu |
| `real_data_no_pipe_{id}.npz` | non | inconnu | — |

**Format** : array `(H, W, 4)` float16/float32 — cle `data['data']`
**Canaux** : `[0]=Bx`, `[1]=By`, `[2]=Bz`, `[3]=Norm` — unite : nanoTesla
**NaN** : zones hors trajectoire drone — masquer, ne pas interpoler
**Resolution** : 0.2m/pixel

---

## Utilisation

### T2 — Algorithme geometrique (resultat principal)

```bash
# Tester sur un fichier unique
python scripts/geometric_width.py --file data/raw/real_data/real_data_00000.npz

# Evaluer sur les 51 donnees reelles labelisees
python scripts/geometric_width.py \
    --csv data/pipe_presence_width_detection_label.csv \
    --data_dir data/raw \
    --reals_only

# Evaluer sur 100 fichiers synthetiques
python scripts/geometric_width.py \
    --csv data/pipe_presence_width_detection_label.csv \
    --data_dir data/raw \
    --n_samples 100
```

**Resultats sur donnees reelles (51 fichiers) :**

```
MAE     : 0.61m   (objectif : < 1m)
Mediane : 0.21m
< 1m    : 92%
< 5m    : 98%
```

**Principe de l'algorithme (7 etapes) :**

1. Profil Bz par ligne -> lissage gaussien (sigma=3) -> argmin = position du pipe
2. Trace brute du pipe (sequence de colonnes minimales)
3. Lissage de la trace (moyenne glissante, fenetre=10)
4. Tangente locale : `a = Delta_row / Delta_col`
5. Perpendiculaire analytique : `a_perp = -1/a`
6. Marche pixel par pixel (pas=0.5px) jusqu'au premier NaN (bord de carte)
7. Largeur locale = `(d_gauche + d_droite) * 0.2m/px` -> mediane finale

Visualisation interactive : `docs/geometric_width_explainer.html`

---

### Exploration des Donnees (EDA)

```bash
python scripts/run_eda.py --data_dir data/raw --output_dir outputs/figures
python scripts/run_eda.py --figures 1,3,5
```

Figures generees : visualisation 4 canaux, distributions + test KS, separabilite
PCA/KNN/SVM, profils 1D dipole, domain gap synth vers reel.

---

### Entrainement et Inference

#### T1 — pipe_present

```bash
python task1/train.py --mode baseline --data_dir data/raw
python task1/train.py --mode cnn --model cnn --epochs 20 --batch_size 32
python task1/inference.py --input data/raw/real_data/ --model task1/checkpoints/best_model.pt
# -> {"pipeline_present": 1, "confidence": 0.9999, "model": "cnn"}
```

#### T2 — map_width (algorithme geometrique recommande)

```bash
# Methode recommandee : algorithme geometrique (sans entrainement)
python scripts/geometric_width.py --file image.npz

# Methode alternative : CNN (GPU recommande, MAE ~8-9m)
python task2/train.py --mode cnn --model cnn --epochs 50 --batch_size 32 \
    --csv data/pipe_presence_width_detection_label.csv --data_dir data/raw
python task2/inference.py --input image.npz --model task2/checkpoints/best_model.pt
```

#### T3 — current_sufficient

```bash
python task3/train.py --mode baseline --data_dir data/raw
python task3/train.py --mode cnn --model cnn --epochs 20 --batch_size 32
python task3/inference.py --input image.npz --model task3/checkpoints/best_model.pt
# -> {"current_sufficient": 1, "label": "clean", "confidence": 0.97}
```

#### T4 — parallel_pipelines

```bash
python task4/train.py --mode baseline --data_dir data/raw
python task4/train.py --mode cnn --model cnn --epochs 30 --batch_size 32
python task4/inference.py --input image.npz --model task4/checkpoints/best_model.pt
# -> {"parallel_pipelines": 1, "label": "parallel", "confidence": 0.99}
```

---

## Points Techniques Cles

**Normalisation** : zscore individuelle par canal par image. Les ordres de grandeur absolus
varient de ~1 nT (synthetique no-pipe) a ~2000 nT (pipe fort courant) — une normalisation
globale effacerait cette variabilite structurelle.

**NaN** : zones hors trajectoire drone. Masques avant normalisation, remplaces par 0.0
apres. Ne pas interpoler. Ces NaN constituent les bornes naturelles utilisees par
l'algorithme geometrique T2.

**Domain gap synth vers reel** : les synthetiques no-pipe sont centrees sur ~0 nT, les
reelles ont un offset DC positif. La normalisation individuelle attenue ce probleme —
valide : T1 CNN = 100% sur les 102 donnees reelles.

**Resize** : images de 150x150 a 4000x3750 px -> resize scipy vers 128x128. Global
Average Pooling en sortie -> invariant a la taille d'entree.

**T2 — pourquoi le deep learning echoue** : `map_width` est une propriete geometrique
de la carte de vol (largeur du couloir survole), non une caracteristique statistique du
signal magnetique. Tous les modeles CNN/DenseNet plafonnent a ~8-9m MAE quelle que soit
la duree d'entrainement. L'algorithme geometrique est 13x plus precis.

**T4 — correction label** : les images no-pipe ont `parallel_pipelines = None` (pas 0).
Les exclure du dataset T4 est indispensable pour un entrainement correct.

**Framework** : PyTorch — livrables `train.py` + `inference.py` + poids `.pt` pour
chaque tache.

---

## Observations EDA

- Distributions non-gaussiennes (test KS, p~0) — skewness et kurtosis sont des features
  discriminantes
- Domain gap visible sur canal Norm : synthetiques no-pipe ~0 nT, reelles offset DC positif
- Signature dipole sur Bx/By (profil antisymetrique) et Bz (creux) — le CNN apprend
  cette structure spatiale
- Correlation inter-canaux elevee avec pipe (>0.8) vs sans pipe — feature utile pour la
  baseline ML
- Distribution map_width tres asymetrique (2m -> 155m, moyenne 36.9m) -> entrainement
  en log-space indispensable pour T2 CNN

---

## Limitations Connues

- Les donnees reelles n'ont pas de label T3 (current_sufficient) ni T4
  (parallel_pipelines) — impossible d'evaluer ces taches sur des donnees de terrain.
- L'algorithme geometrique T2 suppose que les bords NaN de la carte correspondent
  exactement aux limites de la trajectoire de vol. Sur des cartes avec NaN internes
  (signal hors-plage), la mesure peut etre sous-estimee.
- Le domain gap synth -> reel reste partiellement ouvert : les synthetiques no-pipe
  n'ont pas l'offset DC positif des donnees reelles. La normalisation individuelle
  attenue le probleme mais ne l'elimine pas.
- Le resize scipy a 128x128 perd de l'information sur les grandes cartes
  (4000x3750 px -> 128x128). Un patch central ou un multi-scale ferait mieux.

---

## Publications

Les articles de recherche associes au projet sont disponibles dans le dossier `docs/`.

| Langue | Fichier |
|--------|---------|
| Francais | [Article-Skipper-Groupe5-Fr.pdf](docs/Article-Skipper-Groupe5-Fr.pdf) |
| Anglais  | [Article-Skipper-Groupe5-En.pdf](docs/Article-Skipper-Groupe5-En.pdf) |

---

## Licence

MIT — voir [LICENSE](LICENSE)

---

## Contributing

Ce projet est soumis dans le cadre du Mastere Data & IA HETIC 2025-2026.
Les contributions externes ne sont pas acceptees pendant la periode d'evaluation.
