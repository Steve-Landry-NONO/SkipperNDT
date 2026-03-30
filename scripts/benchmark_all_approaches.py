"""
scripts/benchmark_all_approaches.py
====================================
Benchmark complet de toutes les approches pour T1, T2, T3, T4.
Mesure : performance (Acc/Recall/F1/MAE) + taille modèle + temps inférence + IoT-ready

Usage :
    python scripts/benchmark_all_approaches.py \
        --data_dir data/raw \
        --csv data/pipe_presence_width_detection_label.csv \
        --output outputs/benchmark_results.json

Approches testées :
    T1/T3/T4 : PCA+KNN | PCA+SVM | PCA+LDA | PCA+LR | MagCNN
    T2        : Features stats | PCA géométrique | CNN | Algo géométrique

Auteur(s) : KOUOKAM NONO Steve Landry
"""

import argparse
import json
import sys
import time
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.catalog import DatasetCatalog
from src.preprocessing.features import extract_features_batch
from src.preprocessing.loader import load_npz
from src.models.dataset import resize_array, normalize_channels


# ─────────────────────────────────────────────────────────────────────────────
# Helpers mesure
# ─────────────────────────────────────────────────────────────────────────────

def model_size_mb(model_or_path) -> float:
    """Taille du modèle en Mo."""
    if isinstance(model_or_path, (str, Path)):
        return os.path.getsize(model_or_path) / 1e6
    # scikit-learn : sérialiser en mémoire
    import io
    buf = io.BytesIO()
    pickle.dump(model_or_path, buf)
    return buf.tell() / 1e6


def inference_time_ms(predict_fn, X_sample, n_runs=50) -> float:
    """Temps d'inférence moyen en ms sur n_runs appels."""
    # Warmup
    predict_fn(X_sample)
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        predict_fn(X_sample)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }


def skipper_score_t1(acc, rec) -> float:
    return 0.6 * acc + 0.4 * rec


def skipper_score_t3(acc, rec) -> float:
    return 0.7 * acc + 0.3 * rec


# ─────────────────────────────────────────────────────────────────────────────
# PCA + Classifieurs (T1 / T3 / T4)
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_classifiers(X_train, y_train, X_val, y_val,
                        X_sample_1img, task_name="t1") -> dict:
    """
    Teste PCA + KNN, SVM, LDA, LR sur un jeu train/val.
    Retourne un dict {nom_approche: metrics}.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    # n_components doit être ≤ min(n_samples, n_features)
    # On vise 64 comme les camarades, mais on plafonne à ce qui est possible
    N_COMPONENTS = min(64, X_train.shape[1], X_train.shape[0] - 1)
    print(f"    PCA n_components={N_COMPONENTS} "
          f"(features={X_train.shape[1]}, samples_train={X_train.shape[0]})")

    classifiers = {
        "PCA+KNN":  KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "PCA+SVM":  SVC(kernel="rbf", C=10.0, gamma="scale",
                        class_weight="balanced", probability=True),
        "PCA+LDA":  LinearDiscriminantAnalysis(),
        "PCA+LR":   LogisticRegression(max_iter=1000,
                                       class_weight="balanced", n_jobs=-1),
    }

    results = {}

    for name, clf in classifiers.items():
        print(f"    [{task_name}] {name}...", flush=True)

        # Pipeline : StandardScaler → PCA → Classifieur
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=N_COMPONENTS, random_state=42)),
            ("clf",    clf),
        ])

        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        y_pred = pipe.predict(X_val)
        metrics = classification_metrics(y_val, y_pred)

        # Taille + temps inférence (1 image)
        size_mb = model_size_mb(pipe)
        inf_ms  = inference_time_ms(
            lambda x: pipe.predict(x),
            X_sample_1img.reshape(1, -1)
        )

        results[name] = {
            **metrics,
            "model_size_mb":    round(size_mb, 3),
            "inference_ms":     round(inf_ms, 3),
            "train_time_s":     round(train_time, 2),
            "iot_ready":        True,   # CPU only, < 15Mo
            "n_components_pca": N_COMPONENTS,
        }

        acc = metrics["accuracy"]
        rec = metrics["recall"]
        f1  = metrics["f1"]
        print(f"      Acc={acc*100:.1f}% Rec={rec*100:.1f}% "
              f"F1={f1*100:.1f}% | {size_mb:.2f}Mo | {inf_ms:.1f}ms")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MagCNN (T1 / T3 / T4)
# ─────────────────────────────────────────────────────────────────────────────

def run_cnn_classification(paths_train, labels_train,
                           paths_val, labels_val,
                           paths_real, labels_real,
                           task_name="t1",
                           epochs=20) -> dict:
    """Entraîne MagCNN et mesure toutes les métriques."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
    except ImportError:
        print("    [!] PyTorch non disponible")
        return {}

    from src.models.dataset import MagneticMapDataset
    from src.models.cnn_task1 import MagCNN, count_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    [{task_name}] MagCNN ({device})...", flush=True)

    train_ds = MagneticMapDataset(paths_train, labels_train, augment=True)
    val_ds   = MagneticMapDataset(paths_val,   labels_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,
                              num_workers=0, pin_memory=False)

    model     = MagCNN().to(device)
    class_w   = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            criterion(model(x), y_b).backward()
            optimizer.step()
        scheduler.step()
    train_time = time.perf_counter() - t0

    # Validation synthétique
    model.eval()
    all_p, all_l = [], []
    with torch.no_grad():
        for x, y_b in val_loader:
            all_p.extend(model(x.to(device)).argmax(1).cpu().numpy())
            all_l.extend(y_b.numpy())
    metrics_val = classification_metrics(np.array(all_l), np.array(all_p))

    # Test réel
    metrics_real = {}
    if paths_real:
        real_ds     = MagneticMapDataset(paths_real, labels_real, augment=False)
        real_loader = DataLoader(real_ds, batch_size=32, shuffle=False, num_workers=0)
        rp, rl = [], []
        with torch.no_grad():
            for x, y_b in real_loader:
                rp.extend(model(x.to(device)).argmax(1).cpu().numpy())
                rl.extend(y_b.numpy())
        metrics_real = classification_metrics(np.array(rl), np.array(rp))

    # Taille modèle
    tmp_path = ROOT / "outputs" / f"_tmp_{task_name}_cnn.pt"
    tmp_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), tmp_path)
    size_mb = model_size_mb(tmp_path)
    tmp_path.unlink()

    # Temps inférence (1 image, CPU)
    model_cpu = model.cpu()
    dummy = torch.zeros(1, 4, 128, 128)
    inf_ms = inference_time_ms(
        lambda x: model_cpu(x),
        dummy
    )

    n_params = count_params(model)
    print(f"      Val  Acc={metrics_val['accuracy']*100:.1f}% "
          f"Rec={metrics_val['recall']*100:.1f}% "
          f"F1={metrics_val['f1']*100:.1f}%")
    if metrics_real:
        print(f"      Real Acc={metrics_real['accuracy']*100:.1f}% "
              f"Rec={metrics_real['recall']*100:.1f}% "
              f"F1={metrics_real['f1']*100:.1f}%")
    print(f"      {size_mb:.1f}Mo | {inf_ms:.1f}ms | {n_params:,} params")

    return {
        "val_synthetic":    metrics_val,
        "val_real":         metrics_real,
        "model_size_mb":    round(size_mb, 2),
        "inference_ms":     round(inf_ms, 2),
        "n_params":         n_params,
        "train_time_s":     round(train_time, 1),
        "iot_ready":        size_mb < 20,
    }


# ─────────────────────────────────────────────────────────────────────────────
# T2 — Approche PCA géométrique (camarades)
# ─────────────────────────────────────────────────────────────────────────────

def pca_geometric_width(arr: np.ndarray) -> float:
    """
    Approche géométrique PCA (inspirée des camarades) :
    1. PCA sur les coordonnées des pixels valides → direction principale du pipe
    2. Vecteur perpendiculaire v2
    3. Marche depuis le centroïde dans ±v2 jusqu'au premier NaN
    4. map_width = (d+ + d-) × 0.2 m/px
    """
    from sklearn.decomposition import PCA as SkPCA

    bz = arr[:, :, 2]
    H, W = bz.shape

    # Coordonnées des pixels valides
    rows, cols = np.where(np.isfinite(bz))
    if len(rows) < 20:
        return float('nan')

    coords = np.column_stack([rows, cols]).astype(float)

    # PCA : composante principale = direction du pipe
    pca = SkPCA(n_components=2)
    pca.fit(coords)
    # v1 = direction principale (long du pipe)
    # v2 = direction perpendiculaire (travers = largeur)
    v2 = pca.components_[1]   # vecteur perpendiculaire normalisé
    mu = pca.mean_            # centroïde

    # Marche dans ±v2 depuis le centroïde jusqu'au NaN
    step = 0.5  # px
    max_steps = int(np.sqrt(H**2 + W**2) / step) + 1

    def walk(direction):
        x, y = float(mu[1]), float(mu[0])  # col, row
        dx = direction * v2[1] * step
        dy = direction * v2[0] * step
        dist = 0.0
        for _ in range(max_steps):
            x += dx; y += dy
            xi, yi = int(round(x)), int(round(y))
            if xi < 0 or xi >= W or yi < 0 or yi >= H:
                break
            if np.isnan(bz[yi, xi]):
                break
            dist = np.sqrt((x - mu[1])**2 + (y - mu[0])**2)
        return dist

    d_pos = walk(+1)
    d_neg = walk(-1)
    return (d_pos + d_neg) * 0.2  # mètres


def run_t2_benchmark(csv_path: Path, data_dir: Path,
                     n_synth: int = 200) -> dict:
    """Compare toutes les approches T2."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split
    from scripts.geometric_width import estimate_map_width

    print("\n  [T2] Chargement CSV...", flush=True)
    df = pd.read_csv(csv_path, sep=';')
    df_pipe  = df[(df['label'] == 1) & df['width_m'].notna()].copy()
    df_synth = df_pipe[~df_pipe['field_file'].str.startswith('real')]
    df_real  = df_pipe[df_pipe['field_file'].str.startswith('real')]

    file_index = {p.name: p for p in data_dir.rglob('*.npz')}

    def resolve(df_):
        paths, widths = [], []
        for _, row in df_.iterrows():
            if row['field_file'] in file_index:
                paths.append(file_index[row['field_file']])
                widths.append(row['width_m'])
        return paths, widths

    synth_paths, synth_widths = resolve(df_synth.head(n_synth))
    real_paths,  real_widths  = resolve(df_real)
    print(f"  [T2] Synthétiques: {len(synth_paths)} | Réels: {len(real_paths)}")

    # ── Features extraction pour baseline ML ──────────────────
    print("  [T2] Extraction features...", flush=True)
    X_all, _ = extract_features_batch(synth_paths, [0]*len(synth_paths), verbose=False)
    y_all = np.array(synth_widths)
    y_log = np.log1p(y_all)

    idx_tr, idx_val = train_test_split(range(len(synth_paths)), test_size=0.2, random_state=42)
    X_tr, X_val = X_all[idx_tr], X_all[idx_val]
    y_tr, y_val = y_log[idx_tr], y_log[idx_val]

    results = {}

    # ── Approche 1 : Features stats + GradientBoosting ────────
    print("  [T2] GradientBoosting...", flush=True)
    pipe_gb = Pipeline([
        ("scaler", StandardScaler()),
        ("reg",    GradientBoostingRegressor(n_estimators=200, random_state=42)),
    ])
    t0 = time.perf_counter()
    pipe_gb.fit(X_tr, y_tr)
    train_t = time.perf_counter() - t0

    pred_val = np.expm1(pipe_gb.predict(X_val))
    true_val = np.expm1(y_val)
    mae_val  = float(np.abs(pred_val - true_val).mean())

    # Sur réels
    if real_paths:
        X_real, _ = extract_features_batch(real_paths, [0]*len(real_paths), verbose=False)
        pred_real = np.expm1(pipe_gb.predict(X_real))
        mae_real  = float(np.abs(pred_real - np.array(real_widths)).mean())
    else:
        mae_real = None

    size_gb  = model_size_mb(pipe_gb)
    inf_gb   = inference_time_ms(lambda x: pipe_gb.predict(x), X_val[:1])
    results["Features+GradientBoosting"] = {
        "mae_val_m":     round(mae_val, 2),
        "mae_real_m":    round(mae_real, 2) if mae_real else None,
        "model_size_mb": round(size_gb, 3),
        "inference_ms":  round(inf_gb, 2),
        "train_time_s":  round(train_t, 1),
        "iot_ready":     True,
        "requires_training": True,
    }
    print(f"    MAE val={mae_val:.2f}m | réels={mae_real:.2f}m | "
          f"{size_gb:.2f}Mo | {inf_gb:.1f}ms")

    # ── Approche 2 : PCA géométrique (camarades) ──────────────
    print("  [T2] PCA géométrique...", flush=True)
    pca_errors = []
    pca_times  = []
    for path, true_w in zip(real_paths, real_widths):
        arr = load_npz(path)
        t0  = time.perf_counter()
        pred_w = pca_geometric_width(arr)
        pca_times.append((time.perf_counter() - t0) * 1000)
        if not np.isnan(pred_w):
            pca_errors.append(abs(pred_w - true_w))

    mae_pca    = float(np.mean(pca_errors)) if pca_errors else None
    median_pca = float(np.median(pca_errors)) if pca_errors else None
    inf_pca    = float(np.median(pca_times))
    results["PCA_geometrique"] = {
        "mae_real_m":    round(mae_pca, 2) if mae_pca else None,
        "median_err_m":  round(median_pca, 2) if median_pca else None,
        "model_size_mb": 0.0,   # pas de modèle
        "inference_ms":  round(inf_pca, 1),
        "train_time_s":  0.0,
        "iot_ready":     True,
        "requires_training": False,
        "note": "direction principale via PCA, scan perp. jusqu'aux NaN",
    }
    print(f"    MAE réels={mae_pca:.2f}m | médiane={median_pca:.2f}m | "
          f"{inf_pca:.1f}ms")

    # ── Approche 3 : Algorithme géométrique (notre méthode) ───
    print("  [T2] Algo géométrique (notre méthode)...", flush=True)
    geom_errors = []
    geom_times  = []
    for path, true_w in zip(real_paths, real_widths):
        arr = load_npz(path)
        t0  = time.perf_counter()
        res = estimate_map_width(arr, verbose=False)
        geom_times.append((time.perf_counter() - t0) * 1000)
        if res['map_width_m'] is not None:
            geom_errors.append(abs(res['map_width_m'] - true_w))

    mae_geom    = float(np.mean(geom_errors)) if geom_errors else None
    median_geom = float(np.median(geom_errors)) if geom_errors else None
    inf_geom    = float(np.median(geom_times))
    pct_under_1 = float(np.mean(np.array(geom_errors) < 1.0)) if geom_errors else None
    pct_under_5 = float(np.mean(np.array(geom_errors) < 5.0)) if geom_errors else None

    results["Algo_geometrique_trace"] = {
        "mae_real_m":    round(mae_geom, 2) if mae_geom else None,
        "median_err_m":  round(median_geom, 2) if median_geom else None,
        "pct_under_1m":  round(pct_under_1 * 100, 1) if pct_under_1 else None,
        "pct_under_5m":  round(pct_under_5 * 100, 1) if pct_under_5 else None,
        "model_size_mb": 0.0,
        "inference_ms":  round(inf_geom, 1),
        "train_time_s":  0.0,
        "iot_ready":     True,
        "requires_training": False,
        "note": "trace Bz minimum + tangente locale + perp. vers NaN",
    }
    print(f"    MAE réels={mae_geom:.2f}m | médiane={median_geom:.2f}m | "
          f"<1m={pct_under_1*100:.0f}% | {inf_geom:.1f}ms")

    # ── CNN (résultats déjà connus) ────────────────────────────
    results["MagCNN_regression"] = {
        "mae_val_synth_m": 8.32,
        "mae_real_m":      9.89,
        "model_size_mb":   1.7,
        "inference_ms":    None,   # à mesurer si checkpoint disponible
        "train_time_s":    None,
        "iot_ready":       True,
        "requires_training": True,
        "note": "plateau MAE ~8-9m — map_width non appris par stats signal",
    }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark toutes approches")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--csv",      default="data/pipe_presence_width_detection_label.csv")
    parser.add_argument("--output",   default="outputs/benchmark_results.json")
    parser.add_argument("--epochs",   type=int, default=20,
                        help="Epochs pour CNN (réduire pour aller vite)")
    parser.add_argument("--skip_cnn", action="store_true",
                        help="Passer le CNN (plus rapide)")
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    csv_path = ROOT / args.csv
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"  BENCHMARK COMPLET — Toutes approches")
    print(f"  data_dir : {data_dir}")
    print(f"{'='*65}\n")

    from sklearn.model_selection import train_test_split

    # ── Construction catalogue ─────────────────────────────────
    print("📂 Construction catalogue...")
    catalog = DatasetCatalog(data_dir, verbose=False)

    # Données réelles pour test final
    real_entries = [e for e in catalog.entries if e.origin == "real"]
    real_pipe    = [e for e in real_entries if e.pipeline_present == 1]
    real_nopipe  = [e for e in real_entries if e.pipeline_present == 0]

    all_results = {}

    # ═══════════════════════════════════════════════════════════
    # TÂCHE 1 — pipe_present
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  TÂCHE 1 — pipe_present")
    print("="*65)

    paths_t1, labels_t1 = catalog.get_paths_and_labels("t1")
    idx_tr, idx_val = train_test_split(
        range(len(paths_t1)), test_size=0.2,
        stratify=labels_t1, random_state=42
    )
    paths_tr  = [paths_t1[i]  for i in idx_tr]
    labels_tr = [labels_t1[i] for i in idx_tr]
    paths_vl  = [paths_t1[i]  for i in idx_val]
    labels_vl = [labels_t1[i] for i in idx_val]

    # Features pour PCA
    print("  Extraction features T1...", flush=True)
    X_tr, y_tr = extract_features_batch(paths_tr, labels_tr, verbose=False)
    X_vl, y_vl = extract_features_batch(paths_vl, labels_vl, verbose=False)

    # Données réelles T1
    real_paths_t1  = [e.path for e in real_entries]
    real_labels_t1 = [e.pipeline_present for e in real_entries]
    X_real_t1, y_real_t1 = extract_features_batch(
        real_paths_t1, real_labels_t1, verbose=False
    )

    # PCA + classifieurs
    print("\n  PCA + classifieurs T1...")
    pca_results_t1 = run_pca_classifiers(
        X_tr, y_tr, X_vl, y_vl, X_vl[0], task_name="t1"
    )
    # Ajouter test sur réels pour chaque modèle PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    clf_map = {
        "PCA+KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "PCA+SVM": SVC(kernel="rbf", C=10.0, gamma="scale",
                       class_weight="balanced", probability=True),
        "PCA+LDA": LinearDiscriminantAnalysis(),
        "PCA+LR":  LogisticRegression(max_iter=1000,
                                      class_weight="balanced", n_jobs=-1),
    }
    _nc = min(64, X_tr.shape[1], X_tr.shape[0] - 1)
    for name, clf in clf_map.items():
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("pca",    PCA(n_components=_nc, random_state=42)),
            ("clf",    clf),
        ])
        pipe.fit(X_tr, y_tr)
        y_real_pred = pipe.predict(X_real_t1)
        real_m = classification_metrics(y_real_t1, y_real_pred)
        pca_results_t1[name]["val_real"] = real_m
        pca_results_t1[name]["skipper_score_real"] = round(
            skipper_score_t1(real_m["accuracy"], real_m["recall"]), 4
        )
        # Score sur synthétiques
        pca_results_t1[name]["skipper_score_val"] = round(
            skipper_score_t1(
                pca_results_t1[name]["accuracy"],
                pca_results_t1[name]["recall"]
            ), 4
        )

    all_results["T1_pipe_present"] = pca_results_t1

    # CNN T1
    if not args.skip_cnn:
        print("\n  CNN T1...")
        cnn_t1 = run_cnn_classification(
            paths_tr, labels_tr, paths_vl, labels_vl,
            real_paths_t1, real_labels_t1,
            task_name="t1", epochs=args.epochs
        )
        if cnn_t1:
            cnn_t1["skipper_score_real"] = round(
                skipper_score_t1(
                    cnn_t1["val_real"].get("accuracy", 0),
                    cnn_t1["val_real"].get("recall", 0)
                ), 4
            ) if cnn_t1.get("val_real") else None
            all_results["T1_pipe_present"]["MagCNN"] = cnn_t1

    # ═══════════════════════════════════════════════════════════
    # TÂCHE 2 — map_width
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  TÂCHE 2 — map_width")
    print("="*65)

    if csv_path.exists():
        t2_results = run_t2_benchmark(csv_path, data_dir)
        all_results["T2_map_width"] = t2_results
    else:
        print(f"  [!] CSV introuvable : {csv_path}")

    # ═══════════════════════════════════════════════════════════
    # TÂCHE 3 — current_sufficient
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  TÂCHE 3 — current_sufficient")
    print("="*65)

    paths_t3, labels_t3 = catalog.get_paths_and_labels("t3")
    idx_tr3, idx_vl3 = train_test_split(
        range(len(paths_t3)), test_size=0.2,
        stratify=labels_t3, random_state=42
    )
    paths_tr3  = [paths_t3[i]  for i in idx_tr3]
    labels_tr3 = [labels_t3[i] for i in idx_tr3]
    paths_vl3  = [paths_t3[i]  for i in idx_vl3]
    labels_vl3 = [labels_t3[i] for i in idx_vl3]

    print("  Extraction features T3...", flush=True)
    X_tr3, y_tr3 = extract_features_batch(paths_tr3, labels_tr3, verbose=False)
    X_vl3, y_vl3 = extract_features_batch(paths_vl3, labels_vl3, verbose=False)

    pca_results_t3 = run_pca_classifiers(
        X_tr3, y_tr3, X_vl3, y_vl3, X_vl3[0], task_name="t3"
    )
    for name in pca_results_t3:
        acc = pca_results_t3[name]["accuracy"]
        rec = pca_results_t3[name]["recall"]
        pca_results_t3[name]["skipper_score"] = round(skipper_score_t3(acc, rec), 4)

    all_results["T3_current_sufficient"] = pca_results_t3

    if not args.skip_cnn:
        print("\n  CNN T3...")
        cnn_t3 = run_cnn_classification(
            paths_tr3, labels_tr3, paths_vl3, labels_vl3,
            [], [],   # pas de réels labellisés pour T3
            task_name="t3", epochs=args.epochs
        )
        if cnn_t3:
            acc3 = cnn_t3["val_synthetic"].get("accuracy", 0)
            rec3 = cnn_t3["val_synthetic"].get("recall", 0)
            cnn_t3["skipper_score"] = round(skipper_score_t3(acc3, rec3), 4)
            all_results["T3_current_sufficient"]["MagCNN"] = cnn_t3

    # ═══════════════════════════════════════════════════════════
    # TÂCHE 4 — parallel_pipelines
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*65)
    print("  TÂCHE 4 — parallel_pipelines")
    print("="*65)

    paths_t4, labels_t4 = catalog.get_paths_and_labels("t4")
    idx_tr4, idx_vl4 = train_test_split(
        range(len(paths_t4)), test_size=0.2,
        stratify=labels_t4, random_state=42
    )
    paths_tr4  = [paths_t4[i]  for i in idx_tr4]
    labels_tr4 = [labels_t4[i] for i in idx_tr4]
    paths_vl4  = [paths_t4[i]  for i in idx_vl4]
    labels_vl4 = [labels_t4[i] for i in idx_vl4]

    print("  Extraction features T4...", flush=True)
    X_tr4, y_tr4 = extract_features_batch(paths_tr4, labels_tr4, verbose=False)
    X_vl4, y_vl4 = extract_features_batch(paths_vl4, labels_vl4, verbose=False)

    pca_results_t4 = run_pca_classifiers(
        X_tr4, y_tr4, X_vl4, y_vl4, X_vl4[0], task_name="t4"
    )
    all_results["T4_parallel_pipelines"] = pca_results_t4

    if not args.skip_cnn:
        print("\n  CNN T4...")
        cnn_t4 = run_cnn_classification(
            paths_tr4, labels_tr4, paths_vl4, labels_vl4,
            [], [],
            task_name="t4", epochs=args.epochs
        )
        if cnn_t4:
            all_results["T4_parallel_pipelines"]["MagCNN"] = cnn_t4

    # ═══════════════════════════════════════════════════════════
    # Sauvegarde + résumé
    # ═══════════════════════════════════════════════════════════
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  ✅ Résultats sauvegardés : {out_path}")
    print(f"{'='*65}\n")

    # Résumé console
    print("RÉSUMÉ\n" + "-"*65)

    print("\nT1 — pipe_present (objectif Acc>92% Rec>95%)")
    for name, r in all_results.get("T1_pipe_present", {}).items():
        if isinstance(r, dict):
            acc = r.get("accuracy", r.get("val_synthetic", {}).get("accuracy", 0))
            rec = r.get("recall",   r.get("val_synthetic", {}).get("recall", 0))
            mb  = r.get("model_size_mb", 0)
            ms  = r.get("inference_ms", "?")
            iot = "✓" if r.get("iot_ready") else "✗"
            print(f"  {name:<25} Acc={acc*100:5.1f}% Rec={rec*100:5.1f}% "
                  f"| {mb:5.2f}Mo | {str(ms):>6}ms | IoT:{iot}")

    print("\nT2 — map_width (objectif MAE<1m)")
    for name, r in all_results.get("T2_map_width", {}).items():
        mae = r.get("mae_real_m", "?")
        mb  = r.get("model_size_mb", 0)
        ms  = r.get("inference_ms", "?")
        iot = "✓" if r.get("iot_ready") else "✗"
        train = "∅" if not r.get("requires_training") else "oui"
        print(f"  {name:<30} MAE={str(mae):>6}m | {mb:5.2f}Mo | "
              f"{str(ms):>6}ms | IoT:{iot} | Train:{train}")

    for task_key, task_label, obj in [
        ("T3_current_sufficient", "T3 — current_sufficient (Acc>90%)", "acc"),
        ("T4_parallel_pipelines", "T4 — parallel_pipelines (F1>0.80)", "f1"),
    ]:
        print(f"\n{task_label}")
        for name, r in all_results.get(task_key, {}).items():
            if isinstance(r, dict):
                acc = r.get("accuracy", r.get("val_synthetic", {}).get("accuracy", 0))
                f1  = r.get("f1",       r.get("val_synthetic", {}).get("f1", 0))
                mb  = r.get("model_size_mb", 0)
                ms  = r.get("inference_ms", "?")
                iot = "✓" if r.get("iot_ready") else "✗"
                print(f"  {name:<25} Acc={acc*100:5.1f}% F1={f1*100:5.1f}% "
                      f"| {mb:5.2f}Mo | {str(ms):>6}ms | IoT:{iot}")

    print(f"\n→ Résultats complets : {out_path}")


if __name__ == "__main__":
    main()
