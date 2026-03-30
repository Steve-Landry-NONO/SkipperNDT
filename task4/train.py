"""
task4/train.py

Entrainement — Tache 4 : Detection de conduites paralleles (parallel_pipelines).

Objectif SKIPPER : F1 > 0.80

Labels (synthetiques uniquement, no_pipe et reels exclus) :
    parallel_*  -> parallel_pipelines = 1  (500  fichiers)
    sample_*    -> parallel_pipelines = 0  (1251 fichiers single pipe)

Note : les images no_pipe ont parallel_pipelines = None (pas 0). Les exclure
est indispensable — inclure des None traites comme 0 pollue l'entrainement.

La sauvegarde du meilleur modele est basee sur le F1 de validation (pas val_loss)
car le desequilibre de classes 1:2.5 rendrait la loss peu discriminante.

Usage :
    python task4/train.py --mode baseline --data_dir data/raw
    python task4/train.py --mode cnn --model cnn --epochs 30 --batch_size 32

Sorties :
    task4/checkpoints/best_model.pt
    task4/results/metrics.json
    task4/results/baseline_results.json

Auteur(s) : GILHAS Radia
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.catalog import DatasetCatalog
from src.preprocessing.features import extract_features_batch, N_FEATURES, FEATURE_NAMES

TARGET_F1 = 0.80


def compute_f1(preds: np.ndarray, labels: np.ndarray):
    """
    Calcule F1, precision et recall pour la classe positive (parallel=1).

    Returns:
        (f1, precision, recall) tous en float
    """
    tp   = int(((preds == 1) & (labels == 1)).sum())
    fp   = int(((preds == 1) & (labels == 0)).sum())
    fn   = int(((preds == 0) & (labels == 1)).sum())
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    return float(f1), float(prec), float(rec)


def run_baseline(data_dir: Path, out_dir: Path) -> dict:
    """
    Baseline ML par cross-validation 5-fold.

    La metrique de comparaison est le F1 (classe parallel=1),
    conformement a l'objectif SKIPPER T4.
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import (
        classification_report, make_scorer, f1_score, recall_score,
    )
    from sklearn.pipeline import Pipeline
    import pickle

    print(f"\n{'='*60}")
    print(f"  Baseline ML — Tache 4 : parallel_pipelines")
    print(f"{'='*60}")

    print("\n  construction du catalogue...")
    catalog = DatasetCatalog(data_dir, verbose=False)
    paths, labels = catalog.get_paths_and_labels("t4")
    n_par = sum(labels)
    n_sin = len(labels) - n_par
    print(f"  {len(paths)} fichiers | parallel(1)={n_par} | single(0)={n_sin}")
    print(f"  ratio desequilibre : 1:{n_sin/n_par:.1f}")

    print("\n  extraction des features...")
    t0 = time.time()
    X, y = extract_features_batch(paths, labels, verbose=True)
    print(f"  {X.shape} features extraites en {time.time()-t0:.0f}s")

    scorers = {
        "f1":       make_scorer(f1_score,     pos_label=1),
        "accuracy": "accuracy",
        "recall":   make_scorer(recall_score, pos_label=1),
    }

    models = {
        "SVM_linear": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True)),
        ]),
        "SVM_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10.0, gamma="scale",
                          class_weight="balanced", probability=True)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
            )),
        ]),
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\n  evaluation cross-validation (5 folds)\n")
    print(f"  {'Modele':<22} {'F1':>10} {'Recall':>10} {'Accuracy':>10}")
    print("  " + "-"*54)

    best_f1   = 0.0
    best_name = ""
    best_pipe = None

    for name, pipe in models.items():
        t0      = time.time()
        cv_res  = cross_validate(pipe, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        f1      = cv_res["test_f1"].mean()
        acc     = cv_res["test_accuracy"].mean()
        rec     = cv_res["test_recall"].mean()
        elapsed = time.time() - t0

        results[name] = {
            "f1":       float(f1),
            "accuracy": float(acc),
            "recall":   float(rec),
            "time_s":   float(elapsed),
        }

        ok = "ok" if f1 >= TARGET_F1 else "x"
        print(f"  {name:<22} {f1*100:>8.1f}% {ok} {rec*100:>8.1f}%  "
              f"{acc*100:>8.1f}%  ({elapsed:.0f}s)")

        if f1 > best_f1:
            best_f1   = f1
            best_name = name
            best_pipe = pipe

    print(f"\n  meilleur modele : {best_name} (F1 = {best_f1*100:.1f}%)")
    print(f"  objectif SKIPPER : F1 >= {TARGET_F1*100:.0f}%")

    print(f"\n  entrainement final sur 100% des donnees...")
    best_pipe.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_best.pkl", "wb") as f:
        pickle.dump({
            "model":         best_pipe,
            "scaler_inside": True,
            "feature_names": FEATURE_NAMES,
            "n_features":    N_FEATURES,
            "task":          "t4_parallel_pipelines",
        }, f)
    print(f"  modele sauvegarde : {out_dir / 'baseline_best.pkl'}")

    y_pred = best_pipe.predict(X)
    print(f"\n  rapport (train complet, verification) :\n")
    print(classification_report(y, y_pred, target_names=["single(0)", "parallel(1)"]))

    summary = {
        "task":       "t4_parallel_pipelines",
        "best_model": best_name,
        "best_f1":    float(best_f1),
        "models":     results,
    }
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  resultats : {out_dir / 'baseline_results.json'}")

    return summary


def run_cnn(
    data_dir:   Path,
    out_dir:    Path,
    model_name: str   = "cnn",
    epochs:     int   = 30,
    batch_size: int   = 16,
    lr:         float = 1e-3,
    val_split:  float = 0.2,
) -> dict:
    """
    Entrainement CNN pour la Tache 4.

    Le checkpoint est sauvegarde sur le meilleur F1 de validation (pas val_loss)
    car la metrique SKIPPER est le F1 et le desequilibre de classes rend la loss
    peu comparable entre epochs.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("PyTorch non disponible.")
        return {}

    from src.models.dataset import MagneticMapDataset
    from src.models.cnn_task1 import get_model, count_params

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CNN — Tache 4 : parallel_pipelines | modele: {model_name}")
    print(f"  device: {device}")
    print(f"{'='*60}")

    catalog = DatasetCatalog(data_dir, verbose=False)
    paths, labels = catalog.get_paths_and_labels("t4")
    n_par = sum(labels)
    n_sin = len(labels) - n_par
    print(f"\n  {len(paths)} fichiers | parallel(1)={n_par} | single(0)={n_sin}")
    print(f"  ratio desequilibre : 1:{n_sin/n_par:.1f}")

    idx_train, idx_val = train_test_split(
        range(len(paths)), test_size=val_split, stratify=labels, random_state=42
    )

    train_ds = MagneticMapDataset(
        [paths[i] for i in idx_train], [labels[i] for i in idx_train], augment=True
    )
    val_ds = MagneticMapDataset(
        [paths[i] for i in idx_val], [labels[i] for i in idx_val], augment=False
    )

    nw = 0
    pm = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pm)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pm)

    print(f"  train: {len(train_ds)} | val: {len(val_ds)}")

    model     = get_model(model_name).to(device)
    print(f"  parametres : {count_params(model):,}")

    class_w   = train_ds.class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1_val      = 0.0
    patience_counter = 0
    patience         = 10
    history          = []

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val F1':>9} "
          f"{'Precision':>10} {'Recall':>9} {'LR':>10}")
    print("-" * 74)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        model.eval()
        val_loss       = 0.0
        all_preds      = []
        all_labels_val = []
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                logits     = model(x)
                val_loss  += criterion(logits, y_batch).item() * len(x)
                all_preds.extend(logits.argmax(dim=1).cpu().numpy())
                all_labels_val.extend(y_batch.cpu().numpy())

        val_loss       /= len(val_ds)
        all_preds       = np.array(all_preds)
        all_labels_val  = np.array(all_labels_val)
        acc             = (all_preds == all_labels_val).mean()
        f1, prec, rec   = compute_f1(all_preds, all_labels_val)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_f1": f1, "val_precision": prec, "val_recall": rec, "val_acc": float(acc),
        })

        ok = "ok" if f1 >= TARGET_F1 else "x"
        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
              f"{f1*100:>8.1f}% {ok} {prec*100:>8.1f}%  {rec*100:>8.1f}%  {current_lr:>10.6f}")

        if f1 > best_f1_val:
            best_f1_val      = f1
            patience_counter = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_f1":          f1,
                "val_precision":   prec,
                "val_recall":      rec,
                "val_acc":         float(acc),
                "model_name":      model_name,
                "task":            "t4_parallel_pipelines",
            }, out_dir / "best_model.pt")
            print(f"         meilleur modele sauvegarde (val_F1={f1*100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  early stopping epoch {epoch} (patience={patience})")
                break

    torch.save({"epoch": epoch, "model_state": model.state_dict()}, out_dir / "last_model.pt")

    best_epoch = max(history, key=lambda h: h["val_f1"])
    results = {
        "task":           "t4_parallel_pipelines",
        "model":          model_name,
        "best_val_f1":    float(best_f1_val),
        "best_val_prec":  float(best_epoch["val_precision"]),
        "best_val_rec":   float(best_epoch["val_recall"]),
        "best_val_acc":   float(best_epoch["val_acc"]),
        "epochs_trained": epoch,
        "history":        history,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  best val_F1        : {best_f1_val*100:.1f}%  (objectif: {TARGET_F1*100:.0f}%)")
    print(f"  best val_Precision : {best_epoch['val_precision']*100:.1f}%")
    print(f"  best val_Recall    : {best_epoch['val_recall']*100:.1f}%")
    print(f"  checkpoints        : {out_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train — Tache 4 : parallel_pipelines")
    parser.add_argument("--mode",       type=str, default="baseline",
                        choices=["baseline", "cnn"])
    parser.add_argument("--model",      type=str, default="cnn",
                        choices=["cnn", "densenet"])
    parser.add_argument("--data_dir",   type=str, default="data/raw")
    parser.add_argument("--out_dir",    type=str, default="task4/checkpoints")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.2)
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir  = ROOT / args.out_dir

    if args.mode == "baseline":
        run_baseline(data_dir, ROOT / "task4/results")
    else:
        run_cnn(
            data_dir, out_dir,
            model_name = args.model,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            val_split  = args.val_split,
        )


if __name__ == "__main__":
    main()
