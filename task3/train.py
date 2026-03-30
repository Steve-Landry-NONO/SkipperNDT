"""
task3/train.py

Entrainement — Tache 3 : Suffisance du courant d'excitation (current_sufficient).

Objectif SKIPPER : score = 70% * Accuracy + 30% * Recall, cible >= 90%

Labels (extraits du nom de fichier) :
    clean_field  -> current_sufficient = 1  (~1983 fichiers synthetiques)
    noisy_field  -> current_sufficient = 0  (~850  fichiers synthetiques)
    real_data_*  -> label inconnu           (exclus de l'entrainement)

Usage :
    python task3/train.py --mode baseline --data_dir data/raw
    python task3/train.py --mode cnn --model cnn --epochs 20 --batch_size 32

Sorties :
    task3/checkpoints/best_model.pt
    task3/results/metrics.json
    task3/results/baseline_results.json

Auteur(s) : TUEKAM Ludovic
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

TARGET_ACC    = 0.90
TARGET_REC    = 0.90
SKIPPER_W_ACC = 0.70
SKIPPER_W_REC = 0.30


def run_baseline(data_dir: Path, out_dir: Path) -> dict:
    """
    Baseline ML par cross-validation 5-fold.

    Seules les donnees synthetiques avec label T3 connu sont utilisees
    (clean_field = 1, noisy_field = 0). Les donnees reelles sont exclues.
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import classification_report, make_scorer, recall_score
    from sklearn.pipeline import Pipeline
    import pickle

    print(f"\n{'='*60}")
    print(f"  Baseline ML — Tache 3 : current_sufficient")
    print(f"{'='*60}")

    print("\n  construction du catalogue...")
    catalog = DatasetCatalog(data_dir, verbose=False)
    paths, labels = catalog.get_paths_and_labels("t3")
    n_clean = sum(labels)
    n_noisy = len(labels) - n_clean
    print(f"  {len(paths)} fichiers | clean(1)={n_clean} | noisy(0)={n_noisy}")

    print("\n  extraction des features...")
    t0 = time.time()
    X, y = extract_features_batch(paths, labels, verbose=True)
    print(f"  {X.shape} features extraites en {time.time()-t0:.0f}s")

    scorers = {
        "accuracy": "accuracy",
        "recall":   make_scorer(recall_score, pos_label=1),
        "f1":       "f1",
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
                n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42
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
    print(f"  {'Modele':<22} {'Accuracy':>10} {'Recall':>10} {'F1':>10} {'Score SKIPPER':>14}")
    print("  " + "-"*66)

    best_score = 0.0
    best_name  = ""
    best_pipe  = None

    for name, pipe in models.items():
        t0      = time.time()
        cv_res  = cross_validate(pipe, X, y, cv=cv, scoring=scorers, n_jobs=-1)
        acc     = cv_res["test_accuracy"].mean()
        rec     = cv_res["test_recall"].mean()
        f1      = cv_res["test_f1"].mean()
        elapsed = time.time() - t0

        skipper_score = SKIPPER_W_ACC * acc + SKIPPER_W_REC * rec
        results[name] = {
            "accuracy":      float(acc),
            "recall":        float(rec),
            "f1":            float(f1),
            "skipper_score": float(skipper_score),
            "time_s":        float(elapsed),
        }

        ok_acc = "ok" if acc >= TARGET_ACC else "x"
        ok_rec = "ok" if rec >= TARGET_REC else "x"
        print(f"  {name:<22} {acc*100:>8.1f}% {ok_acc} {rec*100:>8.1f}% {ok_rec} "
              f"{f1*100:>8.1f}%  [SKIPPER={skipper_score*100:.1f}%] ({elapsed:.0f}s)")

        if skipper_score > best_score:
            best_score = skipper_score
            best_name  = name
            best_pipe  = pipe

    print(f"\n  meilleur modele : {best_name} (score SKIPPER = {best_score*100:.1f}%)")

    print(f"\n  entrainement final sur 100% des donnees...")
    best_pipe.fit(X, y)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_best.pkl", "wb") as f:
        pickle.dump({
            "model":          best_pipe,
            "scaler_inside":  True,
            "feature_names":  FEATURE_NAMES,
            "n_features":     N_FEATURES,
            "task":           "t3_current_sufficient",
        }, f)
    print(f"  modele sauvegarde : {out_dir / 'baseline_best.pkl'}")

    y_pred = best_pipe.predict(X)
    print(f"\n  rapport (train complet, verification) :\n")
    print(classification_report(y, y_pred, target_names=["noisy(0)", "clean(1)"]))

    summary = {
        "task":               "t3_current_sufficient",
        "best_model":         best_name,
        "best_skipper_score": float(best_score),
        "skipper_weights":    {"accuracy": SKIPPER_W_ACC, "recall": SKIPPER_W_REC},
        "models":             results,
    }
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  resultats : {out_dir / 'baseline_results.json'}")

    return summary


def run_cnn(
    data_dir:   Path,
    out_dir:    Path,
    model_name: str   = "cnn",
    epochs:     int   = 50,
    batch_size: int   = 16,
    lr:         float = 1e-3,
    val_split:  float = 0.2,
) -> dict:
    """
    Entrainement CNN pour la Tache 3.

    Architecture identique a T1 — seuls les labels changent (t3 vs t1).
    L'early stopping est base sur la val_loss (patience=10 epochs).
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
    print(f"  CNN — Tache 3 : current_sufficient | modele: {model_name}")
    print(f"  device: {device}")
    print(f"{'='*60}")

    catalog = DatasetCatalog(data_dir, verbose=False)
    paths, labels = catalog.get_paths_and_labels("t3")
    n_clean = sum(labels)
    n_noisy = len(labels) - n_clean
    print(f"\n  {len(paths)} fichiers | clean(1)={n_clean} | noisy(0)={n_noisy}")

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
    best_val_loss    = float("inf")
    patience_counter = 0
    patience         = 10
    history          = []

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val Acc':>9} "
          f"{'Val Rec':>9} {'LR':>10}")
    print("-" * 62)

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

        acc = (all_preds == all_labels_val).mean()
        tp  = ((all_preds == 1) & (all_labels_val == 1)).sum()
        fn  = ((all_preds == 0) & (all_labels_val == 1)).sum()
        rec = tp / (tp + fn + 1e-8)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_acc": float(acc), "val_recall": float(rec),
        })

        ok_acc = "ok" if acc >= TARGET_ACC else "x"
        ok_rec = "ok" if rec >= TARGET_REC else " "
        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
              f"{acc*100:>8.1f}% {ok_acc} {rec*100:>8.1f}% {ok_rec} {current_lr:>10.6f}")

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss":        val_loss,
                "val_acc":         float(acc),
                "val_recall":      float(rec),
                "model_name":      model_name,
                "task":            "t3_current_sufficient",
            }, out_dir / "best_model.pt")
            print(f"         meilleur modele sauvegarde (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  early stopping epoch {epoch} (patience={patience})")
                break

    torch.save({"epoch": epoch, "model_state": model.state_dict()}, out_dir / "last_model.pt")

    results = {
        "task":           "t3_current_sufficient",
        "model":          model_name,
        "best_val_loss":  float(best_val_loss),
        "best_val_acc":   float(max(h["val_acc"]    for h in history)),
        "best_val_rec":   float(max(h["val_recall"] for h in history)),
        "epochs_trained": epoch,
        "history":        history,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    best_skipper = SKIPPER_W_ACC * results["best_val_acc"] + SKIPPER_W_REC * results["best_val_rec"]
    print(f"\n  best val_acc    : {results['best_val_acc']*100:.1f}%  (objectif: {TARGET_ACC*100:.0f}%)")
    print(f"  best val_recall : {results['best_val_rec']*100:.1f}%")
    print(f"  score SKIPPER   : {best_skipper*100:.1f}%  (70%*Acc + 30%*Rec)")
    print(f"  checkpoints     : {out_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train — Tache 3 : current_sufficient")
    parser.add_argument("--mode",       type=str, default="baseline",
                        choices=["baseline", "cnn"])
    parser.add_argument("--model",      type=str, default="cnn",
                        choices=["cnn", "densenet"])
    parser.add_argument("--data_dir",   type=str, default="data/raw")
    parser.add_argument("--out_dir",    type=str, default="task3/checkpoints")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.2)
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir  = ROOT / args.out_dir

    if args.mode == "baseline":
        run_baseline(data_dir, ROOT / "task3/results")
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
