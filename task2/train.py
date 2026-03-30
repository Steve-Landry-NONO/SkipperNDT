"""
task2/train.py

Entrainement — Tache 2 : Prediction de la largeur de carte magnetique (map_width).

Objectif SKIPPER : MAE < 1 metre

Note importante : l'algorithme geometrique (scripts/geometric_width.py) atteint
MAE = 0.61m sans entrainement. Ce script CNN est conserve pour comparaison mais
n'est pas la methode recommandee pour la production.

Raison de l'echec du deep learning sur T2 : map_width est une propriete geometrique
de la carte de vol (largeur du couloir survole par le drone), pas une caracteristique
statistique du signal magnetique. Aucun CNN ne peut l'inferer de facon fiable.

Dataset : uniquement les fichiers avec label=1 (pipe present), soit ~1751 fichiers.
Distribution : 2.01m a 154.84m, moyenne 36.9m, tres asymetrique.
-> Entrainement en log-space (log1p/expm1) pour symetriser la distribution.

Usage :
    python task2/train.py --csv data/pipe_presence_width_detection_label.csv \\
                          --data_dir data/raw --mode cnn --epochs 50 --batch_size 32

Sorties :
    task2/checkpoints/best_model.pt
    task2/results/metrics.json
    task2/results/baseline_results.json

Auteur(s) : MAKAMTA Linda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def run_baseline(csv_path: Path, data_dir: Path, out_dir: Path) -> dict:
    """
    Baseline ML par cross-validation 5-fold sur features statistiques.

    L'entrainement et l'evaluation sont faits en log-space.
    Le MAE est converti en metres (expm1) pour l'affichage.
    """
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import make_scorer, mean_absolute_error
    import pickle

    from src.preprocessing.features import extract_features_batch

    print(f"\n{'='*60}")
    print(f"  Baseline ML — Tache 2 : map_width (regression)")
    print(f"{'='*60}")

    df       = pd.read_csv(csv_path, sep=";")
    df_pipe  = df[(df["label"] == 1) & df["width_m"].notna()].copy()
    df_synth = df_pipe[~df_pipe["field_file"].str.startswith("real")].copy()

    print(f"\n  {len(df_synth)} fichiers synthetiques avec width_m")
    print(f"  width_m : min={df_synth['width_m'].min():.1f}m  "
          f"max={df_synth['width_m'].max():.1f}m  mean={df_synth['width_m'].mean():.1f}m")

    file_index = {p.name: p for p in data_dir.rglob("*.npz")}
    print(f"  {len(file_index)} fichiers .npz indexes")

    paths, widths = [], []
    for _, row in df_synth.iterrows():
        fname = row["field_file"]
        if fname in file_index:
            paths.append(file_index[fname])
            widths.append(row["width_m"])
    print(f"  {len(paths)} fichiers matches avec le CSV")

    print(f"\n  extraction des features ({len(paths)} fichiers)...")
    t0 = time.time()
    X, _ = extract_features_batch(paths, [0] * len(paths), verbose=True)
    y     = np.array(widths)
    y_log = np.log1p(y)
    print(f"  {X.shape} features en {time.time()-t0:.0f}s")

    def mae_metres_score(y_true_log, y_pred_log):
        return -mean_absolute_error(np.expm1(y_true_log), np.expm1(y_pred_log))

    scorer = make_scorer(mae_metres_score)

    models = {
        "SVR_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=42)),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg",    GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42
            )),
        ]),
    }

    cv      = KFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print(f"\n  evaluation cross-validation (5 folds)\n")
    print(f"  {'Modele':<22} {'MAE (m)':>10} {'Objectif':>10}")
    print("  " + "-"*44)

    best_mae  = float("inf")
    best_name = ""
    best_pipe = None

    for name, pipe in models.items():
        t0      = time.time()
        cv_res  = cross_validate(pipe, X, y_log, cv=cv, scoring=scorer, n_jobs=-1)
        mae     = -cv_res["test_score"].mean()
        elapsed = time.time() - t0

        results[name] = {"mae_m": float(mae), "time_s": float(elapsed)}

        ok = "ok" if mae < 1.0 else "x"
        print(f"  {name:<22} {mae:>8.2f}m {ok}   <1m  ({elapsed:.0f}s)")

        if mae < best_mae:
            best_mae  = mae
            best_name = name
            best_pipe = pipe

    print(f"\n  meilleur modele : {best_name} (MAE = {best_mae:.2f}m)")

    print(f"\n  entrainement final sur 100% des donnees synthetiques...")
    best_pipe.fit(X, y_log)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "baseline_best.pkl", "wb") as f:
        pickle.dump({
            "model":     best_pipe,
            "task":      "t2_map_width",
            "log_space": True,
        }, f)
    print(f"  modele sauvegarde : {out_dir / 'baseline_best.pkl'}")

    summary = {
        "task":       "t2_map_width",
        "best_model": best_name,
        "best_mae_m": float(best_mae),
        "models":     results,
    }
    with open(out_dir / "baseline_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def run_cnn(
    csv_path:   Path,
    data_dir:   Path,
    out_dir:    Path,
    model_name: str   = "cnn",
    epochs:     int   = 30,
    batch_size: int   = 16,
    lr:         float = 1e-3,
    val_split:  float = 0.2,
) -> dict:
    """
    Entrainement CNN de regression pour la Tache 2.

    Loss Huber (delta=1.0) pour la robustesse aux grandes largeurs.
    Le checkpoint est sauvegarde sur le meilleur MAE de validation (en metres).
    Les 51 donnees reelles labelisees sont evaluees separement apres l'entrainement.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("PyTorch non disponible.")
        return {}

    from src.models.cnn_task2 import get_regressor, count_params
    from src.models.dataset_regression import RegressionDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  CNN Regression — Tache 2 : map_width | modele: {model_name}")
    print(f"  device: {device}")
    print(f"{'='*60}")

    df       = pd.read_csv(csv_path, sep=";")
    df_pipe  = df[(df["label"] == 1) & df["width_m"].notna()].copy()
    df_synth = df_pipe[~df_pipe["field_file"].str.startswith("real")].copy()
    df_real  = df_pipe[df_pipe["field_file"].str.startswith("real")].copy()

    def find_path(filename: str) -> Path:
        for p in data_dir.rglob(filename):
            return p
        return data_dir / filename

    synth_paths  = [find_path(f) for f in df_synth["field_file"]]
    synth_widths = df_synth["width_m"].tolist()
    valid        = [(p, w) for p, w in zip(synth_paths, synth_widths) if p.exists()]
    synth_paths  = [v[0] for v in valid]
    synth_widths = [v[1] for v in valid]

    real_paths  = [find_path(f) for f in df_real["field_file"]]
    real_widths = df_real["width_m"].tolist()
    real_valid  = [(p, w) for p, w in zip(real_paths, real_widths) if p.exists()]
    real_paths  = [v[0] for v in real_valid]
    real_widths = [v[1] for v in real_valid]

    print(f"\n  synthetiques : {len(synth_paths)} | reels : {len(real_paths)}")
    print(f"  width_m : min={min(synth_widths):.1f}m  max={max(synth_widths):.1f}m  "
          f"mean={np.mean(synth_widths):.1f}m")

    idx_train, idx_val = train_test_split(
        range(len(synth_paths)), test_size=val_split, random_state=42
    )

    train_ds = RegressionDataset(
        [synth_paths[i] for i in idx_train],
        [synth_widths[i] for i in idx_train],
        augment=True,
    )
    val_ds = RegressionDataset(
        [synth_paths[i] for i in idx_val],
        [synth_widths[i] for i in idx_val],
        augment=False,
    )

    nw = 0
    pm = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=nw, pin_memory=pm)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=nw, pin_memory=pm)

    print(f"  train: {len(train_ds)} | val: {len(val_ds)}")

    model     = get_regressor(model_name).to(device)
    print(f"  parametres : {count_params(model):,}")

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_mae_val     = float("inf")
    patience_counter = 0
    patience         = 10
    history          = []

    print(f"\n{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val MAE':>10} {'LR':>10}")
    print("-" * 54)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y_batch in train_loader:
            x, y_batch = x.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)
        train_loss /= len(train_ds)

        model.eval()
        val_loss    = 0.0
        all_preds   = []
        all_targets = []
        with torch.no_grad():
            for x, y_batch in val_loader:
                x, y_batch = x.to(device), y_batch.to(device)
                pred        = model(x)
                val_loss   += criterion(pred, y_batch).item() * len(x)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())

        val_loss /= len(val_ds)
        mae_m     = float(np.abs(
            np.expm1(np.array(all_preds)) - np.expm1(np.array(all_targets))
        ).mean())

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            "val_mae_m": mae_m,
        })

        ok = "ok" if mae_m < 1.0 else "x"
        print(f"{epoch:>6} {train_loss:>12.4f} {val_loss:>10.4f} "
              f"{mae_m:>8.2f}m {ok}  {current_lr:>10.6f}")

        if mae_m < best_mae_val:
            best_mae_val     = mae_m
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_mae_m":   mae_m,
                "val_loss":    val_loss,
                "model_name":  model_name,
                "task":        "t2_map_width",
                "log_space":   True,
            }, out_dir / "best_model.pt")
            print(f"         meilleur modele sauvegarde (MAE={mae_m:.2f}m)")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  early stopping epoch {epoch} (patience={patience})")
                break

    torch.save({"epoch": epoch, "model_state": model.state_dict()}, out_dir / "last_model.pt")

    # Test final sur les donnees reelles
    real_mae = None
    if real_paths:
        print(f"\n  test sur {len(real_paths)} donnees reelles...")
        real_ds     = RegressionDataset(real_paths, real_widths, augment=False)
        real_loader = DataLoader(real_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0, pin_memory=False)

        ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        all_preds_r = []
        all_tgts_r  = []
        with torch.no_grad():
            for x, y_batch in real_loader:
                all_preds_r.extend(model(x.to(device)).cpu().numpy())
                all_tgts_r.extend(y_batch.numpy())

        preds_r_m = np.expm1(np.array(all_preds_r))
        tgts_r_m  = np.expm1(np.array(all_tgts_r))
        real_mae  = float(np.abs(preds_r_m - tgts_r_m).mean())

        ok_r = "ok" if real_mae < 1.0 else "x"
        print(f"  MAE sur reels : {real_mae:.2f}m {ok_r}  (objectif: <1m)")

        for fname, pred_m, true_m in zip([p.name for p in real_paths], preds_r_m, tgts_r_m):
            err = abs(pred_m - true_m)
            print(f"  {fname:<40} pred={pred_m:6.1f}m  true={true_m:6.1f}m  err={err:5.1f}m")

    results = {
        "task":           "t2_map_width",
        "model":          model_name,
        "best_val_mae_m": float(best_mae_val),
        "real_mae_m":     real_mae,
        "epochs_trained": epoch,
        "history":        history,
    }

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    ok = "OK" if best_mae_val < 1.0 else "KO"
    print(f"\n  {ok} best val MAE : {best_mae_val:.2f}m  (objectif: <1m)")
    if real_mae:
        print(f"  MAE reels : {real_mae:.2f}m")
    print(f"  checkpoints : {out_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train — Tache 2 : map_width")
    parser.add_argument("--mode",       type=str, default="cnn",
                        choices=["baseline", "cnn"])
    parser.add_argument("--model",      type=str, default="cnn",
                        choices=["cnn", "densenet"])
    parser.add_argument("--csv",        type=str,
                        default="data/pipe_presence_width_detection_label.csv")
    parser.add_argument("--data_dir",   type=str, default="data/raw")
    parser.add_argument("--out_dir",    type=str, default="task2/checkpoints")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=16)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_split",  type=float, default=0.2)
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    data_dir = ROOT / args.data_dir
    out_dir  = ROOT / args.out_dir

    if not csv_path.exists():
        print(f"CSV introuvable : {csv_path}")
        print(f"placer pipe_presence_width_detection_label.csv dans data/")
        return

    if args.mode == "baseline":
        run_baseline(csv_path, data_dir, ROOT / "task2/results")
    else:
        run_cnn(
            csv_path, data_dir, out_dir,
            model_name = args.model,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            lr         = args.lr,
            val_split  = args.val_split,
        )


if __name__ == "__main__":
    main()
