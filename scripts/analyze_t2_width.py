"""
scripts/analyze_t2_width.py

Analyse comparative des approches geometriques pour predire map_width.

Compare cinq methodes de mesure :
    - W * 0.2  : largeur de l'image en pixels convertie en metres
    - H * 0.2  : hauteur de l'image
    - min(H,W) * 0.2
    - FWHM Bz  : demi-largeur a mi-hauteur du profil transversal moyen
    - Seuil 20%: largeur ou |Bz| >= 20% du max

Sert a justifier le choix de l'algorithme geometrique perpendiculaire
par rapport aux approches plus naives.

Usage :
    python scripts/analyze_t2_width.py \\
        --csv data/pipe_presence_width_detection_label.csv \\
        --data_dir data/raw

Auteur(s) : KOUOKAM NONO Steve Landry
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def analyze(csv_path: Path, data_dir: Path, n_samples: int = 100):
    df      = pd.read_csv(csv_path, sep=";")
    df_pipe = df[(df["label"] == 1) & df["width_m"].notna()].copy()
    df_synth = df_pipe[~df_pipe["field_file"].str.startswith("real")]

    print(f"  analyse sur {min(n_samples, len(df_synth))} fichiers synthetiques\n")

    file_index = {p.name: p for p in data_dir.rglob("*.npz")}

    results = []
    for _, row in df_synth.head(n_samples).iterrows():
        fname = row["field_file"]
        if fname not in file_index:
            continue

        arr    = np.load(file_index[fname])["data"].astype("float32")
        H, W, _ = arr.shape
        true_w = row["width_m"]

        min_dim_m = min(H, W) * 0.2
        w_m       = W * 0.2
        h_m       = H * 0.2

        bz      = arr[:, :, 2]
        profile = np.nanmean(np.abs(bz), axis=0)
        valid   = ~np.isnan(profile)
        if valid.sum() > 0:
            p_clean = np.where(valid, profile, 0.0)
            p_max   = p_clean.max()
            if p_max > 0:
                fwhm_cols = np.where(p_clean >= 0.5 * p_max)[0]
                fwhm_m    = (
                    (fwhm_cols[-1] - fwhm_cols[0]) * 0.2 if len(fwhm_cols) > 1 else 0.0
                )
                cols_20   = np.where(p_clean >= 0.20 * p_max)[0]
                w20_m     = (
                    (cols_20[-1] - cols_20[0]) * 0.2 if len(cols_20) > 1 else 0.0
                )
            else:
                fwhm_m = w20_m = 0.0
        else:
            fwhm_m = w20_m = 0.0

        results.append({
            "file":      fname[:45],
            "true_w":    true_w,
            "W_m":       w_m,
            "H_m":       h_m,
            "min_dim_m": min_dim_m,
            "fwhm_m":    fwhm_m,
            "w20_m":     w20_m,
            "err_W":     abs(w_m - true_w),
            "err_H":     abs(h_m - true_w),
            "err_min":   abs(min_dim_m - true_w),
            "err_fwhm":  abs(fwhm_m - true_w),
            "err_w20":   abs(w20_m - true_w),
        })

    df_r = pd.DataFrame(results)
    if len(df_r) == 0:
        print("  aucun fichier trouve — verifier data_dir")
        return

    print("=" * 65)
    print(f"  {'Approche':<30} {'MAE (m)':>10} {'Corr':>8}")
    print("-" * 65)

    approaches = {
        "W * 0.2 (largeur image)":  ("err_W",    "W_m"),
        "H * 0.2 (hauteur image)":  ("err_H",    "H_m"),
        "min(H,W) * 0.2":           ("err_min",  "min_dim_m"),
        "FWHM Bz (seuil 50%)":      ("err_fwhm", "fwhm_m"),
        "Largeur Bz (seuil 20%)":   ("err_w20",  "w20_m"),
    }

    best_mae  = float("inf")
    best_name = ""
    for name, (err_col, val_col) in approaches.items():
        mae  = df_r[err_col].mean()
        corr = df_r[val_col].corr(df_r["true_w"])
        ok   = "ok" if mae < 1.0 else ("~" if mae < 5.0 else "x")
        print(f"  {name:<30} {mae:>8.2f}m {ok} {corr:>7.3f}")
        if mae < best_mae:
            best_mae  = mae
            best_name = name

    print("=" * 65)
    print(f"\n  meilleure approche : {best_name} (MAE={best_mae:.2f}m)")

    print("\n  detail 10 premiers fichiers :")
    print(f"  {'Fichier':<45} {'true':>7} {'W*0.2':>7} {'FWHM':>7} {'W20%':>7}")
    print("  " + "-" * 73)
    for _, r in df_r.head(10).iterrows():
        print(f"  {r['file']:<45} {r['true_w']:>6.1f}m {r['W_m']:>6.1f}m "
              f"{r['fwhm_m']:>6.1f}m {r['w20_m']:>6.1f}m")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",       default="data/pipe_presence_width_detection_label.csv")
    parser.add_argument("--data_dir",  default="data/raw")
    parser.add_argument("--n_samples", type=int, default=100)
    args = parser.parse_args()

    analyze(ROOT / args.csv, ROOT / args.data_dir, args.n_samples)


if __name__ == "__main__":
    main()
