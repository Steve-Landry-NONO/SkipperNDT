"""
scripts/geometric_width.py

Estimation de map_width par projection perpendiculaire le long de la trace du pipe.

Principe : le pipe correspond au minimum local de Bz sur chaque ligne de la carte.
En suivant cette trace et en projetant perpendiculairement a la tangente locale,
on atteint les bords NaN de la carte (limites de la trajectoire de vol).
La distance parcourue dans les deux sens donne la largeur locale.
La mediane de l'ensemble des mesures est retournee comme estimation finale.

Resultats sur 51 donnees reelles labelisees :
    MAE = 0.61m  (objectif SKIPPER : < 1m)
    Mediane erreur = 0.21m
    < 1m : 92%  |  < 5m : 98%

Usage :
    python scripts/geometric_width.py --file data/raw/real_data/real_data_00000.npz
    python scripts/geometric_width.py \\
        --csv data/pipe_presence_width_detection_label.csv \\
        --data_dir data/raw --reals_only

Auteur(s) : KOUOKAM NONO Steve Landry
"""

import argparse
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Etape 1 : Detection de la trace du pipe
# ─────────────────────────────────────────────────────────────────────────────

def find_pipe_trace(bz: np.ndarray, smooth_sigma: float = 3.0) -> np.ndarray:
    """
    Extrait la trace du pipe comme la sequence des minima de Bz ligne par ligne.

    Un lissage gaussien (sigma=3) est applique avant la recherche du minimum
    pour eviter les faux minima dus au bruit capteur. Les lignes avec moins de
    10 pixels valides sont ignorees (NaN retourne).

    Args:
        bz           : composante verticale du champ magnetique, shape (H, W)
        smooth_sigma : ecart-type du lissage gaussien pre-detection

    Returns:
        trace : array (H,) — colonne du minimum pour chaque ligne, NaN si invalide
    """
    H, W = bz.shape
    trace = np.full(H, np.nan)

    for row in range(H):
        profile = bz[row, :]
        valid   = np.isfinite(profile)

        if valid.sum() < 10:
            continue

        profile_filled = profile.copy()
        profile_filled[~valid] = np.nanmean(profile)

        if smooth_sigma > 0:
            profile_smooth = ndimage.gaussian_filter1d(profile_filled, sigma=smooth_sigma)
            profile_smooth[~valid] = np.nan
        else:
            profile_smooth = profile.copy()

        trace[row] = np.nanargmin(profile_smooth)

    return trace


def smooth_trace(trace: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Lisse la trace du pipe par moyenne glissante pour reduire le bruit de detection.

    Args:
        trace  : array (H,) issu de find_pipe_trace
        window : demi-fenetre de la moyenne glissante (en lignes)

    Returns:
        trace lissee, meme shape que l'entree
    """
    smoothed = trace.copy()
    rows     = np.where(np.isfinite(trace))[0]

    if len(rows) < 2 * window:
        return smoothed

    kernel      = np.ones(window) / window
    smooth_vals = np.convolve(trace[rows], kernel, mode="same")
    smoothed[rows] = smooth_vals

    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# Etape 2 : Mesure par projection perpendiculaire
# ─────────────────────────────────────────────────────────────────────────────

def walk_perpendicular(
    bz: np.ndarray,
    row: int,
    col: float,
    slope_perp: float,
    direction: int,
    step: float = 0.5,
) -> float:
    """
    Marche le long d'une direction perpendiculaire depuis (row, col) jusqu'au
    premier pixel NaN ou jusqu'au bord de l'image.

    Le cap est fixe a la diagonale de l'image (borne physique maximale) pour
    eviter un plafonnement artificiel sur les grandes cartes sans NaN en bordure.

    Args:
        bz         : composante Bz, shape (H, W)
        row, col   : point de depart de la marche
        slope_perp : pente de la direction perpendiculaire (espace image row/col)
        direction  : +1 ou -1
        step       : pas en pixels

    Returns:
        distance en pixels jusqu'au dernier pixel valide avant NaN/bord
    """
    H, W      = bz.shape
    max_steps = int(np.sqrt(H**2 + W**2) / step) + 1

    if abs(slope_perp) > 1e6:
        dx, dy = 0.0, direction * step
    else:
        norm = np.sqrt(1 + slope_perp**2)
        dx   = direction * step / norm
        dy   = direction * slope_perp * step / norm

    x, y      = float(col), float(row)
    last_dist = 0.0

    for _ in range(max_steps):
        x += dx
        y += dy
        xi, yi = int(round(x)), int(round(y))

        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            break
        if np.isnan(bz[yi, xi]):
            break

        last_dist = np.sqrt((x - col)**2 + (y - row)**2)

    return last_dist


def measure_width_at_point(
    bz: np.ndarray,
    row: int,
    col: float,
    slope_tangent: float,
) -> float:
    """
    Mesure la largeur de la carte au point (row, col) en projetant dans la
    direction perpendiculaire a la tangente locale du pipe.

    Args:
        bz             : composante Bz, shape (H, W)
        row, col       : point sur la trace lissee
        slope_tangent  : pente locale de la trace (Delta_row / Delta_col)

    Returns:
        largeur locale en metres (pixels * 0.2m/px)
    """
    if abs(slope_tangent) < 1e-8:
        slope_perp = 1e8
    elif abs(slope_tangent) > 1e6:
        slope_perp = 0.0
    else:
        slope_perp = -1.0 / slope_tangent

    dist_pos = walk_perpendicular(bz, row, col, slope_perp, direction=+1)
    dist_neg = walk_perpendicular(bz, row, col, slope_perp, direction=-1)

    return (dist_pos + dist_neg) * 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Algorithme principal
# ─────────────────────────────────────────────────────────────────────────────

def estimate_map_width(
    arr: np.ndarray,
    step_rows: int = 5,
    smooth_sigma: float = 3.0,
    trace_smooth_window: int = 10,
    verbose: bool = False,
) -> dict:
    """
    Estime map_width par projection perpendiculaire le long de la trace du pipe.

    Args:
        arr                  : carte magnetique (H, W, 4) float32
        step_rows            : espacement en lignes entre les points de mesure
        smooth_sigma         : lissage gaussien du profil Bz avant detection du min
        trace_smooth_window  : fenetre de lissage de la trace

    Returns:
        dict avec les cles :
            map_width_m  : mediane des largeurs locales en metres (None si echec)
            n_measures   : nombre de mesures realisees
            widths_m     : liste des largeurs individuelles
            trace        : trace lissee (array H,)
    """
    bz = arr[:, :, 2]
    H, W = bz.shape

    trace = find_pipe_trace(bz, smooth_sigma=smooth_sigma)
    trace = smooth_trace(trace, window=trace_smooth_window)

    valid_rows = np.where(np.isfinite(trace))[0]
    if len(valid_rows) < 4:
        return {"map_width_m": None, "n_measures": 0, "widths_m": [], "trace": trace}

    if verbose:
        print(f"  trace : {len(valid_rows)} lignes valides / {H}")
        print(f"  colonnes : min={np.nanmin(trace):.0f}  max={np.nanmax(trace):.0f}")

    widths        = []
    measure_rows  = valid_rows[::step_rows]

    for i in range(len(measure_rows) - 1):
        r1, r2 = measure_rows[i], measure_rows[i + 1]
        c1, c2 = trace[r1], trace[r2]

        if np.isnan(c1) or np.isnan(c2):
            continue

        dr = float(r2 - r1)
        dc = float(c2 - c1)
        slope_tangent = 1e8 if abs(dc) < 1e-8 else dr / dc

        row_mid = (r1 + r2) // 2
        col_mid = (c1 + c2) / 2

        w = measure_width_at_point(bz, row_mid, col_mid, slope_tangent)
        if w > 0:
            widths.append(w)

    if not widths:
        return {"map_width_m": None, "n_measures": 0, "widths_m": [], "trace": trace}

    widths    = np.array(widths)
    map_width = float(np.median(widths))

    if verbose:
        print(f"  {len(widths)} mesures | min={widths.min():.1f}m  max={widths.max():.1f}m  "
              f"mediane={map_width:.1f}m  std={widths.std():.1f}m")

    return {
        "map_width_m": map_width,
        "n_measures":  len(widths),
        "widths_m":    widths.tolist(),
        "trace":       trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation sur le dataset
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    csv_path: Path,
    data_dir: Path,
    n_samples: int = 100,
    reals_only: bool = False,
    all_samples: bool = False,
):
    """
    Evalue l'algorithme geometrique sur un sous-ensemble du dataset labellise.

    Args:
        csv_path   : chemin vers pipe_presence_width_detection_label.csv
        data_dir   : dossier racine des .npz
        n_samples  : nombre de fichiers a evaluer (ignore si all_samples=True)
        reals_only : evaluer uniquement sur les donnees reelles
        all_samples: evaluer sur tout le dataset
    """
    df = pd.read_csv(csv_path, sep=";")

    if reals_only:
        df_pipe = df[
            (df["label"] == 1)
            & df["width_m"].notna()
            & df["field_file"].str.startswith("real")
        ].copy()
        print(f"  mode : donnees reelles uniquement ({len(df_pipe)} fichiers labelises)")
    else:
        df_pipe = df[
            (df["label"] == 1)
            & df["width_m"].notna()
            & ~df["field_file"].str.startswith("real")
        ].copy()
        print(f"  mode : donnees synthetiques ({len(df_pipe)} fichiers disponibles)")

    if not all_samples:
        df_pipe = df_pipe.sample(min(n_samples, len(df_pipe)), random_state=42)

    file_index = {p.name: p for p in data_dir.rglob("*.npz")}
    print(f"  {len(file_index)} fichiers indexes")

    errors = []
    n_ok   = 0

    print(f"\n  {'Fichier':<45} {'true':>7} {'pred':>7} {'err':>7}")
    print("  " + "-" * 68)

    for _, row in df_pipe.iterrows():
        fname = row["field_file"]
        if fname not in file_index:
            continue

        arr    = np.load(file_index[fname])["data"].astype("float32")
        true_w = row["width_m"]
        result = estimate_map_width(arr, step_rows=5, verbose=False)
        pred_w = result["map_width_m"]

        if pred_w is not None:
            err = abs(pred_w - true_w)
            errors.append(err)
            ok  = "ok" if err < 1.0 else ("~" if err < 5.0 else "x")
            print(f"  {fname[:45]:<45} {true_w:>6.1f}m {pred_w:>6.1f}m {err:>6.1f}m {ok}")
            n_ok += 1
        else:
            print(f"  {fname[:45]:<45} {true_w:>6.1f}m {'N/A':>7}")

    if errors:
        errors = np.array(errors)
        print(f"\n  fichiers traites : {n_ok}/{len(df_pipe)}")
        print(f"  MAE    : {errors.mean():.2f}m")
        print(f"  Mediane: {np.median(errors):.2f}m")
        print(f"  < 1m   : {(errors < 1.0).sum()}/{n_ok} ({100*(errors<1.0).mean():.0f}%)")
        print(f"  < 5m   : {(errors < 5.0).sum()}/{n_ok} ({100*(errors<5.0).mean():.0f}%)")
        print(f"  < 10m  : {(errors < 10.0).sum()}/{n_ok} ({100*(errors<10.0).mean():.0f}%)")
        print(f"  objectif SKIPPER : MAE < 1m")

    return errors


def main():
    parser = argparse.ArgumentParser(description="Mesure geometrique map_width")
    parser.add_argument("--csv",        default="data/pipe_presence_width_detection_label.csv")
    parser.add_argument("--data_dir",   default="data/raw")
    parser.add_argument("--n_samples",  type=int, default=100)
    parser.add_argument("--file",       default=None, help="Tester sur un fichier unique")
    parser.add_argument("--reals_only", action="store_true")
    parser.add_argument("--all",        action="store_true")
    args = parser.parse_args()

    if args.file:
        arr    = np.load(args.file)["data"].astype("float32")
        result = estimate_map_width(arr, verbose=True)
        print(f"\nmap_width estimee : {result['map_width_m']:.2f}m")
        print(f"base sur {result['n_measures']} mesures perpendiculaires")
    else:
        print(f"\n{'='*60}")
        print(f"  Mesure geometrique map_width")
        print(f"  methode : projection perpendiculaire le long de la trace")
        print(f"{'='*60}\n")
        evaluate(
            ROOT / args.csv,
            ROOT / args.data_dir,
            n_samples=args.n_samples,
            reals_only=args.reals_only,
            all_samples=args.all,
        )


if __name__ == "__main__":
    main()
