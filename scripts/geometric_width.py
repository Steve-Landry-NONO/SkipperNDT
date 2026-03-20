"""
scripts/geometric_width.py
Mesure géométrique de map_width par projection perpendiculaire le long de la trace du pipe.

Algorithme (inspiré du calcul différentiel) :
  1. Trouver la trace du pipe : minimum de Bz ligne par ligne → squelette du pipe
  2. Lisser la trace pour avoir des points réguliers
  3. Pour chaque couple de points voisins (p1, p2) :
       - Calculer la tangente : pente a = (y2-y1)/(x2-x1)
       - Calculer la perpendiculaire : pente_perp = -1/a
       - Depuis p1, marcher le long de la perpendiculaire dans les deux directions
       - S'arrêter au dernier pixel valide avant NaN ou bord
       - Largeur locale = distance gauche + distance droite
  4. map_width = médiane des largeurs locales (robuste aux outliers)

Usage :
    python scripts/geometric_width.py
    python scripts/geometric_width.py --csv data/pipe_presence_width_detection_label.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
import argparse
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 1 : Trouver la trace du pipe (squelette)
# ─────────────────────────────────────────────────────────────────────────────

def find_pipe_trace(bz: np.ndarray, smooth_sigma: float = 3.0) -> np.ndarray:
    """
    Trouve la trace du pipe = colonne du minimum Bz pour chaque ligne.
    
    Le pipe correspond au minimum de Bz (creux dans le signal).
    On lisse d'abord pour éviter les faux minima dus au bruit.
    
    Returns:
        trace : array (H,) — colonne du minimum pour chaque ligne valide
                NaN si la ligne n'a pas assez de pixels valides
    """
    H, W = bz.shape
    trace = np.full(H, np.nan)
    
    for row in range(H):
        profile = bz[row, :]
        valid   = np.isfinite(profile)
        n_valid = valid.sum()
        
        # Ignorer les lignes avec moins de 10 pixels valides
        if n_valid < 10:
            continue
        
        # Remplir les NaN avec la moyenne locale pour le lissage
        profile_filled = profile.copy()
        profile_filled[~valid] = np.nanmean(profile)
        
        # Lissage gaussien pour réduire le bruit
        if smooth_sigma > 0:
            profile_smooth = ndimage.gaussian_filter1d(profile_filled, sigma=smooth_sigma)
            # Remettre les NaN aux bons endroits
            profile_smooth[~valid] = np.nan
        else:
            profile_smooth = profile.copy()
        
        # Minimum sur les pixels valides uniquement
        trace[row] = np.nanargmin(profile_smooth)
    
    return trace


def smooth_trace(trace: np.ndarray, window: int = 10) -> np.ndarray:
    """
    Lisse la trace du pipe par moyenne glissante.
    Élimine les sauts brusques dus au bruit.
    """
    smoothed = trace.copy()
    valid    = np.isfinite(trace)
    rows     = np.where(valid)[0]
    
    if len(rows) < 2 * window:
        return smoothed
    
    # Moyenne glissante sur les points valides
    trace_valid = trace[rows]
    kernel      = np.ones(window) / window
    smooth_vals = np.convolve(trace_valid, kernel, mode='same')
    smoothed[rows] = smooth_vals
    
    return smoothed


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE 2 : Mesure de largeur par perpendiculaires
# ─────────────────────────────────────────────────────────────────────────────

def walk_perpendicular(bz: np.ndarray, row: int, col: float,
                       slope_perp: float, direction: int,
                       step: float = 0.5) -> float:
    """
    Marche le long d'une perpendiculaire depuis (row, col) dans une direction.
    S'arrête au premier pixel NaN (bord de la carte) ou bord image.

    max_steps est calé sur la diagonale de l'image pour éviter un cap artificiel
    (le bug 200m = 2000 × 0.5px × 0.2m qui plafonnait les grandes cartes sans NaN).

    Returns:
        distance en pixels jusqu'au dernier pixel valide avant NaN/bord
    """
    H, W = bz.shape
    # Cap = diagonale entière de l'image (jamais plus loin physiquement)
    max_steps = int(np.sqrt(H**2 + W**2) / step) + 1

    if abs(slope_perp) > 1e6:  # quasi-vertical
        dx = 0.0
        dy = direction * step
    else:
        norm = np.sqrt(1 + slope_perp**2)
        dx   = direction * step / norm
        dy   = direction * slope_perp * step / norm

    x, y      = float(col), float(row)
    last_dist = 0.0

    for i in range(max_steps):
        x += dx
        y += dy

        xi, yi = int(round(x)), int(round(y))

        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            break

        if np.isnan(bz[yi, xi]):
            break

        last_dist = np.sqrt((x - col)**2 + (y - row)**2)

    return last_dist


def measure_width_at_point(bz: np.ndarray, row: int, col: float,
                            slope_tangent: float) -> float:
    """
    Mesure la largeur de la carte au point (row, col) en projetant
    perpendiculairement à la tangente locale du pipe.
    
    Si la tangente est quasi-horizontale (pipe rectiligne horizontal),
    la perpendiculaire est quasi-verticale → on marche vers le haut/bas.
    Si la tangente est diagonale (pipe courbe), la perpendiculaire tourne aussi.
    """
    # Pente perpendiculaire : pente_perp = -1 / pente_tangente
    if abs(slope_tangent) < 1e-8:
        # Tangente horizontale → perpendiculaire verticale
        slope_perp = 1e8
    elif abs(slope_tangent) > 1e6:
        # Tangente verticale → perpendiculaire horizontale
        slope_perp = 0.0
    else:
        slope_perp = -1.0 / slope_tangent
    
    # Marcher dans les deux directions perpendiculaires
    dist_pos = walk_perpendicular(bz, row, col, slope_perp, direction=+1)
    dist_neg = walk_perpendicular(bz, row, col, slope_perp, direction=-1)
    
    return (dist_pos + dist_neg) * 0.2  # convertir pixels → mètres


# ─────────────────────────────────────────────────────────────────────────────
# ALGORITHME PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def estimate_map_width(arr: np.ndarray,
                       step_rows: int = 5,
                       smooth_sigma: float = 3.0,
                       trace_smooth_window: int = 10,
                       verbose: bool = False) -> dict:
    """
    Estime map_width par projection perpendiculaire le long de la trace du pipe.
    
    Args:
        arr              : (H, W, 4) float32
        step_rows        : espacement entre les points de mesure (en lignes)
        smooth_sigma     : lissage gaussien du profil Bz avant détection du min
        trace_smooth_window : fenêtre de lissage de la trace
    
    Returns:
        dict avec 'map_width_m', 'n_measures', 'widths_m', 'trace'
    """
    bz = arr[:, :, 2]   # Canal Bz
    H, W = bz.shape
    
    # ── Étape 1 : Trace du pipe ───────────────────────────────
    trace = find_pipe_trace(bz, smooth_sigma=smooth_sigma)
    trace = smooth_trace(trace, window=trace_smooth_window)
    
    valid_rows = np.where(np.isfinite(trace))[0]
    if len(valid_rows) < 4:
        return {'map_width_m': None, 'n_measures': 0, 'widths_m': [], 'trace': trace}
    
    if verbose:
        print(f"  Trace trouvée sur {len(valid_rows)} lignes / {H}")
        print(f"  Colonnes trace : min={np.nanmin(trace):.0f}  max={np.nanmax(trace):.0f}")
    
    # ── Étape 2 : Mesures perpendiculaires ───────────────────
    widths = []
    
    # Prendre des couples de points espacés de step_rows
    # (assez proches pour approximer une droite → tangente)
    measure_rows = valid_rows[::step_rows]
    
    for i in range(len(measure_rows) - 1):
        r1 = measure_rows[i]
        r2 = measure_rows[i + 1]
        c1 = trace[r1]
        c2 = trace[r2]
        
        if np.isnan(c1) or np.isnan(c2):
            continue
        
        dr = float(r2 - r1)
        dc = float(c2 - c1)
        
        # Pente de la tangente : dy/dx en coordonnées image (row, col)
        # Attention : en image, row=y, col=x
        if abs(dc) < 1e-8:
            slope_tangent = 1e8   # vertical dans l'espace (col, row)
        else:
            slope_tangent = dr / dc  # pente dans l'espace (col, row)
        
        # Mesure au point milieu entre r1 et r2
        row_mid = (r1 + r2) // 2
        col_mid = (c1 + c2) / 2
        
        w = measure_width_at_point(bz, row_mid, col_mid, slope_tangent)
        
        if w > 0:
            widths.append(w)
    
    if not widths:
        return {'map_width_m': None, 'n_measures': 0, 'widths_m': [], 'trace': trace}
    
    widths = np.array(widths)

    # Médiane : robuste aux mesures aberrantes (coins, bords irréguliers)
    map_width = float(np.median(widths))
    
    if verbose:
        print(f"  {len(widths)} mesures | min={widths.min():.1f}m  max={widths.max():.1f}m  "
              f"médiane={map_width:.1f}m  std={widths.std():.1f}m")
    
    return {
        'map_width_m': map_width,
        'n_measures':  len(widths),
        'widths_m':    widths.tolist(),
        'trace':       trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ÉVALUATION SUR LE DATASET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(csv_path: Path, data_dir: Path, n_samples: int = 100,
             reals_only: bool = False, all_samples: bool = False):
    df = pd.read_csv(csv_path, sep=';')
    if reals_only:
        df_pipe = df[(df['label'] == 1) & df['width_m'].notna() &
                     df['field_file'].str.startswith('real')].copy()
        print(f"  Mode : données RÉELLES uniquement ({len(df_pipe)} fichiers labellisés)")
    else:
        df_pipe = df[(df['label'] == 1) & df['width_m'].notna() &
                     ~df['field_file'].str.startswith('real')].copy()
        print(f"  Mode : données synthétiques ({len(df_pipe)} fichiers disponibles)")
    if not all_samples:
        df_pipe = df_pipe.sample(min(n_samples, len(df_pipe)), random_state=42)
    
    file_index = {p.name: p for p in data_dir.rglob('*.npz')}
    print(f"  {len(file_index)} fichiers indexés")
    
    errors = []
    n_ok   = 0
    
    print(f"\n  {'Fichier':<45} {'true':>7} {'pred':>7} {'err':>7}")
    print("  " + "-"*68)
    
    for _, row in df_pipe.iterrows():
        fname = row['field_file']
        if fname not in file_index:
            continue
        
        arr    = np.load(file_index[fname])['data'].astype('float32')
        true_w = row['width_m']
        
        result = estimate_map_width(arr, step_rows=5, verbose=False)
        pred_w = result['map_width_m']
        
        if pred_w is not None:
            err = abs(pred_w - true_w)
            errors.append(err)
            ok  = '✓' if err < 1.0 else ('~' if err < 5.0 else '✗')
            print(f"  {fname[:45]:<45} {true_w:>6.1f}m {pred_w:>6.1f}m {err:>6.1f}m {ok}")
            n_ok += 1
        else:
            print(f"  {fname[:45]:<45} {true_w:>6.1f}m {'N/A':>7}")
    
    if errors:
        errors = np.array(errors)
        print(f"\n  ══════════════════════════════════════════")
        print(f"  Fichiers traités : {n_ok}/{len(df_pipe)}")
        print(f"  MAE    : {errors.mean():.2f}m")
        print(f"  Médiane: {np.median(errors):.2f}m")
        print(f"  < 1m   : {(errors < 1.0).sum()}/{n_ok} ({100*(errors<1.0).mean():.0f}%)")
        print(f"  < 5m   : {(errors < 5.0).sum()}/{n_ok} ({100*(errors<5.0).mean():.0f}%)")
        print(f"  < 10m  : {(errors < 10.0).sum()}/{n_ok} ({100*(errors<10.0).mean():.0f}%)")
        print(f"  Objectif SKIPPER : MAE < 1m")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Mesure géométrique map_width")
    parser.add_argument('--csv',         default='data/pipe_presence_width_detection_label.csv')
    parser.add_argument('--data_dir',    default='data/raw')
    parser.add_argument('--n_samples',   type=int, default=100)
    parser.add_argument('--file',        default=None, help='Tester sur un fichier unique')
    parser.add_argument('--reals_only',  action='store_true', help='Évaluer uniquement sur les données réelles')
    parser.add_argument('--all',         action='store_true', help='Évaluer sur tout le dataset (ignorer n_samples)')
    args = parser.parse_args()
    
    if args.file:
        arr    = np.load(args.file)['data'].astype('float32')
        result = estimate_map_width(arr, verbose=True)
        print(f"\nmap_width estimée : {result['map_width_m']:.2f}m")
        print(f"Basée sur {result['n_measures']} mesures perpendiculaires")
    else:
        print(f"\n{'='*60}")
        print(f"  Mesure géométrique map_width")
        print(f"  Algorithme : projection perpendiculaire le long de la trace")
        print(f"{'='*60}\n")
        evaluate(ROOT / args.csv, ROOT / args.data_dir,
                 n_samples=args.n_samples,
                 reals_only=args.reals_only,
                 all_samples=args.all)


if __name__ == '__main__':
    main()
