"""
src/preprocessing/labeling.py
Extraction automatique des labels multi-tâches depuis la nomenclature des fichiers.

Convention de nommage du dataset :
  sample_{id}_perfect_straight_clean_field.npz     → pipe unique, signal propre
  sample_{id}_perfect_straight_noisy_field.npz     → pipe unique, signal bruité
  sample_{id}_offset_straight_clean_field.npz      → pipe unique avec offset, propre
  sample_{id}_no_pipe_straight_clean_field.npz     → pas de pipe
  sample_{id}_no_pipe_curved_clean_field.npz       → pas de pipe, courbé
  parallel_{id}_straight_same_clean_field.npz      → 2 pipes parallèles, même sens, propre
  parallel_{id}_straight_opposite_noisy_field.npz  → 2 pipes parallèles, sens opposé, bruité
  parallel_{id}_curved_same_noisy_field.npz        → 2 pipes courbes parallèles, bruité
  real_data_{id}.npz                               → données réelles avec pipe
  real_data_no_pipe_{id}.npz                       → données réelles sans pipe

Labels extraits par tâche :
  T1 - pipeline_present   : 0 ou 1
  T2 - map_width          : float (extrait du fichier si disponible, sinon NaN)
  T3 - current_sufficient : 0 (noisy) ou 1 (clean) ou NaN (inconnu)
  T4 - parallel_pipelines : 0 (single) ou 1 (parallel) ou NaN (no_pipe/real)
"""

import re
import math
from pathlib import Path
from typing import Dict, Any


def extract_labels(filepath: Path) -> Dict[str, Any]:
    """
    Extrait tous les labels disponibles à partir du nom de fichier.

    Retourne un dict avec les clés :
        origin              : 'real' | 'synth'
        pipe_type           : 'no_pipe' | 'single' | 'parallel' | 'unknown'
        field_quality       : 'clean' | 'noisy' | 'unknown'
        shape_type          : 'straight' | 'curved' | 'unknown'
        direction           : 'same' | 'opposite' | 'unknown'  (pour parallel)
        pipeline_present    : 0 | 1                             (Tâche 1)
        current_sufficient  : 0 | 1 | None                     (Tâche 3)
        parallel_pipelines  : 0 | 1 | None                     (Tâche 4)
    """
    name = filepath.stem.lower()

    # ─── Origine ────────────────────────────────────────────
    parent = filepath.parent.name.lower()
    if "real" in parent or name.startswith("real"):
        origin = "real"
    else:
        origin = "synth"

    # ─── Type de pipe ────────────────────────────────────────
    if "no_pipe" in name:
        pipe_type = "no_pipe"
    elif name.startswith("parallel"):
        pipe_type = "parallel"
    elif name.startswith("sample"):
        pipe_type = "single"
    else:
        pipe_type = "unknown"

    # ─── Qualité du signal ───────────────────────────────────
    if "clean_field" in name:
        field_quality = "clean"
    elif "noisy_field" in name:
        field_quality = "noisy"
    else:
        field_quality = "unknown"

    # ─── Forme ───────────────────────────────────────────────
    if "straight" in name:
        shape_type = "straight"
    elif "curved" in name:
        shape_type = "curved"
    else:
        shape_type = "unknown"

    # ─── Direction (pour parallèles) ─────────────────────────
    if "same" in name:
        direction = "same"
    elif "opposite" in name:
        direction = "opposite"
    else:
        direction = "unknown"

    # ─── TÂCHE 1 : pipeline_present ──────────────────────────
    pipeline_present = 0 if pipe_type == "no_pipe" else 1

    # ─── TÂCHE 3 : current_sufficient ────────────────────────
    # clean_field = courant suffisant, noisy_field = insuffisant
    if field_quality == "clean":
        current_sufficient = 1
    elif field_quality == "noisy":
        current_sufficient = 0
    else:
        current_sufficient = None  # données réelles : inconnu

    # ─── TÂCHE 4 : parallel_pipelines ────────────────────────
    # Seuls les fichiers avec un pipe ont un sens pour T4
    # no_pipe → pas de conduite → question non pertinente → None
    if pipe_type == "parallel":
        parallel_pipelines = 1
    elif pipe_type == "single":
        parallel_pipelines = 0
    else:
        parallel_pipelines = None  # no_pipe ou réel : exclu de T4

    return {
        "origin":             origin,
        "pipe_type":          pipe_type,
        "field_quality":      field_quality,
        "shape_type":         shape_type,
        "direction":          direction,
        "pipeline_present":   pipeline_present,
        "current_sufficient": current_sufficient,
        "parallel_pipelines": parallel_pipelines,
    }


def summarize_labels(label_list: list) -> None:
    """Affiche un résumé statistique d'une liste de dicts de labels."""
    if not label_list:
        print("  [!] Aucun label à résumer")
        return

    n = len(label_list)
    print(f"\n{'='*60}")
    print(f"  RÉSUMÉ DATASET — {n} fichiers")
    print(f"{'='*60}")

    # Origine
    real  = sum(1 for l in label_list if l["origin"] == "real")
    synth = sum(1 for l in label_list if l["origin"] == "synth")
    print(f"  Origine    : réel={real} | synthétique={synth}")

    # T1
    t1_pipe   = sum(1 for l in label_list if l["pipeline_present"] == 1)
    t1_nopipe = sum(1 for l in label_list if l["pipeline_present"] == 0)
    print(f"  T1 pipe    : présent={t1_pipe} | absent={t1_nopipe}")

    # T3
    t3_ok  = sum(1 for l in label_list if l["current_sufficient"] == 1)
    t3_nok = sum(1 for l in label_list if l["current_sufficient"] == 0)
    t3_unk = sum(1 for l in label_list if l["current_sufficient"] is None)
    print(f"  T3 courant : suffisant={t3_ok} | insuffisant={t3_nok} | inconnu={t3_unk}")

    # T4
    t4_par = sum(1 for l in label_list if l["parallel_pipelines"] == 1)
    t4_sin = sum(1 for l in label_list if l["parallel_pipelines"] == 0)
    t4_unk = sum(1 for l in label_list if l["parallel_pipelines"] is None)
    print(f"  T4 parallel: parallèle={t4_par} | unique={t4_sin} | inconnu={t4_unk}")

    # Types
    from collections import Counter
    pipe_types = Counter(l["pipe_type"] for l in label_list)
    shapes     = Counter(l["shape_type"] for l in label_list)
    qualities  = Counter(l["field_quality"] for l in label_list)
    print(f"  Types pipe : {dict(pipe_types)}")
    print(f"  Formes     : {dict(shapes)}")
    print(f"  Qualité    : {dict(qualities)}")
    print(f"{'='*60}\n")
