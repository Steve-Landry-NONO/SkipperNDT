"""
src/preprocessing/labeling.py

Extraction automatique des labels multi-taches depuis la nomenclature des fichiers.

Convention de nommage :
    sample_{id}_perfect_straight_clean_field.npz  -> pipe unique, signal propre
    sample_{id}_no_pipe_straight_clean_field.npz  -> pas de pipe
    parallel_{id}_straight_same_clean_field.npz   -> 2 pipes paralleles, propre
    real_data_{id}.npz                             -> donnees reelles avec pipe
    real_data_no_pipe_{id}.npz                     -> donnees reelles sans pipe

Labels extraits :
    T1 - pipeline_present   : 0 ou 1
    T3 - current_sufficient : 0 (noisy) | 1 (clean) | None (reel inconnu)
    T4 - parallel_pipelines : 0 (single) | 1 (parallel) | None (no_pipe/reel)

Auteur(s) : MAKAMTA Linda
"""

from pathlib import Path
from typing import Dict, Any
from collections import Counter


def extract_labels(filepath: Path) -> Dict[str, Any]:
    """
    Extrait tous les labels disponibles a partir du nom de fichier.

    Returns:
        dict avec les cles : origin, pipe_type, field_quality, shape_type,
        direction, pipeline_present, current_sufficient, parallel_pipelines
    """
    name   = filepath.stem.lower()
    parent = filepath.parent.name.lower()

    origin = "real" if ("real" in parent or name.startswith("real")) else "synth"

    if "no_pipe" in name:
        pipe_type = "no_pipe"
    elif name.startswith("parallel"):
        pipe_type = "parallel"
    elif name.startswith("sample"):
        pipe_type = "single"
    else:
        pipe_type = "unknown"

    if "clean_field" in name:
        field_quality = "clean"
    elif "noisy_field" in name:
        field_quality = "noisy"
    else:
        field_quality = "unknown"

    shape_type = (
        "straight" if "straight" in name else
        "curved"   if "curved"   in name else
        "unknown"
    )

    direction = (
        "same"     if "same"     in name else
        "opposite" if "opposite" in name else
        "unknown"
    )

    pipeline_present = 0 if pipe_type == "no_pipe" else 1

    if field_quality == "clean":
        current_sufficient = 1
    elif field_quality == "noisy":
        current_sufficient = 0
    else:
        current_sufficient = None

    if pipe_type == "parallel":
        parallel_pipelines = 1
    elif pipe_type == "single":
        parallel_pipelines = 0
    else:
        parallel_pipelines = None

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
    """Affiche un resume statistique d'une liste de dicts de labels."""
    if not label_list:
        print("  aucun label a resumer")
        return

    n = len(label_list)
    print(f"\n{'='*60}")
    print(f"  resume dataset — {n} fichiers")
    print(f"{'='*60}")

    real  = sum(1 for l in label_list if l["origin"] == "real")
    synth = sum(1 for l in label_list if l["origin"] == "synth")
    print(f"  origine    : reel={real} | synthetique={synth}")

    t1_pipe   = sum(1 for l in label_list if l["pipeline_present"] == 1)
    t1_nopipe = sum(1 for l in label_list if l["pipeline_present"] == 0)
    print(f"  T1 pipe    : present={t1_pipe} | absent={t1_nopipe}")

    t3_ok  = sum(1 for l in label_list if l["current_sufficient"] == 1)
    t3_nok = sum(1 for l in label_list if l["current_sufficient"] == 0)
    t3_unk = sum(1 for l in label_list if l["current_sufficient"] is None)
    print(f"  T3 courant : suffisant={t3_ok} | insuffisant={t3_nok} | inconnu={t3_unk}")

    t4_par = sum(1 for l in label_list if l["parallel_pipelines"] == 1)
    t4_sin = sum(1 for l in label_list if l["parallel_pipelines"] == 0)
    t4_unk = sum(1 for l in label_list if l["parallel_pipelines"] is None)
    print(f"  T4 parallel: parallele={t4_par} | unique={t4_sin} | inconnu={t4_unk}")

    pipe_types = Counter(l["pipe_type"]     for l in label_list)
    shapes     = Counter(l["shape_type"]    for l in label_list)
    qualities  = Counter(l["field_quality"] for l in label_list)
    print(f"  types pipe : {dict(pipe_types)}")
    print(f"  formes     : {dict(shapes)}")
    print(f"  qualite    : {dict(qualities)}")
    print(f"{'='*60}\n")
