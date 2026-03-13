"""
src/preprocessing/catalog.py
Catalogue des fichiers du dataset — chargement lazy (metadata seulement).

Le dataset complet (~500+ fichiers, plusieurs Go) ne peut pas tenir en RAM.
Ce module construit un index léger (Path + labels) sans charger les arrays.
Les données sont lues à la demande via get_array().

Usage typique :
    from src.preprocessing.catalog import DatasetCatalog

    cat = DatasetCatalog("data/raw")
    cat.summary()

    # Accès lazy à un fichier
    entry = cat.entries[42]
    arr   = entry.load()          # lit le .npz au moment voulu

    # Filtrer pour la tâche 1
    t1_entries = cat.filter(task="t1")
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from collections import Counter

from src.preprocessing.loader import load_npz
from src.preprocessing.labeling import extract_labels, summarize_labels


@dataclass
class DataEntry:
    """Métadonnées d'un fichier — pas de données en RAM."""
    path:               Path
    stem:               str
    origin:             str        # 'real' | 'synth'
    pipe_type:          str        # 'no_pipe' | 'single' | 'parallel' | 'unknown'
    field_quality:      str        # 'clean' | 'noisy' | 'unknown'
    shape_type:         str        # 'straight' | 'curved' | 'unknown'
    direction:          str        # 'same' | 'opposite' | 'unknown'
    pipeline_present:   int        # T1 : 0 | 1
    current_sufficient: Optional[int]   # T3 : 0 | 1 | None
    parallel_pipelines: Optional[int]   # T4 : 0 | 1 | None

    def load(self) -> np.ndarray:
        """Charge et retourne l'array (H, W, 4) float32."""
        return load_npz(self.path)

    def labels_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_present":   self.pipeline_present,
            "current_sufficient": self.current_sufficient,
            "parallel_pipelines": self.parallel_pipelines,
        }

    def __repr__(self) -> str:
        return (f"DataEntry({self.stem[:40]} | {self.origin} | "
                f"T1={self.pipeline_present} T3={self.current_sufficient} T4={self.parallel_pipelines})")


class DatasetCatalog:
    """
    Catalogue complet du dataset — index léger en mémoire.

    Args:
        data_dir : chemin vers data/raw (cherche récursivement les .npz)
        verbose  : affiche la progression au chargement
    """

    def __init__(self, data_dir: str | Path, verbose: bool = True):
        self.data_dir = Path(data_dir)
        self.entries: List[DataEntry] = []
        self._build(verbose)

    def _build(self, verbose: bool) -> None:
        files = sorted(self.data_dir.rglob("*.npz"))
        if not files:
            print(f"[!] Aucun .npz trouvé dans {self.data_dir}")
            return

        for f in files:
            lbl = extract_labels(f)
            entry = DataEntry(
                path               = f,
                stem               = f.stem,
                origin             = lbl["origin"],
                pipe_type          = lbl["pipe_type"],
                field_quality      = lbl["field_quality"],
                shape_type         = lbl["shape_type"],
                direction          = lbl["direction"],
                pipeline_present   = lbl["pipeline_present"],
                current_sufficient = lbl["current_sufficient"],
                parallel_pipelines = lbl["parallel_pipelines"],
            )
            self.entries.append(entry)

        if verbose:
            self.summary()

    # ─── Statistiques ────────────────────────────────────────

    def summary(self) -> None:
        """Affiche le résumé du catalogue."""
        summarize_labels([vars(e) for e in self.entries])

    # ─── Filtres ─────────────────────────────────────────────

    def filter(
        self,
        task:               Optional[str]  = None,
        origin:             Optional[str]  = None,
        pipe_type:          Optional[str]  = None,
        field_quality:      Optional[str]  = None,
        pipeline_present:   Optional[int]  = None,
        current_sufficient: Optional[int]  = None,
        parallel_pipelines: Optional[int]  = None,
    ) -> List[DataEntry]:
        """
        Filtre les entrées selon un ou plusieurs critères.

        Args:
            task : raccourci — 't1', 't3', 't4' → filtre automatiquement
                   les entrées pour lesquelles le label de la tâche est disponible
        """
        entries = self.entries

        # Raccourcis par tâche
        if task == "t1":
            # Toutes les entrées ont un label T1
            pass
        elif task == "t3":
            # Seulement celles avec current_sufficient connu
            entries = [e for e in entries if e.current_sufficient is not None]
        elif task == "t4":
            # Seulement single + parallel (pas no_pipe, pas real inconnu)
            entries = [e for e in entries if e.parallel_pipelines is not None and e.origin == "synth"]

        # Filtres additionnels
        if origin             is not None: entries = [e for e in entries if e.origin == origin]
        if pipe_type          is not None: entries = [e for e in entries if e.pipe_type == pipe_type]
        if field_quality      is not None: entries = [e for e in entries if e.field_quality == field_quality]
        if pipeline_present   is not None: entries = [e for e in entries if e.pipeline_present == pipeline_present]
        if current_sufficient is not None: entries = [e for e in entries if e.current_sufficient == current_sufficient]
        if parallel_pipelines is not None: entries = [e for e in entries if e.parallel_pipelines == parallel_pipelines]

        return entries

    def count(self, **kwargs) -> int:
        return len(self.filter(**kwargs))

    # ─── Accès pour entraînement ─────────────────────────────

    def get_paths_and_labels(self, task: str) -> tuple[List[Path], List[int]]:
        """
        Retourne (liste de paths, liste de labels) pour une tâche donnée.
        Utile pour construire un Dataset PyTorch sans tout charger en RAM.
        """
        task_field = {
            "t1": "pipeline_present",
            "t3": "current_sufficient",
            "t4": "parallel_pipelines",
        }
        if task not in task_field:
            raise ValueError(f"Tâche inconnue: {task}. Choix: t1, t3, t4")

        entries = self.filter(task=task)
        paths  = [e.path for e in entries]
        labels = [getattr(e, task_field[task]) for e in entries]
        return paths, labels

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"DatasetCatalog({len(self.entries)} fichiers | {self.data_dir})"
