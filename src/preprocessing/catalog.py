"""
src/preprocessing/catalog.py

Catalogue du dataset — index leger en memoire, chargement lazy des arrays.

Le dataset complet (~500+ fichiers, plusieurs Go) ne peut pas tenir en RAM.
Ce module construit un index de metadata (Path + labels) sans charger les arrays.
Les donnees sont lues a la demande via DataEntry.load().

Auteur(s) : KENGNI Theophane
"""

from __future__ import annotations
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.preprocessing.loader import load_npz
from src.preprocessing.labeling import extract_labels, summarize_labels


@dataclass
class DataEntry:
    """Metadonnees d'un fichier — pas de donnees en RAM."""
    path:               Path
    stem:               str
    origin:             str
    pipe_type:          str
    field_quality:      str
    shape_type:         str
    direction:          str
    pipeline_present:   int
    current_sufficient: Optional[int]
    parallel_pipelines: Optional[int]

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
                f"T1={self.pipeline_present} "
                f"T3={self.current_sufficient} "
                f"T4={self.parallel_pipelines})")


class DatasetCatalog:
    """
    Catalogue complet du dataset — index leger en memoire.

    Args:
        data_dir : chemin vers data/raw (cherche recursivement les .npz)
        verbose  : affiche le resume au chargement
    """

    def __init__(self, data_dir, verbose: bool = True):
        self.data_dir  = Path(data_dir)
        self.entries: List[DataEntry] = []
        self._build(verbose)

    def _build(self, verbose: bool) -> None:
        files = sorted(self.data_dir.rglob("*.npz"))
        if not files:
            print(f"  aucun .npz trouve dans {self.data_dir}")
            return

        for f in files:
            lbl   = extract_labels(f)
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

    def summary(self) -> None:
        summarize_labels([vars(e) for e in self.entries])

    def filter(
        self,
        task:               Optional[str] = None,
        origin:             Optional[str] = None,
        pipe_type:          Optional[str] = None,
        field_quality:      Optional[str] = None,
        pipeline_present:   Optional[int] = None,
        current_sufficient: Optional[int] = None,
        parallel_pipelines: Optional[int] = None,
    ) -> List[DataEntry]:
        """
        Filtre les entrees selon un ou plusieurs criteres.

        task : raccourci — 't1' | 't3' | 't4' filtre automatiquement
               les entrees pour lesquelles le label de la tache est disponible.
        """
        entries = self.entries

        if task == "t3":
            entries = [e for e in entries if e.current_sufficient is not None]
        elif task == "t4":
            entries = [e for e in entries
                       if e.parallel_pipelines is not None and e.origin == "synth"]

        if origin             is not None: entries = [e for e in entries if e.origin == origin]
        if pipe_type          is not None: entries = [e for e in entries if e.pipe_type == pipe_type]
        if field_quality      is not None: entries = [e for e in entries if e.field_quality == field_quality]
        if pipeline_present   is not None: entries = [e for e in entries if e.pipeline_present == pipeline_present]
        if current_sufficient is not None: entries = [e for e in entries if e.current_sufficient == current_sufficient]
        if parallel_pipelines is not None: entries = [e for e in entries if e.parallel_pipelines == parallel_pipelines]

        return entries

    def get_paths_and_labels(self, task: str):
        """
        Retourne (liste de paths, liste de labels) pour une tache donnee.
        Utile pour construire un Dataset PyTorch sans tout charger en RAM.
        """
        task_field = {
            "t1": "pipeline_present",
            "t3": "current_sufficient",
            "t4": "parallel_pipelines",
        }
        if task not in task_field:
            raise ValueError(f"tache inconnue: {task}. Choix: t1, t3, t4")

        entries = self.filter(task=task)
        paths   = [e.path for e in entries]
        labels  = [getattr(e, task_field[task]) for e in entries]
        return paths, labels

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        return f"DatasetCatalog({len(self.entries)} fichiers | {self.data_dir})"
