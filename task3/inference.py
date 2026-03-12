"""
task3/inference.py
Inférence — Tâche 3 : Suffisance du courant d'excitation.

Usage :
    # Avec le CNN
    python task3/inference.py --input image.npz --model task3/checkpoints/best_model.pt

    # Sur un dossier
    python task3/inference.py --input data/raw/real_data/ --model task3/checkpoints/best_model.pt

    # Avec le modèle baseline ML
    python task3/inference.py --input image.npz --model task3/results/baseline_best.pkl

Sortie JSON :
    {"current_sufficient": 1, "confidence": 0.97, "model": "cnn"}

    current_sufficient = 1 → courant suffisant (clean_field)
    current_sufficient = 0 → courant insuffisant (noisy_field)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.loader import load_npz


def predict_baseline(npz_path: Path, model_path: Path) -> dict:
    """Inférence avec le modèle baseline ML (pkl)."""
    import pickle
    from src.preprocessing.features import extract_features

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]

    arr   = load_npz(npz_path)
    feats = extract_features(arr).reshape(1, -1)

    pred  = int(model.predict(feats)[0])
    proba = model.predict_proba(feats)[0] if hasattr(model, "predict_proba") else None
    conf  = float(proba[pred]) if proba is not None else None

    return {
        "current_sufficient": pred,
        "confidence":         conf,
        "label":              "clean" if pred == 1 else "noisy",
        "model":              "baseline_ml",
        "file":               str(npz_path.name),
    }


def predict_cnn(npz_path: Path, model_path: Path) -> dict:
    """Inférence avec le CNN (pt)."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("[!] PyTorch requis pour l'inférence CNN")
        return {}

    from src.models.dataset import resize_array, normalize_channels
    from src.models.cnn_task1 import get_model

    ckpt   = torch.load(model_path, map_location="cpu")
    m_name = ckpt.get("model_name", "cnn")
    model  = get_model(m_name)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    arr    = load_npz(npz_path)
    arr    = resize_array(arr)
    arr    = normalize_channels(arr)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)  # (1, 4, H, W)

    with torch.no_grad():
        logits = model(tensor)
        proba  = F.softmax(logits, dim=1)[0]
        pred   = int(proba.argmax().item())
        conf   = float(proba[pred].item())

    return {
        "current_sufficient": pred,
        "confidence":         conf,
        "label":              "clean" if pred == 1 else "noisy",
        "model":              m_name,
        "file":               str(npz_path.name),
    }


def run_inference(input_path: Path, model_path: Path) -> list:
    """Inférence sur un fichier ou un dossier."""
    use_cnn = model_path.suffix == ".pt"

    files = sorted(input_path.rglob("*.npz")) if input_path.is_dir() else [input_path]

    results = []
    for f in files:
        try:
            r = predict_cnn(f, model_path) if use_cnn else predict_baseline(f, model_path)
            results.append(r)
            label = "CLEAN" if r["current_sufficient"] == 1 else "NOISY"
            conf  = f"{r['confidence']*100:.1f}%" if r.get("confidence") else "N/A"
            print(f"  {f.name:<55} → {label:<6} (conf: {conf})")
        except Exception as e:
            print(f"  [!] Erreur sur {f.name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Inférence — Tâche 3 : current_sufficient")
    parser.add_argument("--input",  required=True, help="Fichier .npz ou dossier")
    parser.add_argument("--model",  required=True, help="Chemin vers le modèle (.pt ou .pkl)")
    parser.add_argument("--output", default=None,  help="Fichier JSON de sortie (optionnel)")
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"[!] Fichier introuvable : {input_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"[!] Modèle introuvable : {model_path}")
        sys.exit(1)

    print(f"\n🔍 Inférence — Tâche 3 : current_sufficient")
    print(f"   Input : {input_path}")
    print(f"   Modèle: {model_path}\n")

    results = run_inference(input_path, model_path)

    clean = sum(1 for r in results if r["current_sufficient"] == 1)
    noisy = sum(1 for r in results if r["current_sufficient"] == 0)
    print(f"\n  📊 Résultat : {clean} CLEAN | {noisy} NOISY sur {len(results)} images")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"  ✓ Résultats sauvegardés : {out}")
    elif len(results) == 1:
        print(f"\n{json.dumps(results[0], indent=2)}")


if __name__ == "__main__":
    main()
