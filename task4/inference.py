"""
task4/inference.py

Inference — Tache 4 : Detection de conduites paralleles.

Classe 1 = deux conduites paralleles
Classe 0 = conduite unique

Usage :
    python task4/inference.py --input image.npz --model task4/checkpoints/best_model.pt

Sortie JSON :
    {"parallel_pipelines": 1, "confidence": 0.99, "label": "parallel", "model": "cnn"}

Auteur(s) : MAKAMTA Linda
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
        "parallel_pipelines": pred,
        "confidence":         conf,
        "label":              "parallel" if pred == 1 else "single",
        "model":              "baseline_ml",
        "file":               str(npz_path.name),
    }


def predict_cnn(npz_path: Path, model_path: Path) -> dict:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("PyTorch requis")
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
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        proba  = F.softmax(logits, dim=1)[0]
        pred   = int(proba.argmax().item())
        conf   = float(proba[pred].item())

    return {
        "parallel_pipelines": pred,
        "confidence":         conf,
        "label":              "parallel" if pred == 1 else "single",
        "model":              m_name,
        "file":               str(npz_path.name),
    }


def run_inference(input_path: Path, model_path: Path) -> list:
    use_cnn = model_path.suffix == ".pt"
    files   = sorted(input_path.rglob("*.npz")) if input_path.is_dir() else [input_path]
    results = []

    for f in files:
        try:
            r     = predict_cnn(f, model_path) if use_cnn else predict_baseline(f, model_path)
            label = "PARALLEL" if r["parallel_pipelines"] == 1 else "SINGLE  "
            conf  = f"{r['confidence']*100:.1f}%" if r.get("confidence") else "N/A"
            print(f"  {f.name:<55} -> {label} (conf: {conf})")
            results.append(r)
        except Exception as e:
            print(f"  erreur sur {f.name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Inference — Tache 4 : parallel_pipelines")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--model",  required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"introuvable : {input_path}")
        sys.exit(1)
    if not model_path.exists():
        print(f"modele introuvable : {model_path}")
        sys.exit(1)

    print(f"\n  inference — Tache 4 : parallel_pipelines")
    print(f"  input  : {input_path}")
    print(f"  modele : {model_path}\n")

    results  = run_inference(input_path, model_path)
    parallel = sum(1 for r in results if r["parallel_pipelines"] == 1)
    single   = sum(1 for r in results if r["parallel_pipelines"] == 0)
    print(f"\n  {parallel} PARALLEL | {single} SINGLE sur {len(results)} images")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results if len(results) > 1 else results[0], f, indent=2)
        print(f"  resultats sauvegardes : {out}")
    elif len(results) == 1:
        print(f"\n{json.dumps(results[0], indent=2)}")


if __name__ == "__main__":
    main()
