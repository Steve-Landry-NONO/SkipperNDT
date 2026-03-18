"""
task2/inference.py
Inférence — Tâche 2 : Prédiction de la largeur de carte magnétique.

Usage :
    python task2/inference.py --input image.npz --model task2/checkpoints/best_model.pt
    python task2/inference.py --input data/raw/real_data/ --model task2/checkpoints/best_model.pt

Sortie JSON :
    {"map_width_m": 42.3, "confidence": null, "model": "cnn"}
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

    arr      = load_npz(npz_path)
    feats    = extract_features(arr).reshape(1, -1)
    pred_log = float(model.predict(feats)[0])
    pred_m   = float(np.expm1(pred_log))

    return {
        "map_width_m": round(pred_m, 2),
        "model":       "baseline_ml",
        "file":        str(npz_path.name),
    }


def predict_cnn(npz_path: Path, model_path: Path) -> dict:
    try:
        import torch
    except ImportError:
        print("[!] PyTorch requis")
        return {}

    from src.models.dataset import resize_array, normalize_channels
    from src.models.cnn_task2 import get_regressor

    ckpt   = torch.load(model_path, map_location="cpu")
    m_name = ckpt.get("model_name", "cnn")
    model  = get_regressor(m_name)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    arr    = load_npz(npz_path)
    arr    = resize_array(arr)
    arr    = normalize_channels(arr)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)

    with torch.no_grad():
        pred_log = float(model(tensor).item())

    pred_m = float(np.expm1(pred_log))

    return {
        "map_width_m": round(pred_m, 2),
        "model":       m_name,
        "file":        str(npz_path.name),
    }


def run_inference(input_path: Path, model_path: Path) -> list:
    use_cnn = model_path.suffix == ".pt"
    files   = sorted(input_path.rglob("*.npz")) if input_path.is_dir() else [input_path]

    results = []
    for f in files:
        try:
            r = predict_cnn(f, model_path) if use_cnn else predict_baseline(f, model_path)
            results.append(r)
            print(f"  {f.name:<55} → {r['map_width_m']:7.2f} m")
        except Exception as e:
            print(f"  [!] Erreur sur {f.name}: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Inférence — Tâche 2 : map_width")
    parser.add_argument("--input",  required=True)
    parser.add_argument("--model",  required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        print(f"[!] Introuvable : {input_path}"); sys.exit(1)
    if not model_path.exists():
        print(f"[!] Modèle introuvable : {model_path}"); sys.exit(1)

    print(f"\n📏 Inférence — Tâche 2 : map_width")
    print(f"   Input : {input_path}")
    print(f"   Modèle: {model_path}\n")

    results = run_inference(input_path, model_path)

    if results:
        widths = [r["map_width_m"] for r in results]
        print(f"\n  📊 {len(results)} images | "
              f"min={min(widths):.1f}m  max={max(widths):.1f}m  "
              f"mean={np.mean(widths):.1f}m")

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
