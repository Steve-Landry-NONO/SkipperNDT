"""
scripts/run_eda.py

Exploration des donnees (EDA) — generation des 5 figures d'analyse.

Structure attendue dans data_dir :
    real_data_*.npz             -> donnees reelles avec pipe
    real_data_no_pipe_*.npz     -> donnees reelles sans pipe
    sample_*_no_pipe_*.npz      -> donnees synthetiques sans pipe
    sample_*_(perfect|offset|curved)_*.npz -> donnees synthetiques avec pipe

Usage :
    python scripts/run_eda.py --data_dir data/raw --output_dir outputs/figures
    python scripts/run_eda.py --figures 1,3,5

Auteur(s) : GILHAS Radia
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from scipy import stats
from matplotlib.lines import Line2D
import warnings

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing.loader import load_npz, image_stats
from src.preprocessing.catalog import DatasetCatalog
from src.preprocessing.labeling import summarize_labels
from src.utils.visualization import (
    plot_channels,
    plot_profile_1d,
    plot_distribution,
    plot_distribution_comparison,
    plot_correlations_and_nan,
    STYLE,
    CHANNEL_NAMES,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_catalog(data_dir: Path) -> DatasetCatalog:
    """
    Construit le catalogue lazy du dataset (metadata uniquement).

    Structure attendue :
        data/raw/
        ├── real_data/                   -> origin='real'
        └── Training_database_float16/   -> origin='synth'
    """
    return DatasetCatalog(data_dir, verbose=True)


def load_sample_dataset(data_dir: Path, max_per_category: int = 5) -> dict:
    """
    Charge un sous-ensemble representatif en RAM pour les figures EDA.

    Les categories chargees couvrent les 4 combinaisons (origine x label)
    ainsi que les cas parallele/noisy/curved pour l'exhaustivite des figures.

    Args:
        data_dir         : dossier racine des .npz
        max_per_category : nombre maximum de fichiers par categorie

    Returns:
        dict { stem: {"data": arr, "origin": ..., "label": ..., ...} }
    """
    catalog    = DatasetCatalog(data_dir, verbose=False)
    categories = [
        ("synth", "single",   "pipe",    "clean"),
        ("synth", "single",   "pipe",    "noisy"),
        ("synth", "parallel", "pipe",    "clean"),
        ("synth", "parallel", "pipe",    "noisy"),
        ("synth", "no_pipe",  "no_pipe", "clean"),
        ("real",  None,  1,  None),
        ("real",  None,  0,  None),
    ]
    datasets = {}
    for origin, pipe_type, pipe_present, quality in categories:
        kwargs = {"origin": origin}
        if pipe_type    is not None: kwargs["pipe_type"]        = pipe_type
        if pipe_present is not None: kwargs["pipeline_present"] = pipe_present
        if quality      is not None: kwargs["field_quality"]    = quality
        entries = catalog.filter(**kwargs)[:max_per_category]
        for e in entries:
            arr = e.load()
            datasets[e.stem] = {
                "data":      arr,
                "origin":    e.origin,
                "label":     "pipe" if e.pipeline_present == 1 else "no_pipe",
                "pipe_type": e.pipe_type,
                "t1":        e.pipeline_present,
                "t3":        e.current_sufficient,
                "t4":        e.parallel_pipelines,
                "path":      e.path,
            }
            print(f"  {e.stem[:55]:55s} | {e.origin:5s} | "
                  f"T1={e.pipeline_present} T3={e.current_sufficient} T4={e.parallel_pipelines}")

    print(f"\n  {len(datasets)} fichiers charges en RAM")
    return datasets


def extract_features(arr: np.ndarray) -> np.ndarray:
    """
    Extrait 20 features statistiques normalisees (5 par canal * 4 canaux).

    Utilise en interne pour les figures de separabilite — version allegee
    de src/preprocessing/features.py (sans correlations ni NaN fraction).
    """
    feats = []
    for c in range(4):
        ch    = arr[:, :, c].flatten()
        valid = ch[np.isfinite(ch)]
        if len(valid) == 0:
            feats.extend([0.0] * 5)
            continue
        ch_n = (valid - valid.mean()) / (valid.std() + 1e-8)
        feats.extend([
            float(ch_n.mean()),
            float(ch_n.std()),
            float(np.percentile(ch_n, 95) - np.percentile(ch_n, 5)),
            float(np.abs(ch_n).max()),
            float(np.mean(np.abs(ch_n) > 2)),
        ])
    return np.array(feats, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def fig1_channel_views(datasets: dict, out_dir: Path) -> None:
    """
    Figure 1 : Vue des 4 canaux pour les 4 cas types
    (synth pipe, synth no_pipe, real pipe, real no_pipe).
    """
    cases = [
        ("synth", "pipe",    "Synthetique — AVEC pipe"),
        ("synth", "no_pipe", "Synthetique — SANS pipe"),
        ("real",  "pipe",    "Reel — AVEC pipe"),
        ("real",  "no_pipe", "Reel — SANS pipe"),
    ]
    fig, axes = plt.subplots(4, 4, figsize=(18, 14))
    fig.suptitle(
        "Visualisation des 4 canaux magnetiques (Bx, By, Bz, Norme) — 4 cas types",
        fontsize=15, fontweight="bold",
    )

    for row, (origin, label, title) in enumerate(cases):
        candidates = [k for k, v in datasets.items()
                      if v["origin"] == origin and v["label"] == label]
        if not candidates:
            continue
        arr  = datasets[candidates[0]]["data"]
        h, w = arr.shape[:2]
        r0, r1 = max(0, h//2 - 150), min(h, h//2 + 150)
        c0, c1 = max(0, w//2 - 150), min(w, w//2 + 150)
        patch = arr[r0:r1, c0:c1, :]

        for col, (ch_name, cmap) in enumerate(
            zip(CHANNEL_NAMES, ["RdBu_r"] * 3 + ["viridis"])
        ):
            ax   = axes[row, col]
            ch   = patch[:, :, col]
            vmin = np.nanpercentile(ch, 2)
            vmax = np.nanpercentile(ch, 98)
            im   = ax.imshow(ch, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row == 0:
                ax.set_title(f"Canal {ch_name}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_visualisation_canaux.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  fig1_visualisation_canaux.png")


def fig2_distributions_ks(datasets: dict, out_dir: Path) -> None:
    """
    Figure 2 : Distributions, fit gaussien et test KS pour chaque canal
    et chaque categorie.
    """
    groups = {cat: [] for cat in STYLE}
    for k, v in datasets.items():
        cat = (v["origin"], v["label"])
        if cat not in groups:
            continue
        for c in range(4):
            ch    = v["data"][:, :, c].flatten()
            valid = ch[np.isfinite(ch)]
            if len(valid) > 40000:
                valid = np.random.choice(valid, 40000, replace=False)
            groups[cat].append((c, (valid - valid.mean()) / (valid.std() + 1e-8)))

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(
        "Distribution des valeurs par canal (normalisees par image) — histogramme + fit gaussien + test KS",
        fontsize=13, fontweight="bold",
    )

    for row_idx, (cat, ch_vals) in enumerate(groups.items()):
        style = STYLE[cat]
        for c in range(4):
            vals_list = [v for ch_c, v in ch_vals if ch_c == c]
            if not vals_list:
                continue
            ax   = axes[row_idx, c]
            vals = np.concatenate(vals_list)
            ax.hist(vals, bins=60, density=True, color=style["color"], alpha=0.7)
            mu, sigma = stats.norm.fit(vals)
            x = np.linspace(vals.min(), vals.max(), 200)
            ax.plot(x, stats.norm.pdf(x, mu, sigma), "k-", lw=2)
            sk       = stats.skew(vals)
            ku       = stats.kurtosis(vals)
            ks_stat, ks_p = stats.kstest(vals, "norm", args=(mu, sigma))
            verdict  = "approx Gauss" if ks_p > 0.05 else "Non-Gauss"
            if row_idx == 0:
                axes[0, c].set_title(f"Canal {CHANNEL_NAMES[c]}", fontsize=11, fontweight="bold")
            ax.set_ylabel(style["label"], fontsize=8)
            ax.set_title(
                f"mu={mu:.2f} sigma={sigma:.2f} | skew={sk:.2f} kurt={ku:.2f}\n"
                f"KS={ks_stat:.3f} -> {verdict}",
                fontsize=7,
            )
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_distributions_ks.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  fig2_distributions_ks.png")


def fig3_separability(datasets: dict, out_dir: Path) -> None:
    """
    Figure 3 : PCA 2D, KNN et SVM pour evaluer la separabilite lineaire
    des features statistiques.
    """
    X, y, origins = [], [], []
    for k, v in datasets.items():
        X.append(extract_features(v["data"]))
        y.append(1 if v["label"] == "pipe" else 0)
        origins.append(v["origin"])
    X, y, origins = np.array(X), np.array(y), np.array(origins)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    pca    = PCA(n_components=2)
    X_pca  = pca.fit_transform(X_sc)
    var_exp = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        "Analyse de Separabilite — PCA + KNN + SVM (features statistiques normalisees par image)",
        fontsize=13, fontweight="bold",
    )

    cat_colors  = {
        (1, "real"):  "#e63946", (0, "real"):  "#457b9d",
        (1, "synth"): "#ff8c69", (0, "synth"): "#a8dadc",
    }
    cat_markers = {"real": "o", "synth": "s"}

    ax = axes[0]
    for xi, yi, oi in zip(X_pca, y, origins):
        ax.scatter(
            xi[0], xi[1],
            c=cat_colors.get((yi, oi), "gray"),
            marker=cat_markers.get(oi, "o"),
            s=130, edgecolors="black", lw=0.8, zorder=3,
        )
    legend_els = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#e63946",
               markersize=11, markeredgecolor="k", label="Real PIPE"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#457b9d",
               markersize=11, markeredgecolor="k", label="Real NO-PIPE"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#ff8c69",
               markersize=11, markeredgecolor="k", label="Synth PIPE"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#a8dadc",
               markersize=11, markeredgecolor="k", label="Synth NO-PIPE"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
    ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)", fontsize=10)
    ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)", fontsize=10)
    ax.set_title(f"PCA 2D — {sum(var_exp)*100:.1f}% variance expliquee",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax2    = axes[1]
    k_vals = [1, 2, 3, 5, 7]
    cv     = min(5, len(X))
    knn_scores = [
        cross_val_score(KNeighborsClassifier(n_neighbors=k), X_sc, y, cv=cv).mean()
        for k in k_vals
    ]
    bars = ax2.bar(
        k_vals, [s * 100 for s in knn_scores],
        color=["#2a9d8f" if s >= 0.92 else "#e76f51" for s in knn_scores],
        edgecolor="black", width=0.6,
    )
    ax2.axhline(92, color="#e63946", ls="--", lw=2, label="Objectif 92%")
    for bar, s in zip(bars, knn_scores):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{s*100:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax2.set_xlabel("K (voisins)", fontsize=10)
    ax2.set_ylabel("Accuracy (CV)", fontsize=10)
    ax2.set_title("KNN", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 110)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis="y")

    ax3    = axes[2]
    svm_res = {}
    for kernel in ["linear", "rbf", "poly"]:
        svm_res[kernel] = cross_val_score(SVC(kernel=kernel, C=1.0), X_sc, y, cv=cv).mean()
    bars2 = ax3.bar(
        list(svm_res.keys()), [v * 100 for v in svm_res.values()],
        color=["#e63946", "#457b9d", "#f4a261"], edgecolor="black", width=0.5,
    )
    ax3.axhline(92, color="#e63946", ls="--", lw=2, label="Objectif 92%")
    for bar, s in zip(bars2, svm_res.values()):
        ax3.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{s*100:.1f}%", ha="center", va="bottom", fontsize=12, fontweight="bold",
        )
    ax3.set_xlabel("Kernel SVM", fontsize=10)
    ax3.set_ylabel("Accuracy (CV)", fontsize=10)
    ax3.set_title("SVM", fontsize=11, fontweight="bold")
    ax3.set_ylim(0, 110)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_separabilite_pca_knn_svm.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  fig3_separabilite_pca_knn_svm.png")

    print(f"\n  resultats separabilite :")
    print(f"    PCA : PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%")
    for k, s in zip(k_vals, knn_scores):
        print(f"    KNN k={k}: {s*100:.1f}%")
    for kernel, s in svm_res.items():
        print(f"    SVM {kernel}: {s*100:.1f}%")


def fig4_profiles(datasets: dict, out_dir: Path) -> None:
    """
    Figure 4 : Profils 1D transversaux (coupe centrale) pour 3 cas types.

    Met en evidence la signature dipole de Bx/By et le creux de Bz
    caracteristiques d'une conduite.
    """
    cases = [
        ("synth", "pipe",    "Synth PIPE"),
        ("synth", "no_pipe", "Synth NO-PIPE"),
        ("real",  "pipe",    "Reel PIPE"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    fig.suptitle(
        "Profils 1D transversaux (coupe centrale) — signature dipole caracteristique du pipe",
        fontsize=13, fontweight="bold",
    )

    for row, (origin, label, title) in enumerate(cases):
        cands = [k for k, v in datasets.items()
                 if v["origin"] == origin and v["label"] == label]
        if not cands:
            continue
        arr   = datasets[cands[0]]["data"]
        h, _  = arr.shape[:2]
        mid   = h // 2
        color = "#e63946" if label == "pipe" else "#457b9d"

        for col, ch_name in enumerate(CHANNEL_NAMES):
            ax      = axes[row, col]
            profile = arr[mid, :, col]
            valid   = np.isfinite(profile)
            x       = np.arange(len(profile)) * 0.2
            ax.plot(x[valid], profile[valid], color=color, lw=1.5)
            ax.fill_between(x[valid], 0, profile[valid], alpha=0.15, color=color)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            if row == 0:
                ax.set_title(f"Canal {ch_name}", fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(title, fontsize=9)
            ax.set_xlabel("Position (m)", fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_profils_1d.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  fig4_profils_1d.png")


def fig5_domain_gap(datasets: dict, out_dir: Path) -> None:
    """
    Figure 5 : Superposition des distributions synthetique vs reel par canal.

    Le domain gap est visible sur le canal Norm : les synthetiques no-pipe
    sont centrees sur ~0 nT tandis que les reelles ont un offset DC positif.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        "Domain Gap : Synthetique vs Reel (distributions normalisees par image — toutes categories)",
        fontsize=13, fontweight="bold",
    )

    for col, ch_name in enumerate(CHANNEL_NAMES):
        ax = axes[col]
        for k, v in datasets.items():
            origin = v["origin"]
            label  = v["label"]
            style  = STYLE.get((origin, label), {"color": "gray", "ls": "-", "label": k})
            ch     = v["data"][:, :, col].flatten()
            valid  = ch[np.isfinite(ch)]
            if len(valid) > 20000:
                valid = np.random.choice(valid, 20000, replace=False)
            valid_n = (valid - valid.mean()) / (valid.std() + 1e-8)
            ax.hist(
                valid_n, bins=60, density=True,
                color=style["color"], alpha=0.35, label=style["label"],
            )
        ax.set_title(f"Canal {ch_name}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Valeur normalisee", fontsize=9)
        if col == 0:
            ax.set_ylabel("Densite", fontsize=9)
        ax.grid(True, alpha=0.3)
        if col == 3:
            ax.legend(fontsize=6, bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    plt.savefig(out_dir / "fig5_domain_gap.png", dpi=110, bbox_inches="tight")
    plt.close()
    print("  fig5_domain_gap.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="EDA — SkipperNDT")
    parser.add_argument("--data_dir",   type=str, default="data/raw")
    parser.add_argument("--output_dir", type=str, default="outputs/figures")
    parser.add_argument(
        "--figures", type=str, default="all",
        help="Figures a generer, ex: '1,3,5' ou 'all'",
    )
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    out_dir  = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  EDA — SkipperNDT")
    print(f"  data_dir : {data_dir}")
    print(f"  out_dir  : {out_dir}")
    print(f"{'='*60}\n")

    print("  construction du catalogue (index sans chargement RAM)...")
    catalog = build_catalog(data_dir)
    if not catalog.entries:
        print("  aucune donnee trouvee — verifier data_dir")
        return

    print("\n  chargement d'un sous-ensemble representatif...")
    datasets = load_sample_dataset(data_dir, max_per_category=5)
    if not datasets:
        print("  aucune donnee chargee — verifier data_dir")
        return

    to_run = (
        set(range(1, 6)) if args.figures == "all"
        else {int(x) for x in args.figures.split(",")}
    )

    print(f"\n  generation des figures...\n")
    if 1 in to_run: fig1_channel_views(datasets, out_dir)
    if 2 in to_run: fig2_distributions_ks(datasets, out_dir)
    if 3 in to_run: fig3_separability(datasets, out_dir)
    if 4 in to_run: fig4_profiles(datasets, out_dir)
    if 5 in to_run: fig5_domain_gap(datasets, out_dir)

    print(f"\n  figures sauvegardees dans {out_dir}")


if __name__ == "__main__":
    main()
