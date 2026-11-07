"""
Assignment 2 - PCA Dimensionality Reduction on the Iris Dataset
===============================================================
Goal: compress the four Iris measurements into a smaller number of
uncorrelated principal components while keeping most of the information.

The script demonstrates:
  1. Standardising features before PCA (PCA is variance-based)
  2. Fitting PCA on all four features and reading the explained variance
  3. Choosing the number of components needed for >= 95% variance
  4. Projecting the data to 2-D and visualising it (scatter + biplot)
  5. Measuring reconstruction error for 1..4 components
  6. Checking that K-Means still works on the PCA-reduced data
  7. Saving the fitted PCA object with joblib

Run:
    python "ASSIGNMENT 2 - PCA Dimensionality Reduction/pca_iris.py"

Outputs are written to ``ASSIGNMENT 2 - PCA Dimensionality Reduction/outputs/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, mean_squared_error, silhouette_score
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
VARIANCE_TARGET = 0.95
N_INIT = 10
CHOSEN_K = 3

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = ROOT / "data" / "iris.csv"
OUTPUT_DIR = HERE / "outputs"

SPECIES_COLORS = {"setosa": "#2a9d8f", "versicolor": "#e9c46a", "virginica": "#e76f51"}

# Axis labels describe the four original measurements.
AXIS_LABELS = {
    "sepal_length": "Sepal length (cm)",
    "sepal_width": "Sepal width (cm)",
    "petal_length": "Petal length (cm)",
    "petal_width": "Petal width (cm)",
}


def apply_plot_style() -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("ggplot")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.unicode_minus": False,
        }
    )


def load_iris_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} is missing. Expected the Iris CSV in WEEK3/data/."
        )
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES) | {"species"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def n_components_for_variance(cumulative: np.ndarray, target: float) -> int:
    """Smallest number of components whose cumulative variance reaches target."""
    return int(np.searchsorted(cumulative, target) + 1)


def reconstruction_errors(X_scaled: np.ndarray) -> pd.DataFrame:
    """MSE after projecting to n components and mapping back to 4-D."""
    rows = []
    for n in range(1, X_scaled.shape[1] + 1):
        pca = PCA(n_components=n, random_state=RANDOM_STATE)
        projected = pca.fit_transform(X_scaled)
        restored = pca.inverse_transform(projected)
        rows.append(
            {
                "n_components": n,
                "reconstruction_mse": float(mean_squared_error(X_scaled, restored)),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_explained_variance(
    pca: PCA, cumulative: np.ndarray, chosen_n: int, path: Path
) -> None:
    components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
    fig, ax1 = plt.subplots(figsize=(7.6, 4.8))

    bars = ax1.bar(
        components,
        pca.explained_variance_ratio_,
        color="#457b9d",
        alpha=0.85,
        width=0.6,
        label="Individual variance",
    )
    ax1.bar_label(bars, fmt="%.1f%%", padding=2, fontsize=8)
    ax1.set_xlabel("Principal component")
    ax1.set_ylabel("Explained variance ratio")
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(components)

    ax2 = ax1.twinx()
    ax2.plot(
        components,
        cumulative,
        "o-",
        color="#e76f51",
        linewidth=2.1,
        label="Cumulative variance",
    )
    ax2.axhline(VARIANCE_TARGET, color="#264653", linestyle="--", linewidth=1.2)
    ax2.text(
        1.05,
        VARIANCE_TARGET + 0.02,
        f"{VARIANCE_TARGET:.0%} target",
        color="#264653",
        fontsize=8,
    )
    ax2.axvline(chosen_n, color="#2a9d8f", linestyle=":", linewidth=1.5)
    ax2.set_ylabel("Cumulative explained variance")
    ax2.set_ylim(0, 1.05)

    ax1.set_title(
        f"PCA explained variance - {chosen_n} components keep "
        f"{cumulative[chosen_n - 1]:.1%}"
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_2d_scatter(scores: np.ndarray, species: pd.Series, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for name in species.unique():
        idx = np.flatnonzero((species == name).to_numpy())
        ax.scatter(
            scores[idx, 0],
            scores[idx, 1],
            s=46,
            alpha=0.85,
            color=SPECIES_COLORS[name],
            edgecolor="white",
            linewidth=0.7,
            label=f"{name} (n={len(idx)})",
        )
    ax.axhline(0, color="#999999", linewidth=0.8)
    ax.axvline(0, color="#999999", linewidth=0.8)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("Iris projected onto the first two principal components")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_biplot(
    scores: np.ndarray,
    species: pd.Series,
    pca: PCA,
    path: Path,
) -> None:
    """Scatter of PC scores plus arrows showing the original feature loadings."""
    fig, ax = plt.subplots(figsize=(7.8, 6.0))
    for name in species.unique():
        idx = np.flatnonzero((species == name).to_numpy())
        ax.scatter(
            scores[idx, 0],
            scores[idx, 1],
            s=34,
            alpha=0.55,
            color=SPECIES_COLORS[name],
            edgecolor="white",
            linewidth=0.5,
            label=name,
        )

    # Arrows are scaled to the data range so they fit on the scatter plot.
    scale = np.abs(scores[:, :2]).max() * 0.85
    for feature, loading in zip(FEATURES, pca.components_[:2].T):
        ax.arrow(
            0,
            0,
            loading[0] * scale,
            loading[1] * scale,
            color="#22333b",
            width=0.012,
            head_width=0.09,
            length_includes_head=True,
            alpha=0.9,
        )
        ax.text(
            loading[0] * scale * 1.10,
            loading[1] * scale * 1.10,
            AXIS_LABELS[feature],
            color="#22333b",
            fontsize=9,
            ha="center",
            va="center",
            fontweight="bold",
        )

    ax.axhline(0, color="#aaaaaa", linewidth=0.7)
    ax.axvline(0, color="#aaaaaa", linewidth=0.7)
    ax.set_xlabel("Principal component 1")
    ax.set_ylabel("Principal component 2")
    ax.set_title("PCA biplot: scores + feature loadings")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_reconstruction_error(errors: pd.DataFrame, chosen_n: int, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(
        errors["n_components"],
        errors["reconstruction_mse"],
        "o-",
        color="#6a4c93",
        linewidth=2.2,
        markersize=8,
    )
    for _, row in errors.iterrows():
        ax.annotate(
            f"{row['reconstruction_mse']:.4f}",
            (row["n_components"], row["reconstruction_mse"]),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
        )
    ax.axvline(chosen_n, color="#2a9d8f", linestyle="--", linewidth=1.4)
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Reconstruction MSE (standardised space)")
    ax.set_title("Information loss when reducing dimensions")
    ax.set_xticks(errors["n_components"])
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    df = load_iris_data()
    X = df[FEATURES]
    species = df["species"]

    print("=" * 74)
    print("ASSIGNMENT 2 - PCA DIMENSIONALITY REDUCTION ON IRIS")
    print("=" * 74)
    print(f"Rows: {len(df)} | Original features: {len(FEATURES)}")
    print(f"Feature correlations:\n{X.corr().round(3).to_string()}")

    # ------------------------------------------------------------------ #
    # 1. Standardise, then fit PCA on all components
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca_full = PCA(random_state=RANDOM_STATE)
    scores_full = pca_full.fit_transform(X_scaled)
    ratios = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(ratios)
    chosen_n = n_components_for_variance(cumulative, VARIANCE_TARGET)

    print("\nExplained variance by component:")
    for i, (ratio, total) in enumerate(zip(ratios, cumulative), start=1):
        print(f"  PC{i}: {ratio:6.2%}  (cumulative {total:6.2%})")
    print(f"\nComponents needed for >= {VARIANCE_TARGET:.0%} variance: {chosen_n}")

    # ------------------------------------------------------------------ #
    # 2. Final PCA with the chosen number of components
    # ------------------------------------------------------------------ #
    pca = PCA(n_components=chosen_n, random_state=RANDOM_STATE)
    scores = pca.fit_transform(X_scaled)
    restored = pca.inverse_transform(scores)
    scaled_mse = mean_squared_error(X_scaled, restored)

    errors = reconstruction_errors(X_scaled)
    print("\nReconstruction error by component count:")
    print(errors.round(5).to_string(index=False))

    # ------------------------------------------------------------------ #
    # 3. Does clustering survive the compression?
    # ------------------------------------------------------------------ #
    full_labels = KMeans(
        n_clusters=CHOSEN_K, n_init=N_INIT, random_state=RANDOM_STATE
    ).fit_predict(X_scaled)
    pca_labels = KMeans(
        n_clusters=CHOSEN_K, n_init=N_INIT, random_state=RANDOM_STATE
    ).fit_predict(scores)

    pca_silhouette = float(silhouette_score(scores, pca_labels))
    full_silhouette = float(silhouette_score(X_scaled, full_labels))
    pca_ari = float(adjusted_rand_score(full_labels, pca_labels))
    species_ari = float(adjusted_rand_score(species, pca_labels))

    print(
        f"\nK-Means(K=3) on 2 PCA components: silhouette {pca_silhouette:.3f}, "
        f"ARI vs true species {species_ari:.3f}"
    )
    print(
        f"K-Means(K=3) on all 4 features:   silhouette {full_silhouette:.3f}, "
        f"ARI vs true species {float(adjusted_rand_score(species, full_labels)):.3f}"
    )
    print(f"Agreement between the two clusterings (ARI): {pca_ari:.3f}")

    metrics = {
        "variance_target": VARIANCE_TARGET,
        "n_original_features": len(FEATURES),
        "n_components_chosen": chosen_n,
        "explained_variance_ratio": [float(v) for v in ratios],
        "cumulative_explained_variance": [float(v) for v in cumulative],
        "variance_retained": float(cumulative[chosen_n - 1]),
        "reconstruction_mse_scaled": float(scaled_mse),
        "reconstruction_error_by_components": errors.to_dict(orient="records"),
        "kmeans_on_full_features": {
            "silhouette": full_silhouette,
            "adjusted_rand_index_vs_species": float(
                adjusted_rand_score(species, full_labels)
            ),
        },
        "kmeans_on_pca": {
            "silhouette": pca_silhouette,
            "adjusted_rand_index_vs_species": species_ari,
            "adjusted_rand_index_vs_full_kmeans": pca_ari,
        },
        "loadings": {
            f"PC{i + 1}": dict(zip(FEATURES, map(float, row)))
            for i, row in enumerate(pca_full.components_)
        },
    }

    # ------------------------------------------------------------------ #
    # 4. Plots + artefacts
    # ------------------------------------------------------------------ #
    plot_explained_variance(pca_full, cumulative, chosen_n, OUTPUT_DIR / "pca_explained_variance.png")
    plot_2d_scatter(scores, species, OUTPUT_DIR / "pca_2d_scatter.png")
    plot_biplot(scores, species, pca_full, OUTPUT_DIR / "pca_biplot.png")
    plot_reconstruction_error(errors, chosen_n, OUTPUT_DIR / "pca_reconstruction_error.png")

    transformed = pd.DataFrame(
        scores, columns=[f"PC{i + 1}" for i in range(chosen_n)]
    )
    transformed.insert(0, "species", species)
    transformed.to_csv(OUTPUT_DIR / "pca_transformed.csv", index=False)

    loadings = pd.DataFrame(
        pca_full.components_.T,
        index=FEATURES,
        columns=[f"PC{i + 1}" for i in range(len(FEATURES))],
    )
    loadings.to_csv(OUTPUT_DIR / "pca_loadings.csv")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    dump(
        {"scaler": scaler, "pca": pca, "features": FEATURES},
        OUTPUT_DIR / "pca_model.joblib",
    )

    print("\nPCA loadings (rows = feature, columns = component):")
    print(loadings.round(3).to_string())
    print(f"\nSaved outputs to {OUTPUT_DIR}")
    print("  pca_explained_variance.png, pca_2d_scatter.png, pca_biplot.png")
    print("  pca_reconstruction_error.png, pca_transformed.csv, pca_loadings.csv")
    print("  metrics.json, pca_model.joblib")


if __name__ == "__main__":
    main()
