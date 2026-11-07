"""
Assignment 1 - K-Means Clustering on the Iris Dataset
=====================================================
Goal: group the 150 Iris flowers into clusters without using the species
labels, then visualise the clusters and check how well they line up with the
three true species.

The script demonstrates:
  1. Loading and standardising the four sepal/petal measurements
  2. The elbow method (inertia + silhouette) to choose K
  3. Fitting K-Means with K = 3
  4. Visualising the clusters in petal and sepal space, with centroids
  5. Comparing the predicted clusters with the true species
     (contingency table, Adjusted Rand Index, Normalised Mutual Information)
  6. Saving the fitted scaler + model with joblib

Run:
    python "ASSIGNMENT 1 - K-Means Clustering/kmeans_iris.py"

Outputs are written to ``ASSIGNMENT 1 - K-Means Clustering/outputs/``.
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
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_INIT = 10
K_RANGE = range(1, 11)
CHOSEN_K = 3
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = ROOT / "data" / "iris.csv"
OUTPUT_DIR = HERE / "outputs"

PALETTE = ["#2a9d8f", "#e9c46a", "#e76f51", "#6a4c93", "#457b9d"]
SPECIES_COLORS = {"setosa": "#2a9d8f", "versicolor": "#e9c46a", "virginica": "#e76f51"}

# Physical meaning of each measurement, used in plot labels.
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
    """Load the local Iris CSV (UCI / Kaggle copy)."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} is missing. Expected the Iris CSV in WEEK3/data/."
        )
    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES) | {"species", "species_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    if df[FEATURES].isna().any().any():
        raise ValueError("Iris measurements contain missing values.")
    return df


def compute_elbow_table(X_scaled: np.ndarray) -> pd.DataFrame:
    """Fit K-Means for a range of K and record the usual selection metrics."""
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        row = {
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }
        # Silhouette / CH / DB are only defined for 2 <= K <= n_samples - 1.
        if 2 <= k < len(X_scaled):
            row["silhouette"] = float(silhouette_score(X_scaled, labels))
            row["calinski_harabasz"] = float(calinski_harabasz_score(X_scaled, labels))
            row["davies_bouldin"] = float(davies_bouldin_score(X_scaled, labels))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_elbow(table: pd.DataFrame, path: Path, chosen_k: int) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(table["k"], table["inertia"], "o-", color="#264653", linewidth=2)
    ax1.axvline(chosen_k, color="#e76f51", linestyle="--", linewidth=1.4)
    ax1.annotate(
        f"elbow at K = {chosen_k}",
        xy=(chosen_k, table.loc[table["k"] == chosen_k, "inertia"].iloc[0]),
        xytext=(chosen_k + 0.6, table["inertia"].max() * 0.62),
        color="#e76f51",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#e76f51"},
    )
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow method")
    ax1.set_xticks(list(table["k"]))

    valid = table.dropna(subset=["silhouette"])
    ax2.plot(valid["k"], valid["silhouette"], "o-", color="#2a9d8f", linewidth=2)
    ax2.axvline(chosen_k, color="#e76f51", linestyle="--", linewidth=1.4)
    best_k = int(valid.loc[valid["silhouette"].idxmax(), "k"])
    ax2.scatter(
        [best_k],
        [valid["silhouette"].max()],
        s=110,
        facecolor="none",
        edgecolor="#e76f51",
        linewidth=2,
        zorder=5,
        label=f"best silhouette (K = {best_k})",
    )
    ax2.set_xlabel("Number of clusters (K)")
    ax2.set_ylabel("Silhouette score (higher is better)")
    ax2.set_title("Silhouette analysis")
    ax2.set_xticks(list(valid["k"]))
    ax2.legend(loc="best")

    fig.suptitle("Choosing K for K-Means on Iris", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    centers_raw: np.ndarray,
    path: Path,
) -> None:
    """Predicted clusters next to the true species in petal space."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True, sharey=True)

    for cluster in np.unique(labels):
        mask = labels == cluster
        ax1.scatter(
            df.loc[mask, "petal_length"],
            df.loc[mask, "petal_width"],
            s=42,
            alpha=0.85,
            color=PALETTE[cluster % len(PALETTE)],
            edgecolor="white",
            linewidth=0.6,
            label=f"Cluster {cluster} (n={mask.sum()})",
        )
    ax1.scatter(
        centers_raw[:, 2],
        centers_raw[:, 3],
        marker="X",
        s=240,
        color="#111111",
        edgecolor="white",
        linewidth=1.6,
        label="Centroids",
        zorder=5,
    )
    ax1.set_xlabel(AXIS_LABELS["petal_length"])
    ax1.set_ylabel(AXIS_LABELS["petal_width"])
    ax1.set_title("K-Means clusters (K = 3)")
    ax1.legend(loc="upper left", fontsize=8)

    for species, group in df.groupby("species"):
        ax2.scatter(
            group["petal_length"],
            group["petal_width"],
            s=42,
            alpha=0.85,
            color=SPECIES_COLORS[species],
            edgecolor="white",
            linewidth=0.6,
            label=f"{species} (n={len(group)})",
        )
    ax2.set_xlabel(AXIS_LABELS["petal_length"])
    ax2.set_ylabel(AXIS_LABELS["petal_width"])
    ax2.set_title("True species (reference)")
    ax2.legend(loc="upper left", fontsize=8)

    fig.suptitle("Iris clusters vs. ground truth", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_contingency(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    image = ax.imshow(table.to_numpy(), cmap="Blues")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("True species")
    ax.set_title("Cluster vs. species contingency table")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            value = int(table.iat[i, j])
            color = "white" if value > table.to_numpy().max() / 2 else "#22333b"
            ax.text(j, i, value, ha="center", va="center", color=color, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Flowers")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def map_clusters_to_species(
    labels: np.ndarray, species: pd.Series
) -> dict[int, str]:
    """Majority-vote mapping from arbitrary cluster ids to species names."""
    frame = pd.DataFrame({"cluster": labels, "species": species.to_numpy()})
    mapping = (
        frame.groupby(["cluster", "species"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates("cluster")
        .set_index("cluster")["species"]
        .to_dict()
    )
    return {int(k): v for k, v in mapping.items()}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    df = load_iris_data()
    X = df[FEATURES]
    y = df["species"]

    print("=" * 74)
    print("ASSIGNMENT 1 - K-MEANS CLUSTERING ON IRIS")
    print("=" * 74)
    print(f"Rows: {len(df)} | Features: {len(FEATURES)} | Species: {y.nunique()}")
    print("Class balance:", y.value_counts().to_dict())

    # ------------------------------------------------------------------ #
    # 1. Standardise (K-Means uses Euclidean distance, so scale matters)
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("\nFeature means after scaling:", np.round(X_scaled.mean(axis=0), 3))
    print("Feature std after scaling:  ", np.round(X_scaled.std(axis=0), 3))

    # ------------------------------------------------------------------ #
    # 2. Elbow / silhouette analysis
    # ------------------------------------------------------------------ #
    table = compute_elbow_table(X_scaled)
    print("\nK selection metrics:")
    print(table.round(3).to_string(index=False, na_rep="-"))
    best_silhouette_k = int(table.dropna(subset=["silhouette"]).loc[
        table.dropna(subset=["silhouette"])["silhouette"].idxmax(), "k"
    ])
    print(f"\nHighest silhouette at K = {best_silhouette_k}; using K = {CHOSEN_K} (true species count).")
    plot_elbow(table, OUTPUT_DIR / "elbow_method.png", CHOSEN_K)

    # ------------------------------------------------------------------ #
    # 3. Fit final model
    # ------------------------------------------------------------------ #
    model = KMeans(n_clusters=CHOSEN_K, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    centers_raw = scaler.inverse_transform(model.cluster_centers_)

    metrics = {
        "chosen_k": CHOSEN_K,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "n_samples": int(len(df)),
        "n_features": len(FEATURES),
        "inertia": float(model.inertia_),
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
        "adjusted_rand_index": float(adjusted_rand_score(y, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(y, labels)),
        "best_silhouette_k": best_silhouette_k,
    }

    # Match arbitrary cluster ids to species for readability.
    mapping = map_clusters_to_species(labels, y)
    mapped = np.array([mapping[int(c)] for c in labels])
    metrics["accuracy_after_majority_mapping"] = float((mapped == y.to_numpy()).mean())
    metrics["cluster_to_species_mapping"] = {str(k): v for k, v in mapping.items()}

    contingency = pd.crosstab(
        pd.Series(y, name="species"),
        pd.Series(labels, name="cluster"),
    )

    print("\nCluster sizes:", pd.Series(labels).value_counts().sort_index().to_dict())
    print("\nCentroids in original units (cm):")
    print(pd.DataFrame(centers_raw, columns=FEATURES).round(3).to_string())
    print("\nContingency table (rows = true species, columns = cluster):")
    print(contingency.to_string())
    print("\nCluster -> species majority mapping:", mapping)
    print(
        f"\nAdjusted Rand Index: {metrics['adjusted_rand_index']:.3f} | "
        f"Normalised Mutual Information: {metrics['normalized_mutual_info']:.3f} | "
        f"Accuracy after mapping: {metrics['accuracy_after_majority_mapping']:.1%}"
    )
    print(f"Silhouette: {metrics['silhouette_score']:.3f} | Inertia: {metrics['inertia']:.2f}")

    # ------------------------------------------------------------------ #
    # 4. Plots + artefacts
    # ------------------------------------------------------------------ #
    plot_clusters(df, labels, centers_raw, OUTPUT_DIR / "kmeans_clusters.png")
    plot_contingency(contingency, OUTPUT_DIR / "cluster_comparison.png")

    assignments = df.copy()
    assignments["cluster"] = labels
    assignments["cluster_species"] = mapped
    assignments.to_csv(OUTPUT_DIR / "cluster_assignments.csv", index=False)

    metrics["elbow_table"] = table.replace({np.nan: None}).to_dict(orient="records")
    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    # One joblib bundle keeps the scaler with the model, which is essential:
    # raw centimetres must be scaled before they reach K-Means.
    dump(
        {"scaler": scaler, "model": model, "features": FEATURES, "cluster_to_species": mapping},
        OUTPUT_DIR / "kmeans_model.joblib",
    )

    print(f"\nSaved outputs to {OUTPUT_DIR}")
    print("  elbow_method.png, kmeans_clusters.png, cluster_comparison.png")
    print("  cluster_assignments.csv, metrics.json, kmeans_model.joblib")


if __name__ == "__main__":
    main()
