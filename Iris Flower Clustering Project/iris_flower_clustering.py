"""
Mini Project 3 - Iris Flower Clustering Project
===============================================
Goal: cluster Iris flowers with K-Means (K = 3), visualise the clusters, and
compare the predicted groups against the true species labels.

Pipeline:
  1. Load the Iris dataset and inspect the class balance
  2. Standardise the four measurements
  3. Use the elbow method and silhouette score to justify K
  4. Fit K-Means with K = 3
  5. Visualise clusters in petal space and in a 2-D PCA projection
  6. Compare predicted clusters with the true species
     (side-by-side scatter, contingency heatmap, ARI / NMI / purity)
  7. Save the model with both joblib and pickle, then reload each copy and
     verify that predictions are identical

Run:
    python "Iris Flower Clustering Project/iris_flower_clustering.py"

Outputs are written to ``Iris Flower Clustering Project/outputs/``.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
CHOSEN_K = 3
N_INIT = 10
K_RANGE = range(1, 11)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = ROOT / "data" / "iris.csv"
OUTPUT_DIR = HERE / "outputs"

PALETTE = ["#2a9d8f", "#e9c46a", "#e76f51", "#6a4c93", "#457b9d"]
SPECIES_COLORS = {"setosa": "#2a9d8f", "versicolor": "#e9c46a", "virginica": "#e76f51"}
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
    required = set(FEATURES) | {"species", "species_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return df


def compute_elbow_table(X_scaled: np.ndarray) -> pd.DataFrame:
    rows = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE)
        labels = model.fit_predict(X_scaled)
        row = {"k": k, "inertia": float(model.inertia_), "silhouette": np.nan}
        if 2 <= k < len(X_scaled):
            row["silhouette"] = float(silhouette_score(X_scaled, labels))
        rows.append(row)
    return pd.DataFrame(rows)


def plot_elbow(table: pd.DataFrame, path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(table["k"], table["inertia"], "o-", color="#264653", linewidth=2)
    ax1.axvline(CHOSEN_K, color="#e76f51", linestyle="--", linewidth=1.4)
    ax1.annotate(
        f"K = {CHOSEN_K}",
        xy=(CHOSEN_K, table.loc[table["k"] == CHOSEN_K, "inertia"].iloc[0]),
        xytext=(CHOSEN_K + 0.7, table["inertia"].max() * 0.60),
        color="#e76f51",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#e76f51"},
    )
    ax1.set_xlabel("Number of clusters (K)")
    ax1.set_ylabel("Inertia (within-cluster sum of squares)")
    ax1.set_title("Elbow method")
    ax1.set_xticks(list(table["k"]))

    valid = table.dropna(subset=["silhouette"])
    best_k = int(valid.loc[valid["silhouette"].idxmax(), "k"])
    ax2.plot(valid["k"], valid["silhouette"], "o-", color="#2a9d8f", linewidth=2)
    ax2.axvline(CHOSEN_K, color="#e76f51", linestyle="--", linewidth=1.4)
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

    fig.suptitle("Choosing K for the Iris clustering", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_clusters(
    df: pd.DataFrame,
    labels: np.ndarray,
    centers_raw: np.ndarray,
    pca_scores: np.ndarray,
    pca_centers: np.ndarray,
    path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.0))

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
    ax1.set_title("Clusters in petal space")
    ax1.legend(loc="upper left", fontsize=8)

    for cluster in np.unique(labels):
        mask = labels == cluster
        ax2.scatter(
            pca_scores[mask, 0],
            pca_scores[mask, 1],
            s=42,
            alpha=0.85,
            color=PALETTE[cluster % len(PALETTE)],
            edgecolor="white",
            linewidth=0.6,
            label=f"Cluster {cluster} (n={mask.sum()})",
        )
    ax2.scatter(
        pca_centers[:, 0],
        pca_centers[:, 1],
        marker="X",
        s=240,
        color="#111111",
        edgecolor="white",
        linewidth=1.6,
        label="Centroids",
        zorder=5,
    )
    ax2.set_xlabel("Principal component 1")
    ax2.set_ylabel("Principal component 2")
    ax2.set_title("Clusters in 2-D PCA space")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("K-Means clusters (K = 3)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_predicted_vs_true(
    pca_scores: np.ndarray,
    labels: np.ndarray,
    species: pd.Series,
    ari: float,
    path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 5.2), sharex=True, sharey=True)

    for cluster in np.unique(labels):
        mask = labels == cluster
        ax1.scatter(
            pca_scores[mask, 0],
            pca_scores[mask, 1],
            s=46,
            alpha=0.85,
            color=PALETTE[cluster % len(PALETTE)],
            edgecolor="white",
            linewidth=0.7,
            label=f"Cluster {cluster} (n={mask.sum()})",
        )
    ax1.set_xlabel("Principal component 1")
    ax1.set_ylabel("Principal component 2")
    ax1.set_title(f"Predicted clusters (K = 3)  ·  ARI = {ari:.3f}")
    ax1.legend(loc="best", fontsize=8)

    for name in species.unique():
        mask = (species == name).to_numpy()
        ax2.scatter(
            pca_scores[mask, 0],
            pca_scores[mask, 1],
            s=46,
            alpha=0.85,
            color=SPECIES_COLORS[name],
            edgecolor="white",
            linewidth=0.7,
            label=f"{name} (n={mask.sum()})",
        )
    ax2.set_xlabel("Principal component 1")
    ax2.set_ylabel("Principal component 2")
    ax2.set_title("True species")
    ax2.legend(loc="best", fontsize=8)

    fig.suptitle("Predicted clusters vs. true labels", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_contingency(table: pd.DataFrame, path: Path) -> None:
    values = table.to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    image = ax.imshow(values, cmap="Blues")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.set_xlabel("Predicted cluster")
    ax.set_ylabel("True species")
    ax.set_title("Contingency table: cluster vs. species")
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            count = int(values[i, j])
            color = "white" if count > values.max() / 2 else "#22333b"
            ax.text(j, i, count, ha="center", va="center", color=color, fontweight="bold")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Flowers")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cluster_profiles(profiles: pd.DataFrame, path: Path) -> None:
    """Heatmap of mean measurements per cluster (raw centimetres)."""
    means = profiles[FEATURES]
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    image = ax.imshow(means.to_numpy(), cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(len(FEATURES)), [AXIS_LABELS[f] for f in FEATURES], rotation=18)
    ax.set_yticks(
        range(len(profiles)),
        [f"Cluster {c} · {s}" for c, s in zip(profiles.index, profiles["mapped_species"])],
    )
    for i in range(means.shape[0]):
        for j in range(means.shape[1]):
            ax.text(j, i, f"{means.iat[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Cluster profiles - mean measurements (cm)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="cm")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def map_clusters_to_species(labels: np.ndarray, species: pd.Series) -> dict[int, str]:
    frame = pd.DataFrame({"cluster": labels, "species": species.to_numpy()})
    mapping = (
        frame.groupby(["cluster", "species"]).size().reset_index(name="count")
        .sort_values("count", ascending=False)
        .drop_duplicates("cluster")
        .set_index("cluster")["species"]
        .to_dict()
    )
    return {int(k): v for k, v in mapping.items()}


def purity_score(labels: np.ndarray, species: pd.Series) -> float:
    frame = pd.DataFrame({"cluster": labels, "species": species.to_numpy()})
    return float(frame.groupby("cluster")["species"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(frame))


def write_model_card(path: Path, metrics: dict, mapping: dict) -> None:
    lines = [
        "IRIS FLOWER CLUSTERING - MODEL CARD",
        "=" * 40,
        f"Algorithm          : K-Means (k-means++ init)",
        f"Clusters (K)       : {metrics['chosen_k']}",
        f"Features           : {', '.join(FEATURES)}",
        f"Scaling            : StandardScaler (fit on all 150 flowers)",
        f"Random state       : {RANDOM_STATE}",
        f"n_init             : {N_INIT}",
        "",
        "QUALITY",
        f"Silhouette         : {metrics['silhouette_score']:.4f}",
        f"Adjusted Rand Index: {metrics['adjusted_rand_index']:.4f}",
        f"Normalised MI      : {metrics['normalized_mutual_info']:.4f}",
        f"Purity             : {metrics['purity']:.4f}",
        f"Accuracy (mapped)  : {metrics['accuracy_after_mapping']:.4f}",
        "",
        "CLUSTER -> SPECIES (majority vote)",
    ]
    lines += [f"  Cluster {k}: {v}" for k, v in sorted(mapping.items())]
    lines += [
        "",
        "ARTEFACTS",
        "  kmeans_model.joblib : scaler + model + metadata bundle",
        "  kmeans_model.pkl    : identical bundle, pickle format",
        "  clustered_iris.csv  : every flower with its cluster id",
        "  cluster_profiles.csv: mean measurements per cluster",
        "  metrics.json        : full evaluation numbers",
        "",
        "NOTE: K-Means is unsupervised. Species labels were used only to",
        "evaluate the clusters, never to fit the model.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    apply_plot_style()

    df = load_iris_data()
    X = df[FEATURES]
    species = df["species"]

    print("=" * 78)
    print("MINI PROJECT 3 - IRIS FLOWER CLUSTERING")
    print("=" * 78)
    print(f"Rows: {len(df)} | Features: {len(FEATURES)} | Species: {species.nunique()}")
    print("Class balance:", species.value_counts().to_dict())
    print("\nSummary statistics (cm):")
    print(X.describe().round(2).to_string())

    # ------------------------------------------------------------------ #
    # 1. Scale
    # ------------------------------------------------------------------ #
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ------------------------------------------------------------------ #
    # 2. Choose K (elbow + silhouette)
    # ------------------------------------------------------------------ #
    elbow = compute_elbow_table(X_scaled)
    print("\nElbow / silhouette table:")
    print(elbow.round(3).to_string(index=False, na_rep="-"))
    best_silhouette_k = int(
        elbow.dropna(subset=["silhouette"]).loc[
            elbow.dropna(subset=["silhouette"])["silhouette"].idxmax(), "k"
        ]
    )
    print(f"\nBest silhouette K = {best_silhouette_k}; project uses K = {CHOSEN_K}.")
    plot_elbow(elbow, OUTPUT_DIR / "elbow_method.png")

    # ------------------------------------------------------------------ #
    # 3. Final K-Means model
    # ------------------------------------------------------------------ #
    model = KMeans(n_clusters=CHOSEN_K, n_init=N_INIT, random_state=RANDOM_STATE)
    labels = model.fit_predict(X_scaled)
    centers_raw = scaler.inverse_transform(model.cluster_centers_)

    # 2-D PCA is only for visualisation here, not part of the clustering.
    pca_vis = PCA(n_components=2, random_state=RANDOM_STATE)
    pca_scores = pca_vis.fit_transform(X_scaled)
    pca_centers = pca_vis.transform(model.cluster_centers_)

    mapping = map_clusters_to_species(labels, species)
    mapped_species = np.array([mapping[int(c)] for c in labels])
    contingency = pd.crosstab(
        pd.Series(species, name="species"),
        pd.Series(labels, name="cluster"),
    )

    profiles = (
        df.assign(cluster=labels)
        .groupby("cluster")[FEATURES]
        .mean()
        .round(3)
    )
    profiles.insert(0, "mapped_species", [mapping[int(c)] for c in profiles.index])
    profiles.insert(0, "n_flowers", pd.Series(labels).value_counts().sort_index().to_numpy())

    metrics = {
        "dataset": "Iris (UCI / Kaggle)",
        "n_samples": int(len(df)),
        "n_features": len(FEATURES),
        "features": FEATURES,
        "chosen_k": CHOSEN_K,
        "best_silhouette_k": best_silhouette_k,
        "random_state": RANDOM_STATE,
        "n_init": N_INIT,
        "inertia": float(model.inertia_),
        "silhouette_score": float(silhouette_score(X_scaled, labels)),
        "calinski_harabasz_score": float(calinski_harabasz_score(X_scaled, labels)),
        "davies_bouldin_score": float(davies_bouldin_score(X_scaled, labels)),
        "adjusted_rand_index": float(adjusted_rand_score(species, labels)),
        "adjusted_mutual_info": float(adjusted_mutual_info_score(species, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(species, labels)),
        "homogeneity": float(homogeneity_score(species, labels)),
        "completeness": float(completeness_score(species, labels)),
        "v_measure": float(v_measure_score(species, labels)),
        "purity": purity_score(labels, species),
        "accuracy_after_mapping": float((mapped_species == species.to_numpy()).mean()),
        "cluster_to_species": {str(k): v for k, v in mapping.items()},
        "cluster_sizes": {str(k): int(v) for k, v in pd.Series(labels).value_counts().sort_index().items()},
        "centroids_cm": {
            str(i): dict(zip(FEATURES, map(float, row))) for i, row in enumerate(centers_raw)
        },
        "elbow_table": elbow.replace({np.nan: None}).to_dict(orient="records"),
    }

    def report(title: str, value: float) -> None:
        print(f"  {title:<28}: {value:.4f}")

    print("\nFinal model (K = 3):")
    report("Inertia", metrics["inertia"])
    report("Silhouette", metrics["silhouette_score"])
    report("Calinski-Harabasz", metrics["calinski_harabasz_score"])
    report("Davies-Bouldin (lower better)", metrics["davies_bouldin_score"])

    print("\nAgreement with true species:")
    report("Adjusted Rand Index", metrics["adjusted_rand_index"])
    report("Adjusted Mutual Info", metrics["adjusted_mutual_info"])
    report("Normalised Mutual Info", metrics["normalized_mutual_info"])
    report("Homogeneity", metrics["homogeneity"])
    report("Completeness", metrics["completeness"])
    report("V-measure", metrics["v_measure"])
    report("Purity", metrics["purity"])
    report("Accuracy after mapping", metrics["accuracy_after_mapping"])

    print("\nContingency table (rows = species, columns = cluster):")
    print(contingency.to_string())
    print("\nCluster -> species majority vote:", mapping)
    print("\nCluster profiles (mean cm):")
    print(profiles.round(3).to_string())

    # ------------------------------------------------------------------ #
    # 4. Visualisations
    # ------------------------------------------------------------------ #
    plot_clusters(
        df, labels, centers_raw, pca_scores, pca_centers,
        OUTPUT_DIR / "clusters_scatter.png",
    )
    plot_predicted_vs_true(
        pca_scores, labels, species, metrics["adjusted_rand_index"],
        OUTPUT_DIR / "predicted_vs_true.png",
    )
    plot_contingency(contingency, OUTPUT_DIR / "cluster_comparison_heatmap.png")
    plot_cluster_profiles(profiles, OUTPUT_DIR / "cluster_profiles.png")

    # ------------------------------------------------------------------ #
    # 5. Save datasets and tables
    # ------------------------------------------------------------------ #
    clustered = df.copy()
    clustered["cluster"] = labels
    clustered["cluster_species"] = mapped_species
    clustered.to_csv(OUTPUT_DIR / "clustered_iris.csv", index=False)

    profiles_out = profiles.copy()
    profiles_out.index.name = "cluster"
    profiles_out.to_csv(OUTPUT_DIR / "cluster_profiles.csv")

    pd.DataFrame(profiles_out[FEATURES]).to_csv(OUTPUT_DIR / "cluster_centroids.csv")

    # ------------------------------------------------------------------ #
    # 6. Persist the model twice (joblib + pickle) and verify a round-trip
    # ------------------------------------------------------------------ #
    bundle = {
        "scaler": scaler,
        "model": model,
        "features": FEATURES,
        "cluster_to_species": mapping,
        "n_clusters": CHOSEN_K,
        "random_state": RANDOM_STATE,
        "dataset": "Iris (UCI / Kaggle)",
    }
    joblib_path = OUTPUT_DIR / "kmeans_model.joblib"
    pickle_path = OUTPUT_DIR / "kmeans_model.pkl"
    dump(bundle, joblib_path)
    with open(pickle_path, "wb") as handle:
        pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)

    loaded_joblib = load(joblib_path)
    with open(pickle_path, "rb") as handle:
        loaded_pickle = pickle.load(handle)

    pred_joblib = loaded_joblib["model"].predict(
        loaded_joblib["scaler"].transform(X)
    )
    pred_pickle = loaded_pickle["model"].predict(
        loaded_pickle["scaler"].transform(X)
    )
    joblib_matches = bool(np.array_equal(pred_joblib, labels))
    pickle_matches = bool(np.array_equal(pred_pickle, labels))

    new_flower = pd.DataFrame(
        [[5.0, 3.4, 1.5, 0.2]], columns=FEATURES
    )
    new_cluster = int(
        loaded_joblib["model"].predict(
            loaded_joblib["scaler"].transform(new_flower)
        )[0]
    )
    new_species = loaded_joblib["cluster_to_species"][new_cluster]
    print(
        f"\nNew flower {new_flower.iloc[0].tolist()} -> cluster {new_cluster} "
        f"({new_species})"
    )

    metrics["persistence"] = {
        "joblib_path": joblib_path.name,
        "pickle_path": pickle_path.name,
        "joblib_reload_predictions_match": joblib_matches,
        "pickle_reload_predictions_match": pickle_matches,
        "new_flower": {
            "measurements": new_flower.iloc[0].tolist(),
            "cluster": new_cluster,
            "mapped_species": new_species,
        },
    }
    if not (joblib_matches and pickle_matches):
        raise RuntimeError("Saved model did not reproduce the original predictions.")

    print(f"joblib round-trip identical: {joblib_matches}")
    print(f"pickle round-trip identical: {pickle_matches}")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    write_model_card(OUTPUT_DIR / "model_card.txt", metrics, mapping)

    print(f"\nSaved outputs to {OUTPUT_DIR}")
    print("  elbow_method.png, clusters_scatter.png, predicted_vs_true.png")
    print("  cluster_comparison_heatmap.png, cluster_profiles.png")
    print("  clustered_iris.csv, cluster_profiles.csv, cluster_centroids.csv")
    print("  kmeans_model.joblib, kmeans_model.pkl, metrics.json, model_card.txt")


if __name__ == "__main__":
    main()
