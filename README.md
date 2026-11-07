# Week 3 — Unsupervised Learning & Model Evaluation

Clustering, dimensionality reduction and model persistence on the Iris
dataset (150 flowers × 4 measurements × 3 species).

## Contents

```
WEEK3/
├── requirements.txt
├── data/
│   └── iris.csv                              # 150 rows, 4 features + species
├── ASSIGNMENT 1 - K-Means Clustering/
│   ├── kmeans_iris.py                        # Elbow method, K-Means, plots
│   ├── outputs/
│   │   ├── elbow_method.png                  # Inertia + silhouette vs K
│   │   ├── kmeans_clusters.png               # Clusters vs true species
│   │   ├── cluster_comparison.png            # Contingency heatmap
│   │   ├── cluster_assignments.csv           # Every flower + cluster id
│   │   ├── metrics.json
│   │   └── kmeans_model.joblib               # Scaler + fitted model
│   └── README.md
├── ASSIGNMENT 2 - PCA Dimensionality Reduction/
│   ├── pca_iris.py                           # PCA, variance, biplot
│   ├── outputs/
│   │   ├── pca_explained_variance.png
│   │   ├── pca_2d_scatter.png
│   │   ├── pca_biplot.png
│   │   ├── pca_reconstruction_error.png
│   │   ├── pca_transformed.csv               # 150 × 2 principal components
│   │   ├── pca_loadings.csv
│   │   ├── metrics.json
│   │   └── pca_model.joblib
│   └── README.md
└── Iris Flower Clustering Project/
    ├── iris_flower_clustering.py             # Full mini-project pipeline
    ├── iris_flower_clustering.ipynb          # Notebook walkthrough (executed)
    ├── iris_flower_clustering.html           # Rendered notebook
    ├── outputs/
    │   ├── elbow_method.png
    │   ├── clusters_scatter.png              # Petal space + PCA space
    │   ├── predicted_vs_true.png             # Predicted clusters vs species
    │   ├── cluster_comparison_heatmap.png
    │   ├── cluster_profiles.png
    │   ├── clustered_iris.csv
    │   ├── cluster_profiles.csv
    │   ├── cluster_centroids.csv
    │   ├── kmeans_model.joblib
    │   ├── kmeans_model.pkl
    │   ├── metrics.json
    │   └── model_card.txt
    └── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

The stack is `pandas`, `numpy`, `scikit-learn`, `matplotlib` and `joblib`
(plus Jupyter for the mini-project notebook).

## Run

```bash
python "ASSIGNMENT 1 - K-Means Clustering/kmeans_iris.py"
python "ASSIGNMENT 2 - PCA Dimensionality Reduction/pca_iris.py"
python "Iris Flower Clustering Project/iris_flower_clustering.py"
```

Or open the executed mini-project notebook:

```bash
jupyter notebook "Iris Flower Clustering Project/iris_flower_clustering.ipynb"
```

All scripts use the headless `matplotlib` Agg backend, so they run over SSH or
in VS Code without a display and save every figure to `outputs/`.

## Results at a glance

| Task | Key result |
|---|---|
| Assignment 1 — K-Means (K=3) | Silhouette **0.460**, inertia **139.82**, ARI vs species **0.620**, accuracy after cluster→species mapping **83.3%** |
| Assignment 2 — PCA | **2 components retain 95.81%** of variance (PC1 72.96%, PC2 22.85%); reconstruction MSE **0.042** in standardised space |
| Mini Project 3 — Iris clustering | K=3 gives purity **83.3%**, ARI **0.620**, NMI **0.659**; joblib and pickle reloads reproduce predictions exactly |

The main lesson: K-Means recovers **setosa** perfectly, but splits the
overlapping **versicolor / virginica** flowers (39 + 11 and 14 + 36 across two
clusters). That is expected — those two species overlap in sepal space and the
algorithm optimises distance, not species labels.

## Notes

- Features are standardised before K-Means and PCA. Without scaling, the
  larger-range sepal length would dominate the distance metric.
- Species labels are **never** used to fit a model. They are used only
  afterwards to evaluate the clusters (ARI, NMI, purity, contingency table).
- `KMeans(n_init=10, random_state=42)` makes every run deterministic.
- Models are saved as a bundle containing the fitted `StandardScaler` and the
  estimator, so raw centimetre measurements can be passed to the loaded model.
