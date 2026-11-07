# Assignment 1 — K-Means Clustering on Iris

Group the 150 Iris flowers into clusters using only the four measurements,
choose K with the elbow method, and visualise the result.

**Script:** `kmeans_iris.py` · **Outputs:** `outputs/`

## Method

1. Load `../data/iris.csv` (150 rows, 4 numeric features, 3 species).
2. **Standardise** the features (`StandardScaler`) — K-Means uses Euclidean
   distance, so unscaled sepal length would dominate the other features.
3. Fit K-Means for `K = 1…10` with `n_init=10`, `random_state=42` and record
   inertia, silhouette, Calinski–Harabasz and Davies–Bouldin scores.
4. Fit the final model with **K = 3** (the known species count).
5. Visualise clusters and centroids in petal space, next to the true species.
6. Compare cluster ids with species labels using a contingency table, the
   Adjusted Rand Index (ARI) and Normalised Mutual Information (NMI).
7. Save the scaler + model bundle to `outputs/kmeans_model.joblib`.

## Result

| Metric | Value |
|---|---:|
| Inertia (WCSS) | 139.82 |
| Silhouette score | 0.460 |
| Calinski–Harabasz | 241.90 |
| Davies–Bouldin (lower is better) | 0.834 |
| Adjusted Rand Index vs species | 0.620 |
| Normalised Mutual Information | 0.659 |
| Accuracy after cluster→species mapping | **83.3%** |

**K selection.** Inertia drops sharply from 600.0 (K=1) to 222.4 (K=2) and
139.8 (K=3), then flattens (K=4: 114.1). The elbow is at K=3, matching the
three species. The silhouette score is actually highest at K=2 (0.582) because
versicolor and virginica overlap; K=3 is still the right choice for the
project because it matches the domain (three species) and exposes that overlap.

**Contingency table** (rows = true species, columns = predicted cluster):

| species | cluster 0 | cluster 1 | cluster 2 |
|---|---:|---:|---:|
| setosa | 0 | **50** | 0 |
| versicolor | **39** | 0 | 11 |
| virginica | 14 | 0 | **36** |

Cluster → species mapping (majority vote): `0 → versicolor`, `1 → setosa`,
`2 → virginica`.

**Interpretation**

- **Setosa is recovered perfectly** (50/50) and is clearly separated by petal
  length/width.
- The remaining two clusters mix versicolor and virginica: 11 versicolor land
  with 36 virginica, and 14 virginica land with 39 versicolor. The two species
  genuinely overlap in the measured space, so distance-based clustering cannot
  separate them cleanly.
- Centroids in centimetres: setosa `(5.01, 3.43, 1.46, 0.25)`, versicolor
  `(5.80, 2.67, 4.37, 1.41)`, virginica `(6.78, 3.10, 5.51, 1.97)`.

## Outputs

| File | Description |
|---|---|
| `elbow_method.png` | Inertia and silhouette vs K, with K=3 marked |
| `kmeans_clusters.png` | Predicted clusters + centroids vs true species (petal space) |
| `cluster_comparison.png` | Contingency-table heatmap |
| `cluster_assignments.csv` | All 150 flowers with `cluster` and `cluster_species` |
| `metrics.json` | Every score, centroid and the full elbow table |
| `kmeans_model.joblib` | `{scaler, model, features, cluster_to_species}` |

## Run

```bash
cd WEEK3
python "ASSIGNMENT 1 - K-Means Clustering/kmeans_iris.py"
```

## Load the saved model

```python
import pandas as pd
from joblib import load

bundle = load("ASSIGNMENT 1 - K-Means Clustering/outputs/kmeans_model.joblib")
flower = pd.DataFrame([[5.0, 3.4, 1.5, 0.2]], columns=bundle["features"])
cluster = bundle["model"].predict(bundle["scaler"].transform(flower))[0]
print(cluster, bundle["cluster_to_species"][int(cluster)])   # 1 setosa
```
