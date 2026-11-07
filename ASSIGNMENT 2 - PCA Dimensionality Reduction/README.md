# Assignment 2 — PCA Dimensionality Reduction on Iris

Compress the four Iris measurements into a smaller number of uncorrelated
principal components, measure how much information survives, and visualise the
data in 2-D.

**Script:** `pca_iris.py` · **Outputs:** `outputs/`

## Method

1. Load `../data/iris.csv` and standardise the four features. PCA maximises
   variance, so features must be on the same scale.
2. Fit PCA on all four features and read `explained_variance_ratio_`.
3. Pick the smallest number of components that reaches **≥ 95%** cumulative
   variance.
4. Project the data onto PC1–PC2, plot the scores and a **biplot** showing the
   original feature loadings as arrows.
5. Reconstruct the data from 1, 2, 3 and 4 components and record the MSE.
6. Check that K-Means (K=3) still finds the same structure on the compressed
   data.
7. Save the scaler + fitted PCA to `outputs/pca_model.joblib`.

## Result

| Component | Explained variance | Cumulative |
|---|---:|---:|
| PC1 | 72.96% | 72.96% |
| PC2 | 22.85% | **95.81%** |
| PC3 | 3.67% | 99.48% |
| PC4 | 0.52% | 100.00% |

**Two components retain 95.81% of the variance**, so the dataset goes from
4-D to 2-D with less than 5% information loss.

| Reconstruction | MSE (standardised space) |
|---|---:|
| 1 component | 0.2704 |
| **2 components** | **0.0419** |
| 3 components | 0.0052 |
| 4 components | 0.0000 |

**Loadings** (how strongly each original feature contributes):

| Feature | PC1 | PC2 |
|---|---:|---:|
| sepal_length | 0.521 | 0.377 |
| sepal_width | −0.269 | **0.923** |
| petal_length | **0.580** | 0.024 |
| petal_width | 0.565 | 0.067 |

- **PC1** is a size axis: sepal length and both petal measurements load
  positively, while sepal width loads negatively.
- **PC2** is almost entirely **sepal width**, which is the feature least
  correlated with the others.

**Clustering after compression**

| Input | Silhouette | ARI vs species |
|---|---:|---:|
| All 4 standardised features | 0.460 | 0.620 |
| 2 principal components | **0.509** | 0.620 |

K-Means on the 2-D projection produces the **same partition** as on the
original 4-D data (ARI between the two clusterings = **1.000**), while the
silhouette score improves slightly. The discarded components were mostly noise
from sepal width.

## Outputs

| File | Description |
|---|---|
| `pca_explained_variance.png` | Scree plot: individual + cumulative variance |
| `pca_2d_scatter.png` | 150 flowers projected onto PC1/PC2, coloured by species |
| `pca_biplot.png` | Scores + feature-loading arrows |
| `pca_reconstruction_error.png` | Reconstruction MSE vs number of components |
| `pca_transformed.csv` | PC1/PC2 coordinates for every flower |
| `pca_loadings.csv` | Full 4×4 loading matrix |
| `metrics.json` | Variance ratios, errors, clustering comparison, loadings |
| `pca_model.joblib` | `{scaler, pca, features}` |

## Run

```bash
cd WEEK3
python "ASSIGNMENT 2 - PCA Dimensionality Reduction/pca_iris.py"
```

## Load the saved model

```python
import pandas as pd
from joblib import load

bundle = load("ASSIGNMENT 2 - PCA Dimensionality Reduction/outputs/pca_model.joblib")
flower = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=bundle["features"])
scores = bundle["pca"].transform(bundle["scaler"].transform(flower))
print(scores)   # 2-D projection
```
