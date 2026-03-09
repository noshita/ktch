---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: ktch
  language: python
  name: python3
---

# Semilandmark analysis

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.datasets import load_landmark_trilobite_cephala
from ktch.landmark import GeneralizedProcrustesAnalysis, combine_landmarks_and_curves
from ktch.plot import explained_variance_ratio_plot, morphospace_plot
```

## Load trilobite cephalon dataset

This dataset contains 2D landmark and semilandmark configurations of trilobite cephala (head shields).
Each specimen has 16 fixed landmarks and 4 curves with semilandmarks (12, 20, 20, and 20 points respectively).

```{code-cell} ipython3
data = load_landmark_trilobite_cephala()
landmarks = data.landmarks
curves = data.curves
```

```{code-cell} ipython3
print("landmarks shape:", landmarks.shape)
print("number of curves per specimen:", len(curves[0]))
for i, c in enumerate(curves[0]):
    print(f"  curve {i}: {c.shape[0]} points")
```

```{code-cell} ipython3
df_meta = pd.DataFrame(data.meta)
df_meta.head()
```

## Visualize raw data

Plot a single specimen, distinguishing fixed landmarks from curve semilandmarks.

```{code-cell} ipython3
specimen_idx = 0

fig, ax = plt.subplots(figsize=(6, 6))

# Fixed landmarks
lm = landmarks[specimen_idx]
ax.scatter(lm[:, 0], lm[:, 1], c="black", s=40, zorder=3, label="landmarks")

# Curve semilandmarks
curve_colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
for i, curve in enumerate(curves[specimen_idx]):
    ax.plot(curve[:, 0], curve[:, 1], "o-", color=curve_colors[i], markersize=3, alpha=0.7, label=f"curve {i} ({curve.shape[0]} pts)")

ax.set_aspect("equal")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
ax.set_title("Specimen 0: landmarks and curve semilandmarks")
```

Plot all specimens overlaid before alignment.

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6, 6))

for i in range(landmarks.shape[0]):
    lm = landmarks[i]
    ax.scatter(lm[:, 0], lm[:, 1], c="gray", s=5, alpha=0.2)
    for curve in curves[i]:
        ax.plot(curve[:, 0], curve[:, 1], c="gray", alpha=0.1, linewidth=0.5)

ax.set_aspect("equal")
ax.set_title("All specimens (before alignment)")
```

## Combine landmarks and curves

`combine_landmarks_and_curves()` merges the fixed landmarks and curve semilandmarks
into a single configuration array and generates the slider matrix that defines the sliding topology.

The `curve_landmarks` parameter specifies which landmarks anchor each curve.
Each curve's semilandmarks slide along the tangent direction, with the
specified landmarks serving as fixed endpoints.

```{code-cell} ipython3
combined, slider_matrix, curve_info = combine_landmarks_and_curves(
    landmarks,
    curves,
    curve_landmarks=data.curve_landmarks,
)
print("combined shape:", combined.shape)
print("slider_matrix shape:", slider_matrix.shape)
print("curve_info:", curve_info)
```

The slider matrix has shape `(n_sliders, 3)` where each row is `[before_index, slider_index, after_index]`.
All semilandmarks are included in the slider matrix. For curve endpoints,
the before/after neighbor is the anchoring landmark.

```{code-cell} ipython3
print("First 5 rows of slider_matrix:")
print(slider_matrix[:5])
```

Reshape the combined array to 2D `(n_specimens, n_points * n_dim)` for GPA input.
Specimens containing NaN values are excluded.

```{code-cell} ipython3
X = combined.reshape(combined.shape[0], -1)

nan_mask = np.any(np.isnan(X), axis=1)
print(f"Specimens with NaN: {nan_mask.sum()} / {X.shape[0]}")

X_clean = X[~nan_mask]
df_meta_clean = df_meta[~nan_mask].reset_index(drop=True)
print(f"Specimens used for analysis: {X_clean.shape[0]}")
```

## GPA with semilandmark sliding

Pass the `curves` parameter to `GeneralizedProcrustesAnalysis` to enable semilandmark sliding
during Procrustes superimposition. Semilandmarks slide along their curve tangent directions
to minimize bending energy.

The `tol` parameter controls convergence tolerance for the iterative GPA algorithm.

```{code-cell} ipython3
gpa = GeneralizedProcrustesAnalysis(n_dim=2, curves=slider_matrix, tol=1e-3)
shapes = gpa.fit_transform(X_clean)
print("shapes shape:", shapes.shape)
```

Visualize the aligned shapes.

```{code-cell} ipython3
n_points = combined.shape[1]
n_dim = 2

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mean shape
mean_shape = shapes.mean(axis=0).reshape(n_points, n_dim)
axes[0].scatter(mean_shape[:16, 0], mean_shape[:16, 1], c="black", s=40, zorder=3)
offset = 16
for i, length in enumerate(curve_info["curve_lengths"]):
    start = offset
    end = offset + length
    axes[0].plot(
        mean_shape[start:end, 0], mean_shape[start:end, 1],
        "o-", color=curve_colors[i], markersize=3, alpha=0.7,
    )
    offset = end
axes[0].set_aspect("equal")
axes[0].set_title("Mean shape")

# All aligned specimens
for i in range(shapes.shape[0]):
    specimen = shapes[i].reshape(n_points, n_dim)
    axes[1].scatter(specimen[:, 0], specimen[:, 1], c="gray", s=1, alpha=0.1)
axes[1].set_aspect("equal")
axes[1].set_title("All aligned specimens")

plt.tight_layout()
```

## PCA

```{code-cell} ipython3
pca = PCA(n_components=10)
pc_scores = pca.fit_transform(shapes)

df_pca = pd.DataFrame(
    pc_scores,
    columns=[f"PC{i+1}" for i in range(pc_scores.shape[1])],
)
df_pca = df_pca.join(df_meta_clean[["genus"]])
df_pca.head()
```

```{code-cell} ipython3
fig, ax = plt.subplots()
explained_variance_ratio_plot(pca, ax=ax, verbose=True)
```

## Morphospace visualization

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 8))
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    ax=ax,
    c="gray",
    alpha=0.5,
    s=20,
)
ax.set(xlabel="PC1", ylabel="PC2")
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=200)

for ax, (i, j) in zip(axes.flat[:3], [(0, 1), (1, 2), (2, 0)]):
    morphospace_plot(
        data=df_pca,
        x=f"PC{i + 1}", y=f"PC{j + 1}",
        reducer=pca,
        n_dim=2,
        shape_type="landmarks_2d",
        components=(i, j),
        n_shapes=5,
        shape_alpha=0.5,
        ax=ax,
        scatter_kw=dict(c="gray", alpha=0.3, s=10),
    )

explained_variance_ratio_plot(pca, ax=axes[1, 1], verbose=True)
```

```{code-cell} ipython3

```
