---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-morphospace)=

# Visualize Morphospace

Plot specimens in principal component space.

## Basic PC scatter plot

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from ktch.datasets import load_landmark_mosquito_wings
from ktch.landmark import GeneralizedProcrustesAnalysis

# Load data
data = load_landmark_mosquito_wings()

# Data is flattened: (n_specimens * n_landmarks, n_dim)
# Reshape to (n_specimens, n_landmarks * n_dim) for GPA
# Mosquito wing dataset: 127 specimens, 18 landmarks, 2D
n_specimens, n_landmarks, n_dim = 127, 18, 2
coords = data.coords.reshape(n_specimens, n_landmarks * n_dim)

# Perform GPA and PCA
gpa = GeneralizedProcrustesAnalysis()
shapes = gpa.fit_transform(coords)

pca = PCA(n_components=3)
scores = pca.fit_transform(shapes)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
scatter = ax.scatter(scores[:, 0], scores[:, 1], c=range(len(scores)), cmap='viridis', alpha=0.7)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
ax.set_aspect('equal')
plt.colorbar(scatter, label='Specimen index')
plt.show()
```

## Plot explained variance

```{code-cell} ipython3
from ktch.plot import explained_variance_ratio_plot

fig, ax = plt.subplots(figsize=(8, 4))
explained_variance_ratio_plot(pca, ax=ax)
plt.show()
```

```{seealso}
- {doc}`../analysis/reconstruct_shapes` for shape reconstruction
- {doc}`../../tutorials/harmonic/elliptic_Fourier_analysis` for complete examples
```
