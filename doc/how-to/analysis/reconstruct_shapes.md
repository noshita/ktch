---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-reconstruct-shapes)=

# Reconstruct Shapes

Transform coefficients back to outlines using `inverse_transform`.

## Basic reconstruction

```{code-cell} ipython3
import numpy as np
from ktch.harmonic import EllipticFourierAnalysis

# Create sample outlines with more points to avoid numerical issues
theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
outlines = np.array([
    np.column_stack([np.cos(theta), np.sin(theta)]),
    np.column_stack([1.2 * np.cos(theta), np.sin(theta)]),
    np.column_stack([np.cos(theta), 1.2 * np.sin(theta)]),
])

efa = EllipticFourierAnalysis(n_harmonics=10)
coefficients = efa.fit_transform(outlines)

# Reconstruct (returns a list)
reconstructed = efa.inverse_transform(coefficients)
print(f"Original shape: {outlines.shape}")
print(f"Number of reconstructed outlines: {len(reconstructed)}")
print(f"First outline shape: {reconstructed[0].shape}")
```

## Reconstruct from PC scores

```{code-cell} ipython3
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
scores = pca.fit_transform(coefficients)

# Reconstruct from PC scores
reconstructed_coef = pca.inverse_transform(scores)
reconstructed_outlines = efa.inverse_transform(reconstructed_coef)
print(f"Reconstructed {len(reconstructed_outlines)} outlines from PCA")
```

## Generate shape at specific PC values

```{code-cell} ipython3
# Shape at mean
mean_score = scores.mean(axis=0)

# Shape at mean +2 SD on PC1
std_score = scores.std(axis=0)
modified_score = mean_score.copy()
modified_score[0] += 2 * std_score[0]

coef = pca.inverse_transform([modified_score])
outline = efa.inverse_transform(coef)[0]
print(f"Generated outline shape: {outline.shape}")
```

```{seealso}
- {doc}`../visualization/morphospace` for visualizing shape variation
```
