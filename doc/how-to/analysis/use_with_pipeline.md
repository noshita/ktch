---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-use-with-pipeline)=

# Use with scikit-learn Pipeline

ktch transformers follow the scikit-learn API.

## Basic usage with GPA

GPA expects flattened input of shape `(n_specimens, n_landmarks * n_dim)`:

```{code-cell} ipython3
import numpy as np
from sklearn.decomposition import PCA
from ktch.landmark import GeneralizedProcrustesAnalysis

# Minimal data: 5 specimens, 4 landmarks, 2D
landmarks_3d = np.array([
    [[0, 0], [1, 0], [1, 1], [0, 1]],
    [[0.1, 0], [1.1, 0], [1, 1.1], [0, 1]],
    [[0, 0.1], [1, 0], [1.1, 1], [0, 1.1]],
    [[0.05, 0.05], [1.05, 0], [1, 1.05], [0, 1]],
    [[0, 0], [1.05, 0.05], [1, 1], [0.05, 1.05]],
], dtype=float)

# Flatten to (n_specimens, n_landmarks * n_dim)
n_specimens, n_landmarks, n_dim = landmarks_3d.shape
landmarks = landmarks_3d.reshape(n_specimens, n_landmarks * n_dim)

# GPA then PCA
gpa = GeneralizedProcrustesAnalysis()
shapes = gpa.fit_transform(landmarks)

pca = PCA(n_components=2)
pc_scores = pca.fit_transform(shapes)
print(f"PC scores shape: {pc_scores.shape}")
```

## Basic usage with EFA

EFA can be used in a sklearn Pipeline for unsupervised transformations:

```{code-cell} ipython3
from sklearn.pipeline import Pipeline
from ktch.harmonic import EllipticFourierAnalysis

# Minimal data: 3 elliptical outlines with variations
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
outlines = np.array([
    np.column_stack([1.0 * np.cos(theta), 0.8 * np.sin(theta)]),
    np.column_stack([1.2 * np.cos(theta), 0.7 * np.sin(theta)]),
    np.column_stack([0.9 * np.cos(theta), 1.0 * np.sin(theta)]),
])

# EFA + PCA pipeline (no y parameter)
pipeline = Pipeline([
    ('efa', EllipticFourierAnalysis(n_harmonics=10)),
    ('pca', PCA(n_components=2))
])

pc_scores = pipeline.fit_transform(outlines)
print(f"PC scores shape: {pc_scores.shape}")
```

## Classification with EFA coefficients

For supervised tasks, apply EFA separately before the classification pipeline:

```{code-cell} ipython3
from sklearn.svm import SVC

# More data for classification
np.random.seed(42)
outlines_more = []
labels = []
for i in range(20):
    scale = 1.0 + 0.1 * np.random.randn()
    outlines_more.append(np.column_stack([scale * np.cos(theta), np.sin(theta)]))
    labels.append(0 if scale < 1.0 else 1)
outlines_more = np.array(outlines_more)
labels = np.array(labels)

# Apply EFA first (unsupervised)
efa = EllipticFourierAnalysis(n_harmonics=10)
coefficients = efa.fit_transform(outlines_more)

# Then use PCA + SVC pipeline on coefficients
pipeline = Pipeline([
    ('pca', PCA(n_components=3)),
    ('svc', SVC())
])

pipeline.fit(coefficients, labels)
print(f"Training accuracy: {pipeline.score(coefficients, labels):.2f}")
```

```{seealso}
- {doc}`cross_validation` for cross-validation examples
- {doc}`../../explanation/morphometrics` for background
```
