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
outlines = [
    np.column_stack([1.0 * np.cos(theta), 0.8 * np.sin(theta)]),
    np.column_stack([1.2 * np.cos(theta), 0.7 * np.sin(theta)]),
    np.column_stack([0.9 * np.cos(theta), 1.0 * np.sin(theta)]),
]

# EFA + PCA pipeline
pipeline = Pipeline([
    ('efa', EllipticFourierAnalysis(n_harmonics=10, norm=True)),
    ('pca', PCA(n_components=2))
])

pc_scores = pipeline.fit_transform(outlines)
print(f"PC scores shape: {pc_scores.shape}")
```

EFA accepts a list of coordinate arrays, where each specimen may have a
different number of points. The Pipeline passes this list through to EFA
without conversion.

## Passing parameterization via metadata routing

EFA computes arc-length parameterization automatically when `t` is not
provided. If you need to supply a custom parameterization `t` through a
Pipeline, use scikit-learn's metadata routing:

```{code-cell} ipython3
import sklearn

theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
outlines = [
    np.column_stack([1.0 * np.cos(theta), 0.8 * np.sin(theta)]),
    np.column_stack([1.2 * np.cos(theta), 0.7 * np.sin(theta)]),
    np.column_stack([0.9 * np.cos(theta), 1.0 * np.sin(theta)]),
]
t = [np.linspace(0, 2 * np.pi, 64, endpoint=False)] * 3

with sklearn.config_context(enable_metadata_routing=True):
    efa = EllipticFourierAnalysis(n_harmonics=10, norm=True)
    efa.set_transform_request(t=True)

    pipeline = Pipeline([
        ('efa', efa),
        ('pca', PCA(n_components=2))
    ])

    # Pass t by name
    pc_scores = pipeline.fit_transform(outlines, t=t)
    print(f"PC scores shape: {pc_scores.shape}")
```

The routing system dispatches `t` to the EFA step based on the
`set_transform_request(t=True)` declaration. Other steps that do not
request `t` simply ignore it.

## Classification with EFA coefficients

EFA can be included in a classification pipeline:

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
labels = np.array(labels)

# EFA + PCA + SVC pipeline
pipeline = Pipeline([
    ('efa', EllipticFourierAnalysis(n_harmonics=10, norm=True)),
    ('pca', PCA(n_components=3)),
    ('svc', SVC())
])

pipeline.fit(outlines_more, labels)
print(f"Training accuracy: {pipeline.score(outlines_more, labels):.2f}")
```

```{seealso}
- {doc}`cross_validation` for cross-validation examples
- {doc}`../../explanation/morphometrics` for background
```
