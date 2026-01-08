---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-cross-validation)=

# Perform Cross-Validation

Use scikit-learn's cross-validation with ktch transformers.

## Cross-validation with EFA

Since EFA's `fit_transform` signature differs from sklearn's convention, apply EFA before cross-validation:

```{code-cell} ipython3
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from ktch.harmonic import EllipticFourierAnalysis

# Generate outline data
theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
np.random.seed(42)

outlines = []
labels = []
for i in range(20):
    scale = 1.0 + 0.2 * np.random.randn()
    outlines.append(np.column_stack([scale * np.cos(theta), np.sin(theta)]))
    labels.append(0 if scale < 1.0 else 1)
outlines = np.array(outlines)
labels = np.array(labels)

# Apply EFA first (unsupervised transformation)
efa = EllipticFourierAnalysis(n_harmonics=10)
coefficients = efa.fit_transform(outlines)

# PCA + SVC pipeline for cross-validation
pipeline = Pipeline([
    ('pca', PCA(n_components=3)),
    ('svc', SVC())
])

scores = cross_val_score(pipeline, coefficients, labels, cv=3)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

## Cross-validation with GPA

GPA expects flattened input. Apply GPA before cross-validation:

```{code-cell} ipython3
from sklearn.model_selection import StratifiedKFold
from ktch.landmark import GeneralizedProcrustesAnalysis

# Generate landmark data (3D then flatten)
np.random.seed(42)
landmarks_3d = np.random.randn(20, 4, 2) * 0.1
landmarks_3d += np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
labels = np.array([0] * 10 + [1] * 10)

# Flatten to (n_specimens, n_landmarks * n_dim)
n_specimens, n_landmarks, n_dim = landmarks_3d.shape
landmarks = landmarks_3d.reshape(n_specimens, n_landmarks * n_dim)

# Apply GPA (unsupervised)
gpa = GeneralizedProcrustesAnalysis()
shapes = gpa.fit_transform(landmarks)

# Cross-validation on aligned shapes
pipeline = Pipeline([
    ('pca', PCA(n_components=2)),
    ('svc', SVC())
])

scores = cross_val_score(pipeline, shapes, labels, cv=3)
print(f"Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f})")
```

```{seealso}
- {doc}`use_with_pipeline` for Pipeline examples
```
