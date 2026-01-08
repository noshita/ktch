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

# Thin-plate spline

```{code-cell} ipython3
import matplotlib.pyplot as plt
import seaborn as sns

from ktch.landmark import GeneralizedProcrustesAnalysis
from ktch.plot import tps_grid_2d_plot
from ktch.datasets import load_landmark_mosquito_wings
```

## Load mosquito wing landmark dataset

from Rohlf and Slice 1990 Syst. Zool.

```{code-cell} ipython3
data_landmark_mosquito_wings = load_landmark_mosquito_wings(as_frame=True)
data_landmark_mosquito_wings.coords
```

## GPA

see also :ref:`generalized_Procrustes_analysis`

```{code-cell} ipython3
X = data_landmark_mosquito_wings.coords.to_numpy().reshape(-1, 18 * 2)
```

```{code-cell} ipython3
gpa = GeneralizedProcrustesAnalysis(tol=10**-5)
```

```{code-cell} ipython3
X_aligned = gpa.fit_transform(X)
```

### Mean shape and an aligned shape

```{code-cell} ipython3
X_reference = gpa.mu_  # mean shape
X_target = X_aligned.reshape(-1, 18, 2)[0]  # the 0-th aligned shape
```

```{code-cell} ipython3
fig = plt.figure()
ax = fig.add_subplot(111)

sns.scatterplot(x=X_reference[:, 0], y=X_reference[:, 1], ax=ax)
sns.scatterplot(x=X_target[:, 0], y=X_target[:, 1], ax=ax)

ax.set_aspect("equal")
```

## Transformation grids of thin-plate splines

```{code-cell} ipython3
tps_grid_2d_plot(X_reference, X_target, outer=0.2, grid_size=0.03)
```
