---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-distribution-on-morphospace)=

# Distribution on Morphospace

Overlay confidence ellipses and convex hulls on scatter plots to
visualize per-group distributions.

## Setup

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis
from ktch.plot import confidence_ellipse_plot, convex_hull_plot, morphospace_plot

data = load_outline_mosquito_wings(as_frame=True)
coords = data.coords.to_numpy().reshape(-1, 100, 2)

efa = EllipticFourierAnalysis(n_harmonics=20)
coef = efa.fit_transform(coords)

pca = PCA(n_components=5)
scores = pca.fit_transform(coef)

df_pca = pd.DataFrame(scores, columns=[f"PC{i + 1}" for i in range(5)])
df_pca.index = data.meta.index
df_pca = df_pca.join(data.meta)
```

## Confidence ellipses

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
confidence_ellipse_plot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
```

### Change confidence level

The default is 95 %. Pass `confidence` to adjust:

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
for conf in [0.5, 0.95, 0.99]:
    confidence_ellipse_plot(
        data=df_pca, x="PC1", y="PC2", hue="genus",
        confidence=conf, palette="Paired", legend=False, ax=ax,
    )
```

### Direct standard-deviation control

Instead of a confidence level, pass `n_std` to set the ellipse
radius directly in units of standard deviations:

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
for n in [1.0, 2.0, 3.0]:
    confidence_ellipse_plot(
        data=df_pca, x="PC1", y="PC2", hue="genus",
        n_std=n, palette="Paired", legend=False, ax=ax,
    )
```

### Adjust axis limits

Ellipses may extend beyond the scatter range. Use `ax.margins()` to
add padding:

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
confidence_ellipse_plot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
ax.margins(0.1)
```

### Filled ellipses

```{code-cell} ipython3
fig, ax = plt.subplots()
confidence_ellipse_plot(
    data=df_pca, x="PC1", y="PC2", hue="genus",
    palette="Paired", fill=True, ax=ax,
)
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax, legend=False)
```

## Convex hulls

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
convex_hull_plot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax)
```

### Filled hulls

```{code-cell} ipython3
fig, ax = plt.subplots()
convex_hull_plot(
    data=df_pca, x="PC1", y="PC2", hue="genus",
    palette="Paired", fill=True, ax=ax,
)
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax, legend=False)
```

## Combine with shape overlays

Draw scatter and ellipses first, expand the axis limits with
`ax.margins()`, then add shape overlays. This ensures the shapes
are placed over the expanded range:

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_pca, x="PC1", y="PC2", hue="genus", palette="Paired", ax=ax,
)
confidence_ellipse_plot(
    data=df_pca, x="PC1", y="PC2", hue="genus",
    palette="Paired", legend=False, ax=ax,
)
ax.margins(0.1)
morphospace_plot(
    reducer=pca,
    descriptor=efa,
    components=(0, 1),
    n_shapes=5,
    shape_scale=0.5,
    ax=ax,
)
```

```{seealso}
- {doc}`morphospace` for scatter plots with shape overlays
- {doc}`shape_variation` for visualizing shape changes along component axes
- {doc}`../../explanation/visualization` for the reconstruction pipeline design
```
