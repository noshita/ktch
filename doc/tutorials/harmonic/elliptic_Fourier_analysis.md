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

# Elliptic Fourier Analysis

```{seealso}
For cases where automatic normalization is not suitable, see {doc}`../../how-to/data/2d_outline_registration`.
```

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis
from ktch.plot import explained_variance_ratio_plot, morphospace_plot, shape_variation_plot
```

## Load mosquito wing outline dataset

from Rohlf and Archie 1984 _Syst. Zool._

```{code-cell} ipython3
data_outline_mosquito_wings = load_outline_mosquito_wings(as_frame=True)
data_outline_mosquito_wings.coords
```

```{code-cell} ipython3
coords = data_outline_mosquito_wings.coords.to_numpy().reshape(-1, 100, 2)
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.lineplot(x=coords[0][:, 0], y=coords[0][:, 1], sort=False, estimator=None, ax=ax)
ax.set_aspect("equal")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(10, 10))
sns.lineplot(
    data=data_outline_mosquito_wings.coords,
    x="x",
    y="y",
    hue="specimen_id",
    sort=False,
    estimator=None,
    ax=ax,
)
ax.set_aspect("equal")
```

## EFA

```{code-cell} ipython3
efa = EllipticFourierAnalysis(n_harmonics=20)
```

```{code-cell} ipython3
coef = efa.fit_transform(coords)
```

## PCA

```{code-cell} ipython3
pca = PCA(n_components=12)
pcscores = pca.fit_transform(coef)
```

```{code-cell} ipython3
df_pca = pd.DataFrame(pcscores)
df_pca["specimen_id"] = list(range(len(pcscores)))
df_pca = df_pca.set_index("specimen_id")
df_pca = df_pca.join(data_outline_mosquito_wings.meta)
df_pca = df_pca.rename(columns={i: ("PC" + str(i + 1)) for i in range(12)})
df_pca
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", ax=ax, palette="Paired")
```

## Shape variation along PC axes

```{code-cell} ipython3
fig = shape_variation_plot(
    pca,
    descriptor=efa,
    components=(0, 1, 2),
    sd_values=(-2, -1, 0, 1, 2),
)
```

## Morphospace

```{code-cell} ipython3
ax = morphospace_plot(
    data=df_pca,
    x="PC1", y="PC2", hue="genus",
    reducer=pca,
    descriptor=efa,
    palette="Paired",
    n_shapes=5,
    shape_scale=0.5,
)
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=200)

for ax, (i, j) in zip(axes.flat[:3], [(0, 1), (1, 2), (2, 0)]):
    morphospace_plot(
        data=df_pca,
        x=f"PC{i + 1}", y=f"PC{j + 1}", hue="genus",
        reducer=pca,
        descriptor=efa,
        components=(i, j),
        palette="Paired",
        n_shapes=5,
        shape_color="gray",
        shape_scale=0.8,
        shape_alpha=0.8,
        ax=ax,
    )

explained_variance_ratio_plot(pca, ax=axes[1, 1])
```
