---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-non-pca-reducer)=

# Use Non-PCA Reducers

Use KernelPCA or other reducers with ktch plot functions via explicit override parameters.

The `reducer` convenience parameter targets PCA-compatible estimators. For reducers with different attribute names (e.g., `KernelPCA` stores eigenvalues in `eigenvalues_` instead of `explained_variance_`), pass the override parameters directly.

## Setup

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis
from ktch.plot import shape_variation_plot, morphospace_plot

data = load_outline_mosquito_wings(as_frame=True)
coords = data.coords.to_numpy().reshape(-1, 100, 2)

efa = EllipticFourierAnalysis(n_harmonics=20)
coef = efa.fit_transform(coords)

kpca = KernelPCA(n_components=5, kernel="rbf", fit_inverse_transform=True)
kpca.fit(coef)
```

## Shape variation with KernelPCA

```{code-cell} ipython3
fig = shape_variation_plot(
    reducer_inverse_transform=kpca.inverse_transform,
    explained_variance=kpca.eigenvalues_,
    n_components=kpca.n_components,
    descriptor=efa,
    components=(0, 1, 2),
)
```

## Morphospace with KernelPCA

```{code-cell} ipython3
scores = kpca.transform(coef)
df_kpca = pd.DataFrame(
    scores[:, :2], columns=["KPC1", "KPC2"],
)
df_kpca.index = data.meta.index
df_kpca = df_kpca.join(data.meta)

ax = morphospace_plot(
    data=df_kpca,
    x="KPC1", y="KPC2", hue="genus",
    reducer_inverse_transform=kpca.inverse_transform,
    n_components=kpca.n_components,
    descriptor=efa,
    palette="Paired",
    n_shapes=5,
)
```

```{seealso}
- {doc}`morphospace` for standard PCA morphospace visualization
- {doc}`shape_variation` for shape variation along component axes
- {doc}`../../explanation/visualization` for the convenience vs override parameter design
```
