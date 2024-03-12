---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Generalized Procrustes analysis

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.datasets import load_landmark_mosquito_wings
from ktch.landmark import GeneralizedProcrustesAnalysis
```



+++

## Load mosquito wing landmark dataset
from Rohlf and Slice 1990 _Syst. Zool._

```{code-cell} ipython3
data_landmark_mosquito_wings = load_landmark_mosquito_wings(as_frame=True)
data_landmark_mosquito_wings.coords
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(
    data = data_landmark_mosquito_wings.coords,
    x="x",y="y", 
    hue="specimen_id", style="coord_id",ax=ax
    )
ax.set_aspect('equal')
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```

For applying generalized Procrustes analysis (GPA), 
we convert the configuration data into DataFrame of shape n_specimens x (n_landmarks*n_dim).

```{code-cell} ipython3
df_coords = data_landmark_mosquito_wings.coords.unstack().swaplevel(1, 0, axis=1).sort_index(axis=1)
df_coords.columns = [dim +"_"+ str(landmark_idx) for landmark_idx,dim in df_coords.columns]
df_coords
```

## GPA

```{code-cell} ipython3
gpa = GeneralizedProcrustesAnalysis().set_output(transform="pandas")
```

```{code-cell} ipython3
df_shapes = gpa.fit_transform(df_coords)
df_shapes
```

We create a DataFrame, called `df_shapes_vis`, of shape (n_specimens*n_landmarks) x n_dim to visualize the aligned shapes.

```{code-cell} ipython3
df_shapes_vis = df_shapes.copy()
df_shapes_vis.columns = pd.MultiIndex.from_tuples([[int(landmark_idx), dim] for dim, landmark_idx in [idx.split("_") for idx in df_shapes_vis.columns]], names=["coord_id","dim"])
df_shapes_vis.sort_index(axis=1, inplace=True)
df_shapes_vis = df_shapes_vis.swaplevel(0,1,axis=1).stack(level=1)
df_shapes_vis
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(
    data = df_shapes_vis,
    x="x",y="y", 
    hue="specimen_id", style="coord_id",ax=ax
    )
ax.set_aspect('equal')
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```

## PCA

```{code-cell} ipython3
pca = PCA(n_components=10).set_output(transform="pandas")
df_pca = pca.fit_transform(df_shapes)

df_pca = df_pca.join(data_landmark_mosquito_wings.meta)
df_pca
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_pca, x="pca0", y = "pca1", 
    hue="genus",palette="Paired",
    ax=ax
    )
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```



+++
