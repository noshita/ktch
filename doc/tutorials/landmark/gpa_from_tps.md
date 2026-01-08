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

# Generalized Procrustes analysis from TPS file

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.landmark import GeneralizedProcrustesAnalysis
from ktch.io import read_tps
```

## Reading TPS file

```{code-cell} ipython3
df_triangles = read_tps("./landmarks_triangle.tps", as_frame=True)
df_triangles
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(
    data = df_triangles,
    x="x",y="y", 
    hue="specimen_id", style="coord_id",ax=ax
    )
ax.set_aspect('equal')
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```

## GPA

```{code-cell} ipython3
gpa = GeneralizedProcrustesAnalysis().set_output(transform="pandas")
```

```{code-cell} ipython3
df_coords = df_triangles.unstack().swaplevel(1, 0, axis=1).sort_index(axis=1)
df_coords.columns = [dim +"_"+ str(landmark_idx) for landmark_idx,dim in df_coords.columns]
df_coords

df_shapes = gpa.fit_transform(df_coords)
df_shapes
```

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

```{code-cell} ipython3

```
