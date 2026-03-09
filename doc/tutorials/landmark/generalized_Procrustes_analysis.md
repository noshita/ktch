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

# Generalized Procrustes analysis

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.datasets import load_landmark_mosquito_wings
from ktch.landmark import GeneralizedProcrustesAnalysis
from ktch.plot import explained_variance_ratio_plot, morphospace_plot, shape_variation_plot
```

## Load mosquito wing landmark dataset

from Rohlf and Slice 1990 _Syst. Zool._

```{code-cell} ipython3
data_landmark_mosquito_wings = load_landmark_mosquito_wings(as_frame=True)
data_landmark_mosquito_wings.coords
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(
    data=data_landmark_mosquito_wings.coords,
    x="x",
    y="y",
    hue="specimen_id",
    style="coord_id",
    ax=ax,
)
ax.set_aspect("equal")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```

For applying generalized Procrustes analysis (GPA),
we convert the configuration data into DataFrame of shape n_specimens x (n_landmarks*n_dim).

```{code-cell} ipython3
def configuration_plot(
    configuration_2d,
    x="x",
    y="y",
    links=None,
    ax=None,
    hue=None,
    hue_order=None,
    c="gray",
    palette=None,
    c_line="gray",
    style=None,
    s=10,
    alpha=1,
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)

    if links is None:
        links = []

    configuration = configuration_2d.reset_index()

    if hue is not None and hue_order is None:
        hue_order = configuration[hue].unique()

    color_map = None
    if hue is not None:
        if palette is not None:
            colors = sns.color_palette(palette, n_colors=len(hue_order))
            color_map = dict(zip(hue_order, colors))
        else:
            hue_dtype = configuration[hue].dtype
            if np.issubdtype(hue_dtype, np.number):
                cmap = sns.cubehelix_palette(as_cmap=True)
                hue_min, hue_max = min(hue_order), max(hue_order)
                color_map = {}
                for hue_val in hue_order:
                    if hue_max > hue_min:
                        norm_val = (hue_val - hue_min) / (hue_max - hue_min)
                    else:
                        norm_val = 0.5
                    color_map[hue_val] = cmap(norm_val)
            else:
                colors = sns.color_palette(n_colors=len(hue_order))
                color_map = dict(zip(hue_order, colors))

    if links:
        if hue is None:
            segments = []
            for link in links:
                link_data = configuration[configuration["coord_id"].isin(link)]
                if len(link_data) == 2:
                    coords = link_data[[x, y]].values
                    segments.append(coords)
            if segments:
                lc = LineCollection(segments, colors=c_line, alpha=alpha)
                ax.add_collection(lc)
        else:
            for specimen in hue_order:
                specimen_data = configuration[configuration[hue] == specimen]
                segments = []
                for link in links:
                    link_data = specimen_data[specimen_data["coord_id"].isin(link)]
                    if len(link_data) == 2:
                        coords = link_data[[x, y]].values
                        segments.append(coords)
                if segments:
                    lc = LineCollection(
                        segments, colors=[color_map[specimen]], alpha=alpha
                    )
                    ax.add_collection(lc)

        ax.autoscale_view()

    axis = sns.scatterplot(
        data=configuration,
        x=x,
        y=y,
        ax=ax,
        hue=hue,
        hue_order=hue_order,
        palette=palette,
        style=style,
        c=c,
        alpha=alpha,
        s=s,
    )

    if axis.legend_:
        sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

    ax.set_aspect("equal")
```

```{code-cell} ipython3
links = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [8, 9],
    [9, 10],
    [10, 11],
    [1, 12],
    [2, 12],
    [13, 14],
    [14, 3],
    [14, 4],
    [15, 5],
    [16, 6],
    [16, 7],
    [17, 8],
    [17, 9],
]

configuration_plot(
    data_landmark_mosquito_wings.coords.loc[1], links=links, alpha=0.5, s=20
)
```

```{code-cell} ipython3
configuration_plot(
    data_landmark_mosquito_wings.coords,
    links=links,
    alpha=0.3,
    hue="specimen_id",
    style="coord_id",
)
```

```{code-cell} ipython3
configuration_plot(
    data_landmark_mosquito_wings.coords.loc[0:2],
    links=links,
    hue="specimen_id",
    style="coord_id",
    palette="Set2",
    s=30,
)
```

```{code-cell} ipython3
index = pd.MultiIndex.from_tuples(
    [(2, i) for i in data_landmark_mosquito_wings.coords.loc[2].index],
    names=["specimen_id", "coord_id"],
)
x2 = pd.DataFrame(
    data_landmark_mosquito_wings.coords.loc[2].to_numpy(),
    columns=["x", "y"],
    index=index,
)
```

```{code-cell} ipython3
data_landmark_mosquito_wings.coords.loc[1:3]
```

```{code-cell} ipython3
df_coords = (
    data_landmark_mosquito_wings.coords.unstack()
    .swaplevel(1, 0, axis=1)
    .sort_index(axis=1)
)
df_coords.columns = [
    dim + "_" + str(landmark_idx) for landmark_idx, dim in df_coords.columns
]
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
df_shapes_vis.columns = pd.MultiIndex.from_tuples(
    [
        [int(landmark_idx), dim]
        for dim, landmark_idx in [idx.split("_") for idx in df_shapes_vis.columns]
    ],
    names=["coord_id", "dim"],
)
df_shapes_vis.sort_index(axis=1, inplace=True)
df_shapes_vis = df_shapes_vis.swaplevel(0, 1, axis=1).stack(level=1, future_stack=True)
df_shapes_vis
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(
    data=df_shapes_vis, x="x", y="y", hue="specimen_id", style="coord_id", ax=ax
)
ax.set_aspect("equal")
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```

```{code-cell} ipython3
configuration_plot(
    df_shapes_vis, links=links, alpha=0.3, hue="specimen_id", style="coord_id"
)
```

```{code-cell} ipython3
configuration_plot(
    df_shapes_vis.loc[0:2],
    links=links,
    hue="specimen_id",
    style="coord_id",
    palette="Set2",
    s=30,
)
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
sns.scatterplot(data=df_pca, x="pca0", y="pca1", hue="genus", palette="Paired", ax=ax)
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
```

## Shape variation along PC axes

```{code-cell} ipython3
fig = shape_variation_plot(
    pca,
    n_dim=2,
    links=links,
    components=(0, 1, 2),
    sd_values=(-2, -1, 0, 1, 2),
)
```

## Morphospace

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=200)

for ax, (i, j) in zip(axes.flat[:3], [(0, 1), (1, 2), (2, 0)]):
    morphospace_plot(
        data=df_pca,
        x=f"pca{i}", y=f"pca{j}", hue="genus",
        reducer=pca,
        n_dim=2,
        components=(i, j),
        links=links,
        palette="Paired",
        n_shapes=5,
        shape_color="gray",
        shape_scale=1.0,
        shape_alpha=0.5,
        ax=ax,
        s=5,
    )
    ax.set(xlabel=f"PC{i + 1}", ylabel=f"PC{j + 1}")

explained_variance_ratio_plot(pca, ax=axes[1, 1])
```
