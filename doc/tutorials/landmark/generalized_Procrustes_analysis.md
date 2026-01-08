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
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.datasets import load_landmark_mosquito_wings
from ktch.landmark import GeneralizedProcrustesAnalysis
from ktch.plot import explained_variance_ratio_plot
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
def configulation_plot(
    configuration_2d,
    x="x",
    y="y",
    links=[],
    ax=None,
    hue=None,
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

    configuration = configuration_2d.reset_index()

    for link in links:
        sns.lineplot(
            data=configuration[configuration["coord_id"].isin(link)],
            x=x,
            y=y,
            sort=False,
            ax=ax,
            hue=hue,
            c=c_line,
            palette=palette,
            alpha=alpha,
            legend=False,
        )

    axis = sns.scatterplot(
        data=configuration,
        x=x,
        y=y,
        ax=ax,
        hue=hue,
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

configulation_plot(
    data_landmark_mosquito_wings.coords.loc[1], links=links, alpha=0.5, s=20
)
```

```{code-cell} ipython3
configulation_plot(
    data_landmark_mosquito_wings.coords,
    links=links,
    hue="specimen_id",
    style="coord_id",
)
```

```{code-cell} ipython3
configulation_plot(
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
configulation_plot(df_shapes_vis, links=links, hue="specimen_id", style="coord_id")
```

```{code-cell} ipython3
configulation_plot(
    df_shapes_vis.loc[0:2], links=links, hue="specimen_id", style="coord_id"
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

## Morphospace

```{code-cell} ipython3
def get_pc_scores_for_morphospace(ax, num=5):
    xrange = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num)
    yrange = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num)
    return xrange, yrange


def plot_recon_morphs(
    pca,
    fig,
    ax,
    n_PCs_xy=[1, 2],
    morph_num=3,
    morph_alpha=1.0,
    morph_scale=1.0,
    links=[],
):
    pc_scores_h, pc_scores_v = get_pc_scores_for_morphospace(ax, morph_num)
    print("PC_h: ", pc_scores_h, ", PC_v: ", pc_scores_v)
    for pc_score_h in pc_scores_h:
        for pc_score_v in pc_scores_v:
            pc_score = np.zeros(pca.n_components_)
            n_PC_h, n_PC_v = n_PCs_xy
            pc_score[n_PC_h - 1] = pc_score_h
            pc_score[n_PC_v - 1] = pc_score_v

            arr_shapes = pca.inverse_transform([pc_score])
            arr_shapes = arr_shapes.reshape(-1, 2)

            df_shapes = pd.DataFrame(arr_shapes, columns=["x", "y"])
            df_shapes["coord_id"] = [i for i in range(len(df_shapes))]
            df_shapes = df_shapes.set_index("coord_id")

            ax_width = ax.get_window_extent().width
            fig_width = fig.get_window_extent().width
            fig_height = fig.get_window_extent().height
            morph_size = morph_scale * ax_width / (fig_width * morph_num)
            loc = ax.transData.transform((pc_score_h, pc_score_v))

            if arr_shapes.shape[1] == 3:
                axins = fig.add_axes(
                    [
                        loc[0] / fig_width - morph_size / 2,
                        loc[1] / fig_height - morph_size / 2,
                        morph_size,
                        morph_size,
                    ],
                    anchor="C",
                    projection="3d",
                )
                axins.patch.set_alpha(0.3)

                configulation_plot(df_shapes, links=links, ax=axins, alpha=morph_alpha)

            else:
                axins = fig.add_axes(
                    [
                        loc[0] / fig_width - morph_size / 2,
                        loc[1] / fig_height - morph_size / 2,
                        morph_size,
                        morph_size,
                    ],
                    anchor="C",
                )
                configulation_plot(df_shapes, links=links, ax=axins, alpha=morph_alpha)

            axins.axis("off")
```

```{code-cell} ipython3
morph_num = 5
morph_scale = 1.0
morph_alpha = 0.5

fig = plt.figure(figsize=(16, 16), dpi=200)
hue_order = df_pca["genus"].unique()

#########
# PC1
#########
ax = fig.add_subplot(2, 2, 1)
sns.scatterplot(
    data=df_pca,
    x="pca0",
    y="pca1",
    hue="genus",
    hue_order=hue_order,
    palette="Paired",
    ax=ax,
    legend=True,
)

plot_recon_morphs(
    pca,
    morph_num=5,
    morph_scale=morph_scale,
    morph_alpha=0.5,
    fig=fig,
    ax=ax,
    links=links,
)

ax.patch.set_alpha(0)
ax.set(xlabel="PC1", ylabel="PC2")

print("PC1-PC2 done")

#########
# PC2
#########
ax = fig.add_subplot(2, 2, 2)
sns.scatterplot(
    data=df_pca,
    x="pca1",
    y="pca2",
    hue="genus",
    hue_order=hue_order,
    palette="Paired",
    ax=ax,
    legend=True,
)

plot_recon_morphs(
    pca,
    morph_num=5,
    morph_scale=morph_scale,
    morph_alpha=0.5,
    fig=fig,
    ax=ax,
    links=links,
    n_PCs_xy=[2, 3],
)


ax.patch.set_alpha(0)
ax.set(xlabel="PC2", ylabel="PC3")

print("PC2-PC3 done")

#########
# PC3
#########
ax = fig.add_subplot(2, 2, 3)
sns.scatterplot(
    data=df_pca,
    x="pca2",
    y="pca0",
    hue="genus",
    hue_order=hue_order,
    palette="Paired",
    ax=ax,
    legend=True,
)

plot_recon_morphs(
    pca,
    morph_num=5,
    morph_scale=morph_scale,
    morph_alpha=0.5,
    fig=fig,
    ax=ax,
    links=links,
    n_PCs_xy=[3, 1],
)


ax.patch.set_alpha(0)
ax.set(xlabel="PC3", ylabel="PC1")

print("PC3-PC1 done")

#########
# CCR
#########

ax = fig.add_subplot(2, 2, 4)
explained_variance_ratio_plot(pca, ax=ax, verbose=True)
```

```{code-cell} ipython3

```
