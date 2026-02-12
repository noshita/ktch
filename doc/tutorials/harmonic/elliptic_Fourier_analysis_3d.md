---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: ktch
  language: python
  name: python3
---

# 3D Elliptic Fourier Analysis

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.decomposition import PCA

from ktch.datasets import load_outline_leaf_bending
from ktch.harmonic import EllipticFourierAnalysis
from ktch.plot import explained_variance_ratio_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is only required for the Sphinx documentation build.
# You do not need this setting when running in Jupyter.
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
```

## Load the leaf bending outline dataset

```{code-cell} ipython3
data = load_outline_leaf_bending(as_frame=True)
data.coords
```

```{code-cell} ipython3
df_meta = data.meta.copy()
df_meta["bending_angle"] = (
    np.degrees(df_meta["alphaB"]).round().astype(int).astype(str) + "\u00b0"
)
df_meta["aspect_ratio"] = df_meta["alpha"]
df_meta
```

```{code-cell} ipython3
coords = data.coords.to_numpy().reshape(-1, 200, 3)
coords.shape
```

## Visualize 3D outlines

```{code-cell} ipython3
representative_ids = [0, 10, 20, 30, 40, 50]
bending_order = [ str(deg)+ "\u00b0" for deg in  [20, 80, 140] ]

dfs = []
for sid in representative_ids:
    df = pd.DataFrame(coords[sid], columns=["x", "y", "z"])
    df["idx"] = data.meta.index[sid]
    df["bending_angle"] = df_meta.iloc[sid]["bending_angle"]
    dfs.append(df)
df_vis = pd.concat(dfs, ignore_index=True)

fig = px.line_3d(
    df_vis,
    x="x",
    y="y",
    z="z",
    line_group="idx",
    color="bending_angle",
    category_orders ={"bending_angle":bending_order},
    color_discrete_sequence=px.colors.qualitative.Set2,
)
fig.update_layout(scene=dict(aspectmode="data"))
fig.show()
```

## 3D EFA without normalization

```{code-cell} ipython3
efa = EllipticFourierAnalysis(n_harmonics=20, n_dim=3)
```

```{code-cell} ipython3
coef = efa.fit_transform(coords, norm=False)
coef.shape
```

### Reconstruction from coefficients

```{code-cell} ipython3
coords_recon = efa.inverse_transform(coef, t_num=200, norm=False)
```

```{code-cell} ipython3
sid = 0
df_orig = pd.DataFrame(coords[sid], columns=["x", "y", "z"])
df_orig["type"] = "original"
df_rec = pd.DataFrame(coords_recon[sid], columns=["x", "y", "z"])
df_rec["type"] = "reconstructed"
df_cmp = pd.concat([df_orig, df_rec], ignore_index=True)

fig = px.line_3d(df_cmp, x="x", y="y", z="z", color="type")
fig.update_layout(scene=dict(aspectmode="data"))
fig.show()
```

## 3D EFA

```{code-cell} ipython3
efa = EllipticFourierAnalysis(n_harmonics=20, n_dim=3)
```

```{code-cell} ipython3
coef = efa.fit_transform(coords, norm=True)
coef.shape
```

### Reconstruction from coefficients

```{code-cell} ipython3
coords_recon = efa.inverse_transform(coef, t_num=200, norm=True)
```

```{code-cell} ipython3
representative_ids = [0, 10, 20, 30, 40, 50]

dfs = []
for sid in representative_ids:
    df = pd.DataFrame(coords_recon[sid], columns=["x", "y", "z"])
    df["idx"] = data.meta.index[sid]
    df["bending_angle"] = df_meta.iloc[sid]["bending_angle"]
    dfs.append(df)
df_vis = pd.concat(dfs, ignore_index=True)

fig = px.line_3d(
    df_vis, x="x", y="y", z="z",
    line_group="idx",
    color="bending_angle",
    category_orders ={"bending_angle":bending_order},
    color_discrete_sequence=px.colors.qualitative.Set2
    )
fig.update_layout(scene=dict(aspectmode="data"))
fig.show()
```

## PCA

```{code-cell} ipython3
pca = PCA(n_components=12)
pcscores = pca.fit_transform(coef)
```

```{code-cell} ipython3
df_pca = pd.DataFrame(pcscores, columns=[f"PC{i+1}" for i in range(12)])
df_pca.index = df_meta.index
df_pca = df_pca.join(df_meta)
df_pca
```

```{code-cell} ipython3
fig, ax = plt.subplots()
explained_variance_ratio_plot(pca, ax=ax, verbose=True)
```

```{code-cell} ipython3

fig, ax = plt.subplots()
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="bending_angle",
    hue_order=bending_order,
    style="aspect_ratio",
    palette="Set2",
    ax=ax,
)
```

## Morphospace

```{code-cell} ipython3
def project_3d_to_2d(points, azim=-60, elev=30):
    """Project 3D points onto 2D plane from given viewpoint.
    Uses the same azimuth/elevation as matplotlib.
    """
    az = np.radians(azim)
    el = np.radians(elev)
    R = np.array([
        [-np.sin(az),            np.cos(az),            0         ],
        [-np.sin(el)*np.cos(az), -np.sin(el)*np.sin(az), np.cos(el)],
    ])
    projected = points @ R.T
    return projected[:, 0], projected[:, 1]


def get_pc_scores_for_morphospace(ax, num=5):
    xrange = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num)
    yrange = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num)
    return xrange, yrange


def plot_recon_morphs_3d(
    pca,
    efa,
    fig,
    ax,
    n_PCs_xy=[1, 2],
    morph_num=3,
    morph_scale=1.0,
    morph_color="lightgray",
    morph_alpha=0.7,
    view=(-60, 30),
):
    """Plot reconstructed shapes in PC space with oblique 2D projection."""
    pc_scores_h, pc_scores_v = get_pc_scores_for_morphospace(ax, morph_num)
    for pc_score_h in pc_scores_h:
        for pc_score_v in pc_scores_v:
            pc_score = np.zeros(pca.n_components_)
            n_PC_h, n_PC_v = n_PCs_xy
            pc_score[n_PC_h - 1] = pc_score_h
            pc_score[n_PC_v - 1] = pc_score_v

            arr_coef = pca.inverse_transform([pc_score])

            ax_width = ax.get_window_extent().width
            fig_width = fig.get_window_extent().width
            fig_height = fig.get_window_extent().height
            morph_size = morph_scale * ax_width / (fig_width * morph_num)
            loc = ax.transData.transform((pc_score_h, pc_score_v))
            axins = fig.add_axes(
                [
                    loc[0] / fig_width - morph_size / 2,
                    loc[1] / fig_height - morph_size / 2,
                    morph_size,
                    morph_size,
                ],
                anchor="C",
            )

            recon = efa.inverse_transform(arr_coef, norm=False)
            x, y = project_3d_to_2d(recon[0], azim=view[0], elev=view[1])

            axins.plot(
                x.astype(float), y.astype(float), color=morph_color, alpha=morph_alpha
            )
            axins.axis("equal")
            axins.axis("off")
```

```{code-cell} ipython3
morph_num = 5
morph_scale = 0.8
morph_color = "gray"
morph_alpha = 0.8
view = (-120, 60)


fig = plt.figure(figsize=(16, 16), dpi=200)

#########
# PC1 vs PC2
#########
ax = fig.add_subplot(2, 2, 1)
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="bending_angle",
    hue_order=bending_order,
    style="aspect_ratio",
    palette="Set2",
    ax=ax,
    legend=True,
)

plot_recon_morphs_3d(
    pca,
    efa,
    morph_num=morph_num,
    morph_scale=morph_scale,
    morph_color=morph_color,
    morph_alpha=morph_alpha,
    fig=fig,
    ax=ax,
    view=view,
)

ax.patch.set_alpha(0)
ax.set(xlabel="PC1", ylabel="PC2")

print("PC1-PC2 done")

#########
# PC2 vs PC3
#########
ax = fig.add_subplot(2, 2, 2)
sns.scatterplot(
    data=df_pca,
    x="PC2",
    y="PC3",
    hue="bending_angle",
    hue_order=bending_order,
    style="aspect_ratio",
    palette="Set2",
    ax=ax,
    legend=True,
)

plot_recon_morphs_3d(
    pca,
    efa,
    morph_num=morph_num,
    morph_scale=morph_scale,
    morph_color=morph_color,
    morph_alpha=morph_alpha,
    fig=fig,
    ax=ax,
    n_PCs_xy=[2, 3],
    view=view,
)

ax.patch.set_alpha(0)
ax.set(xlabel="PC2", ylabel="PC3")

print("PC2-PC3 done")

#########
# PC3 vs PC1
#########
ax = fig.add_subplot(2, 2, 3)
sns.scatterplot(
    data=df_pca,
    x="PC3",
    y="PC1",
    hue="bending_angle",
    hue_order=bending_order,
    style="aspect_ratio",
    palette="Set2",
    ax=ax,
    legend=True,
)

plot_recon_morphs_3d(
    pca,
    efa,
    morph_num=morph_num,
    morph_scale=morph_scale,
    morph_color=morph_color,
    morph_alpha=morph_alpha,
    fig=fig,
    ax=ax,
    n_PCs_xy=[3, 1],
    view=view,
)

ax.patch.set_alpha(0)
ax.set(xlabel="PC3", ylabel="PC1")

print("PC3-PC1 done")

#########
# Explained variance
#########
ax = fig.add_subplot(2, 2, 4)
explained_variance_ratio_plot(pca, ax=ax, verbose=True)
```
