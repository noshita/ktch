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
from ktch.plot import explained_variance_ratio_plot, morphospace_plot
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
efa = EllipticFourierAnalysis(n_harmonics=20, n_dim=3, norm=False)
```

```{code-cell} ipython3
coef = efa.fit_transform(coords)
coef.shape
```

### Reconstruction from coefficients

```{code-cell} ipython3
coords_recon = efa.inverse_transform(coef, t_num=200)
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
efa = EllipticFourierAnalysis(n_harmonics=20, n_dim=3, norm=True)
```

```{code-cell} ipython3
coef = efa.fit_transform(coords)
coef.shape
```

### Reconstruction from coefficients

```{code-cell} ipython3
coords_recon = efa.inverse_transform(coef, t_num=200)
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
def render_curve_3d_fixed_view(coords, ax, *, color="gray", alpha=0.7, **kw):
    kw.pop("links", None)
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], color=color, alpha=alpha)
    ptp = [max(np.max(coords[:, i]) - np.min(coords[:, i]), 1e-10) for i in range(3)]
    ax.set_box_aspect(ptp)
    ax.view_init(elev=60, azim=-120)
```

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2, figsize=(16, 16), dpi=200)

for ax, (i, j) in zip(axes.flat[:3], [(0, 1), (1, 2), (2, 0)]):
    morphospace_plot(
        data=df_pca,
        x=f"PC{i + 1}", y=f"PC{j + 1}",
        hue="bending_angle", hue_order=bending_order,
        palette="Set2",
        reducer=pca,
        descriptor=efa,
        shape_type="curve_3d",
        render_fn=render_curve_3d_fixed_view,
        components=(i, j),
        n_shapes=5,
        shape_scale=0.8,
        shape_color="gray",
        shape_alpha=0.8,
        ax=ax,
        scatter_kw=dict(style="aspect_ratio"),
    )

explained_variance_ratio_plot(pca, ax=axes[1, 1], verbose=True)
```
