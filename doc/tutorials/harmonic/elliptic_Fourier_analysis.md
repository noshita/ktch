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

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis
from ktch.plot import explained_variance_ratio_plot
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
df_pca["specimen_id"] = [i for i in range(1, len(pcscores) + 1)]
df_pca = df_pca.set_index("specimen_id")
df_pca = df_pca.join(data_outline_mosquito_wings.meta)
df_pca = df_pca.rename(columns={i: ("PC" + str(i + 1)) for i in range(12)})
df_pca
```

```{code-cell} ipython3
fig, ax = plt.subplots()
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="genus", ax=ax, palette="Paired")
```

## Morphospace

```{code-cell} ipython3
def get_pc_scores_for_morphospace(ax, num=5):
    xrange = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], num)
    yrange = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], num)
    return xrange, yrange


def plot_recon_morphs(
    pca,
    efa,
    fig,
    ax,
    n_PCs_xy=[1, 2],
    morph_num=3,
    morph_scale=1.0,
    morph_color="lightgray",
    morph_alpha=0.7,
):
    pc_scores_h, pc_scores_v = get_pc_scores_for_morphospace(ax, morph_num)
    print("PC_h: ", pc_scores_h, ", PC_v: ", pc_scores_v)
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

            coords = efa.inverse_transform(arr_coef)
            x = coords[0][:, 0]
            y = coords[0][:, 1]

            axins.plot(
                x.astype(float), y.astype(float), color=morph_color, alpha=morph_alpha
            )
            axins.axis("equal")
            axins.axis("off")
```

```{code-cell} ipython3
fig = plt.figure(figsize=(16, 16), dpi=200)

ax = fig.add_subplot(2, 2, 1)
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="genus",
    hue_order=None,
    palette="Paired",
    ax=ax,
    legend=True,
)

plot_recon_morphs(pca, efa, morph_num=5, morph_scale=0.5, fig=fig, ax=ax)
```

```{code-cell} ipython3
morph_num = 5
morph_scale = 0.8
morph_color = "gray"
morph_alpha = 0.8

hue_order = df_pca["genus"].unique()

fig = plt.figure(figsize=(16, 16), dpi=200)

#########
# PC1
#########
ax = fig.add_subplot(2, 2, 1)
sns.scatterplot(
    data=df_pca,
    x="PC1",
    y="PC2",
    hue="genus",
    hue_order=hue_order,
    palette="Paired",
    ax=ax,
    legend=True,
)

plot_recon_morphs(
    pca,
    efa,
    morph_num=5,
    morph_scale=morph_scale,
    morph_color=morph_color,
    morph_alpha=morph_alpha,
    fig=fig,
    ax=ax,
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
    x="PC2",
    y="PC3",
    hue="genus",
    hue_order=hue_order,
    palette="Paired",
    ax=ax,
    legend=True,
)

plot_recon_morphs(
    pca,
    efa,
    morph_num=5,
    morph_scale=morph_scale,
    morph_color=morph_color,
    morph_alpha=morph_alpha,
    fig=fig,
    ax=ax,
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
    x="PC3",
    y="PC1",
    hue="genus",
    hue_order=hue_order,
    palette="Paired",
    ax=ax,
    legend=True,
)

plot_recon_morphs(
    pca,
    efa,
    morph_num=5,
    morph_scale=morph_scale,
    morph_color=morph_color,
    morph_alpha=morph_alpha,
    fig=fig,
    ax=ax,
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
