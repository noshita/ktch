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

# Disk Harmonic Analysis

```{code-cell} ipython3
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns

from sklearn.decomposition import PCA

from ktch.datasets import load_surface_leaf_bending
from ktch.harmonic import DiskHarmonicAnalysis, xy2polar
```

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is only required for the Sphinx documentation build.
# You do not need this setting when running in Jupyter.
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
```

## Load the surface leaf bending dataset

This dataset contains 60 synthetic 3D leaf surfaces with bending deformation,
each paired with a disk parameterization mesh (unit disk, z=0).

```{code-cell} ipython3
data = load_surface_leaf_bending()

print(f"Number of specimens: {len(data.vertices)}")
print(f"Specimen 1 — vertices: {data.vertices[0].shape}, faces: {data.faces[0].shape}")
print(f"Specimen 1 — param vertices: {data.param_vertices[0].shape}")
```

```{code-cell} ipython3
df_meta = pd.DataFrame(data.meta)
df_meta["bending_angle"] = (
    np.degrees(df_meta["alphaB"]).round().astype(int).astype(str) + "\u00b0"
)
df_meta["aspect_ratio"] = df_meta["alpha"]
df_meta
```

## Visualize 3D surfaces

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show definition: plot_mesh3d()"
:  code_prompt_hide: "Hide definition: plot_mesh3d()"

def plot_mesh3d(vertices, faces, *, title="", opacity=0.5, color="lightblue"):
    """Plot a triangular mesh with plotly."""
    I, J, K = faces.T
    x, y, z = vertices.T

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x, y=y, z=z,
                i=I, j=J, k=K,
                opacity=opacity,
                color=color,
                showscale=False,
            ),
        ]
    )
    fig.update_layout(
        title=title,
        width=700, height=500,
        scene=dict(aspectmode="data"),
    )
    return fig
```

```{code-cell} ipython3
representative_ids = [0, 10, 20, 30, 40, 50]
bending_order = [str(deg) + "\u00b0" for deg in [20, 80, 140]]

for sid in representative_ids:
    label = (
        f"Specimen {sid + 1} "
        f"(bending={df_meta.iloc[sid]['bending_angle']}, "
        f"\u03b1={df_meta.iloc[sid]['aspect_ratio']})"
    )
    fig = plot_mesh3d(
        data.vertices[sid], data.faces[sid], title=label
    )
    fig.show()
```

### Parameter mesh (disk parameterization)

The parameter mesh maps each surface vertex to a point on the unit disk.

```{code-cell} ipython3
fig = plot_mesh3d(
    data.param_vertices[0], data.param_faces[0],
    title="Parameter mesh (specimen 1)", opacity=0.3,
)
fig.show()
```

## Prepare disk parameterization

DHA requires polar coordinates `(r, theta)` on the unit disk. We extract the
`(x, y)` coordinates from the parameter mesh and convert them using `xy2polar`.

```{code-cell} ipython3
r_theta = [
    xy2polar(data.param_vertices[i][:, :2], centered=True)
    for i in range(len(data.vertices))
]

print(f"r_theta[0].shape: {r_theta[0].shape}")
print(f"r range: [{r_theta[0][:, 0].min():.4f}, {r_theta[0][:, 0].max():.4f}]")
print(f"theta range: [{r_theta[0][:, 1].min():.4f}, {r_theta[0][:, 1].max():.4f}]")
```

```{code-cell} ipython3
X = data.vertices
```

## Disk Harmonic Analysis

```{code-cell} ipython3
dha = DiskHarmonicAnalysis(n_harmonics=15, n_dim=3)
```

```{code-cell} ipython3
coef = dha.fit_transform(X, r_theta=r_theta)
coef.shape
```

### Reconstruction from coefficients

Reconstructing with different numbers of harmonics shows how DHA captures
shape at different levels of detail.  The `inverse_transform` method
returns a regular `(n_theta, n_r, 3)` grid for each specimen.

```{code-cell} ipython3
sid = 0
I, J, K = data.faces[sid].T
x_orig, y_orig, z_orig = data.vertices[sid].T

n_max_values = [5, 10, 15]

for n_max in n_max_values:
    surf = dha.inverse_transform(coef[sid:sid+1], n_max=n_max)
    x_r, y_r, z_r = surf[0].transpose(2, 0, 1)

    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=x_orig, y=y_orig, z=z_orig,
                i=I, j=J, k=K,
                opacity=0.3,
                color="lightblue",
                showscale=False,
                name="original",
            ),
            go.Surface(
                x=x_r, y=y_r, z=z_r,
                opacity=0.5,
                showscale=False,
                name=f"n_max={n_max}",
            ),
        ]
    )
    fig.update_layout(
        title=f"Reconstruction with n_max={n_max}",
        width=700, height=500,
        scene=dict(aspectmode="data"),
    )
    fig.show()
```
