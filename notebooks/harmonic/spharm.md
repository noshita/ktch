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

# Spherical Harmonic (SPHARM) Analysis

```{code-cell} ipython3
from pathlib import Path
import shutil
import urllib
import tempfile

import numpy as np

import plotly.graph_objects as go

import vtk
from vtk.numpy_interface import dataset_adapter as dsa

from ktch.harmonic import SphericalHarmonicAnalysis, xyz2spherical
```

## Load 3D potato surface data

```{code-cell} ipython3
# parameter
with urllib.request.urlopen(
    "https://strata.morphometrics.jp/examples/andesred_07_allSegments_para.vtk"
) as response:
    with tempfile.NamedTemporaryFile() as tmp_file:
        shutil.copyfileobj(response, tmp_file)
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(tmp_file.name)
        reader.Update()

        dataset = reader.GetOutput()
        obj_para = dsa.WrapDataObject(dataset)

# surface
with urllib.request.urlopen(
    "https://strata.morphometrics.jp/examples/andesred_07_allSegments_surf.vtk"
) as response:
    with tempfile.NamedTemporaryFile() as tmp_file:
        shutil.copyfileobj(response, tmp_file)
        reader = vtk.vtkDataSetReader()
        reader.SetFileName(tmp_file.name)
        reader.Update()

        dataset = reader.GetOutput()
        obj_surf = dsa.WrapDataObject(dataset)
```

```{code-cell} ipython3
arr_para_faces = np.array(obj_para.Polygons.reshape(-1, 4)[:, 1:])
arr_para = np.array(obj_para.Points)

arr_surf_faces = np.array(obj_surf.Polygons.reshape(-1, 4)[:, 1:])
arr_surf = np.array(obj_surf.Points)
```

### Parameter mesh

```{code-cell} ipython3
I, J, K = arr_para_faces.T
x_surf, y_surf, z_surf = arr_para.T

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=x_surf,
            y=y_surf,
            z=z_surf,
            i=I,
            j=J,
            k=K,
            opacity=0.3,
            showscale=False,
        ),
    ]
)

fig.update_layout(
    width=700,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(
                x=1.1,
                y=1.1,
                z=1.1,
            ),
        ),
        aspectmode="data",
    ),
)

fig.show()
```

### Surface mesh

```{code-cell} ipython3
I, J, K = arr_surf_faces.T
x_surf, y_surf, z_surf = arr_surf.T

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=x_surf,
            y=y_surf,
            z=z_surf,
            i=I,
            j=J,
            k=K,
            opacity=0.3,
            showscale=False,
        ),
    ]
)

fig.update_layout(
    width=700,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(
                x=1.1,
                y=1.1,
                z=1.1,
            ),
        ),
        aspectmode="data",
    ),
)

fig.show()
```

## SPHARM Analysis

```{code-cell} ipython3
spharm_analysis = SphericalHarmonicAnalysis(n_harmonics=10)
X_transform = spharm_analysis.fit_transform([arr_surf], [xyz2spherical(arr_para)])
```

```{code-cell} ipython3
X_transform
```

## Inverse analysis

Reconstruct 3D surface shapes from SPHARM coefficients.

```{code-cell} ipython3
X_coords = spharm_analysis.inverse_transform(X_transform.reshape(1, 3, -1))
```

```{code-cell} ipython3
x_sph, y_sph, z_sph = np.real(X_coords[0].T)

fig = go.Figure(
    data=[
        go.Surface(
            x=x_sph,
            y=y_sph,
            z=z_sph,
            opacity=0.3,
            showscale=False,
        ),
    ]
)

fig.update_layout(
    width=700,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(
                x=1.1,
                y=1.1,
                z=1.1,
            ),
        ),
        aspectmode="data",
    ),
)

fig.show()
```

### Original vs reconstructed shapes
Overlay the original and reconstructed surface data.
This demonstrates that the original shape is well reproduced.

```{code-cell} ipython3
I, J, K = arr_surf_faces.T
x_surf, y_surf, z_surf = arr_surf.T

x_sph, y_sph, z_sph = np.real(X_coords[0].T)


fig = go.Figure(
    data=[
        go.Mesh3d(
            x=x_surf,
            y=y_surf,
            z=z_surf,
            i=I,
            j=J,
            k=K,
            opacity=0.8,
            showscale=False,
        ),
        go.Surface(
            x=x_sph,
            y=y_sph,
            z=z_sph,
            opacity=0.4,
            showscale=False,
        ),
    ]
)

fig.update_layout(
    width=700,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(
                x=1.1,
                y=1.1,
                z=1.1,
            ),
        ),
        aspectmode="data",
    ),
)

fig.show()
```

```{code-cell} ipython3

```
