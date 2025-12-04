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

# 3D Elliptic Fourier Analysis

```{code-cell} ipython3
import urllib

import numpy as np
import pandas as pd

import plotly.express as px

from ktch.harmonic import EllipticFourierAnalysis
```

## 3D coordinate values of a leaf edge

```{code-cell} ipython3
resp = urllib.request.urlopen(
    "https://strata.morphometrics.jp/examples/rolling_alpha_016_nIntervals_64.csv"
)
arr_coord = np.loadtxt(resp)
df_coord = pd.DataFrame(arr_coord, columns=["x", "y", "z"])
```

```{code-cell} ipython3
fig = px.line_3d(df_coord, x="x", y="y", z="z")
fig.update_layout(scene=dict(aspectmode="data"))
fig.show()
```

## 3D EFA

```{code-cell} ipython3
efa3d = EllipticFourierAnalysis(n_harmonics=20, n_dim=3)
```

```{code-cell} ipython3
coef = efa3d.fit_transform([arr_coord], norm=False).reshape(-1, 6, 21)
print("coefficients (a, b, c, d, e, f) x (n_harmonics + 1)", coef[0].shape)
coef[0]
```

## Reconstruction of 3D coordinate values from Fourier coefficients

```{code-cell} ipython3
arr_coord_recon = efa3d.inverse_transform(coef, t_num=600)
df_coord_recon = pd.DataFrame(arr_coord_recon[0], columns=["x", "y", "z"])
```

```{code-cell} ipython3
fig = px.line_3d(df_coord_recon, x="x", y="y", z="z")
fig.update_layout(scene=dict(aspectmode="data"))
fig.show()
```

```{code-cell} ipython3

```
