---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-load-spharm)=

# Load SPHARM Coefficients

ktch can read spherical harmonic coefficients from SPHARM-PDM `.coef`
files, the output of the `ParaToSPHARMMesh` step of
[SPHARM-PDM](https://www.nitrc.org/projects/spharm-pdm).
This guide reads a sample `.coef` file and reconstructs the 3D shape
encoded by its coefficients.

## Read SPHARM-PDM coefficients

```{code-cell} ipython3
from ktch.datasets import fetch
from ktch.io import read_spharmpdm_coef

coef_path = fetch("danshaku_08_allSegments_SPHARM.coef")
data = read_spharmpdm_coef(coef_path)

print(f"specimen={data.specimen_name}, l_max={data.l_max}, "
      f"shape={data.to_numpy().shape}")
```

`data.coeffs[l]` holds the complex coefficients of degree `l` with shape
`(2*l+1, 3)`. See {doc}`../../explanation/harmonic` for the convention.

## Reconstruct the surface

Convert the SPHARM-PDM coefficients to the real basis used by
{class}`~ktch.harmonic.SphericalHarmonicAnalysis`, then call
`inverse_transform` to evaluate the reconstructed surface on a
`(theta, phi)` grid.

```{code-cell} ipython3
from ktch.harmonic import SphericalHarmonicAnalysis
from ktch.io import spharmpdm_to_sha_coeffs

coeffs = spharmpdm_to_sha_coeffs(data)

sha = SphericalHarmonicAnalysis(n_harmonics=data.l_max)
X_coords = sha.inverse_transform(coeffs)
print(f"surface grid shape: {X_coords.shape}")  # (1, n_theta, n_phi, 3)
```

To use a different angular resolution, pass `theta_range` and
`phi_range` to `inverse_transform`.

## Plot the 3D shape

```{code-cell} ipython3
import plotly.graph_objects as go
```

```{code-cell} ipython3
:tags: [remove-cell]

# Required only for the Sphinx documentation build.
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
```

```{code-cell} ipython3
x, y, z = X_coords[0].T

fig = go.Figure(
    data=[
        go.Surface(x=x, y=y, z=z, opacity=0.8, showscale=False),
    ]
)

fig.update_layout(
    width=700,
    height=700,
    autosize=False,
    scene=dict(
        camera=dict(
            up=dict(x=0, y=0, z=1),
            eye=dict(x=1.1, y=1.1, z=1.1),
        ),
        aspectmode="data",
    ),
)

fig.show()
```

```{seealso}
- {doc}`../../tutorials/harmonic/spharm` to compute SPHARM coefficients
  from a 3D surface mesh.
- {doc}`../../explanation/harmonic` for background on spherical harmonic
  analysis.
```
