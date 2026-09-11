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

## Register precomputed coefficients

SPHARM-PDM coefficients might include each specimen's position,
orientation, and size. Before shape comparison, register them with
{class}`~ktch.harmonic.SphericalHarmonicRegistration`, which removes that
information (if still present) from the coefficients without recomputing
them from the surface. `method="first_order"` uses the degree-1 ellipsoid
to align orientation and the parameter sphere; `scale=False` keeps size.

```{code-cell} ipython3
from ktch.harmonic import SphericalHarmonicRegistration

reg = SphericalHarmonicRegistration(method="first_order", scale=False)
registered = reg.fit_transform(coeffs)
print(f"registered coefficients shape: {registered.shape}")
```

Because it maps coefficients to coefficients, it composes in a
scikit-learn `Pipeline`.

## Reproduce SPHARM-PDM's ellipsoid alignment

In a `*_SPHARM.coef` file of SPHARM-PDM, the parameter sphere is already aligned to the
degree-1 ellipsoid. The `*_SPHARM_ellalign.coef` file additionally rotates
the coordinates so that the ellipsoid axes fall on x, y, and z. That
second step touches the coordinates only, which is what
`align_parameter=False` does:

```{code-cell} ipython3
ellalign = SphericalHarmonicRegistration(
    method="first_order", scale=False, align_parameter=False
).fit_transform(coeffs)
```

This keeps the file's own choice among the eight sign patterns of the
parameter sphere. It therefore agrees with `*_SPHARM_ellalign.coef` for
near-symmetric shapes too, where the default may settle on a different,
equally valid frame of the same shape. It warns when the degree-1 columns
are not orthogonal, which means the parameter sphere was not aligned
upstream; use the default in that case. See
{doc}`../../explanation/harmonic` for the conventions.

One convention differs: SPHARM-PDM's frame is ktch's turned by a half turn
about z, a sign convention of its basis functions. The turn is the same
for every specimen and changes nothing in a shape analysis. To write the
file SPHARM-PDM would have written, turn the result:

```{code-cell} ipython3
import numpy as np

from ktch.harmonic import rotate_spharm_coeffs

spharm_pdm_frame = rotate_spharm_coeffs(
    ellalign, np.diag([-1.0, -1.0, 1.0]), domain="codomain"
)
```

Register every specimen with one and the same call, whichever file it
comes from; a specimen taken from an `_ellalign.coef` file without
registration sits a half turn away from the registered ones.

## Write coefficients to a `.coef` file

Convert flat coefficient vectors back to
{class}`~ktch.io.SpharmPdmData` with
{func}`~ktch.io.sha_coeffs_to_spharmpdm`, then write each specimen with
{func}`~ktch.io.write_spharmpdm_coef`:

```{code-cell} ipython3
import tempfile
from pathlib import Path

from ktch.io import sha_coeffs_to_spharmpdm, write_spharmpdm_coef

out_dir = Path(tempfile.mkdtemp())
for item in sha_coeffs_to_spharmpdm(spharm_pdm_frame, [data.specimen_name]):
    write_spharmpdm_coef(out_dir / f"{item.specimen_name}_ellalign.coef", item)

sorted(p.name for p in out_dir.iterdir())
```

By default the writer uses full precision, and reading the file back gives
the same numbers. Pass `precision=6` to reproduce SPHARM-PDM's own layout.
Registration survives the conversion in both directions. Do not register
again after reading a registered file back; in a pipeline, use
`method=None` as the pass-through.

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
