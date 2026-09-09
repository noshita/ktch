---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

(how-to-sample-evenly-along-shell)=

# Sample evenly along a coiled shell

Space the surface samples of a coiling model at equal steps of arc length
along the shell instead of equal steps in the model's growth parameter.

By default, `ktch.coiling` samples a shell at equal steps of the coiling angle
`theta` (Raup's model) or the growth stage `s` (growing tube model). Because
the tube grows exponentially, equal steps give a dense sampling near the tight
apex and a sparse one at the wide aperture. The conversion functions below map
a uniform grid in arc length `l` back to the model's parameter.

## Prerequisites

- ktch installed with plotting extras (`pip install "ktch[plot]"`)
- Familiarity with {doc}`../../tutorials/coiling/raup_model` or
  {doc}`../../tutorials/coiling/growing_tube_model`

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ktch.coiling import (
    GrowingTubeModel,
    RaupModel,
    l_g,
    l_r,
    s_g,
    theta_r,
)
```

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is only required for the Sphinx documentation build.
# You do not need this setting when running in Jupyter.
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
```

## Raup's model

`l_r` gives the trajectory arc length at a coiling angle, and `theta_r` inverts
it. Take a uniform grid in `l` up to the arc length of the last whorl and map
it back to `theta`:

```{code-cell} ipython3
w_r, t_r, d_r = 2.0, 0.7, 0.05
n_whorls = 4.0
n_theta = 48

theta_uniform = np.linspace(0.0, 2.0 * np.pi * n_whorls, n_theta)

l_max = l_r(theta_uniform[-1], w_r, t_r, d_r)
theta_arclength = theta_r(np.linspace(0.0, l_max, n_theta), w_r, t_r, d_r)
```

Pass the converted grid as `theta_range`:

```{code-cell} ipython3
phi = np.linspace(0.0, 2.0 * np.pi, 40)
raup_model = RaupModel()

X_uniform = raup_model.inverse_transform(
    [w_r, t_r, d_r], theta_range=theta_uniform, phi_range=phi
)
X_arclength = raup_model.inverse_transform(
    [w_r, t_r, d_r], theta_range=theta_arclength, phi_range=phi
)
```

## Growing tube model

`l_g` and `s_g` convert between the growth stage and the arc length. The arc
length depends only on the expansion rate `e_g` (and `r0`), not on the
curvature `c_g` or torsion `t_g`:

```{code-cell} ipython3
e_g, c_g, t_g = 0.04, 0.4, 0.06
d_g = np.hypot(c_g, t_g)  # one whorl advances s by 2*pi / d_g
n_s = 60

s_uniform = np.linspace(0.0, 2.0 * np.pi * n_whorls / d_g, n_s)

l_max = l_g(s_uniform[-1], e_g)
s_arclength = s_g(np.linspace(0.0, l_max, n_s), e_g)
```

Pass the converted grid as `s_range`:

```{code-cell} ipython3
growing_tube_model = GrowingTubeModel(method="closed")

Y_uniform = growing_tube_model.inverse_transform(
    [e_g, c_g, t_g], s_range=s_uniform, phi_range=phi
)
Y_arclength = growing_tube_model.inverse_transform(
    [e_g, c_g, t_g], s_range=s_arclength, phi_range=phi
)
```

## Visualization

Compare the meshes side by side. The arc-length grids space the rings evenly
along the coil.

```{code-cell} ipython3
panels = [
    (X_uniform, "Raup: equal steps in theta"),
    (X_arclength, "Raup: equal steps in arc length"),
    (Y_uniform, "Growing tube: equal steps in s"),
    (Y_arclength, "Growing tube: equal steps in arc length"),
]
fig = make_subplots(
    rows=2,
    cols=2,
    specs=[[{"type": "surface"}] * 2] * 2,
    subplot_titles=[title for _, title in panels],
    horizontal_spacing=0.01,
    vertical_spacing=0.05,
)
for i, (Xi, _) in enumerate(panels):
    if i < 2:  # Raup's model: show apex-up (rotation, not a z-flip)
        Xi = Xi * np.array([-1.0, 1.0, -1.0])
    fig.add_trace(
        go.Surface(
            x=Xi[..., 0], y=Xi[..., 1], z=Xi[..., 2],
            showscale=False, colorscale="Viridis",
        ),
        row=i // 2 + 1,
        col=i % 2 + 1,
    )
fig.update_layout(
    width=900,
    height=800,
    margin=dict(l=0, r=0, t=40, b=0),
    **{name: dict(aspectmode="data") for name in ["scene", "scene2", "scene3", "scene4"]},
)
fig.show()
```

The converted grids take large steps near the apex, where a revolution is
short, and small steps near the aperture, where a revolution is long:

```{code-cell} ipython3
step_theta = np.degrees(np.diff(theta_arclength))
step_s = np.diff(s_arclength)
print(f"Raup, theta step:   first {step_theta[0]:.1f} deg, last {step_theta[-1]:.1f} deg")
print(f"Growing tube, s step: first {step_s[0]:.2f}, last {step_s[-1]:.2f}")
```

## See also

- {doc}`../../tutorials/coiling/raup_model` and
  {doc}`../../tutorials/coiling/growing_tube_model` for the models
- {doc}`../../explanation/coiling` for the theory of the coiling models
