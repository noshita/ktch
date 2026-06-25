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

# Raup's model

In this tutorial, you will generate coiled shells with Raup's model
(Raup & Michelson, 1965; Raup, 1966).

Raup's model describes a shell as a trajectory of a generating curve, which mimics the aperture shape, that
expands, rotates, and translates along a fixed coiling axis. Three parameters
control the geometry:

- whorl expansion rate `w_r`: how fast the tube widens per revolution
- translation rate `t_r`: how fast the coil moves down along the coiling axis
- relative distance of the generating curve from the coiling axis `d_r`

## Setup

```{code-cell} ipython3
# Uncomment if needed
# %pip install "ktch[plot]"
```

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ktch.coiling import RaupModel, l_r, raup, theta_r
```

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is only required for the Sphinx documentation build.
# You do not need this setting when running in Jupyter.
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
```

## Generate your first shell

The `raup` function takes the three parameters and returns a sampled surface. 
We also pass a sampling grid (`theta_range`, `phi_range`);
omit them to use the defaults.

```{code-cell} ipython3
theta = np.linspace(0.0, 2.0 * np.pi * 4.0, 240)  # four whorls
phi = np.linspace(0.0, 2.0 * np.pi, 60)

X = raup(w_r=1.3, t_r=1.5, d_r=0.2, theta_range=theta, phi_range=phi)
X.shape
```

You should see a `(n_theta, n_phi, 3)` array: `n_theta` samples along the coil,
`n_phi` samples around the tube, and the last axis holds the `(x, y, z)`
coordinates of each surface point.

## Visualize in 3D

Let's define a small helper that turns a surface array into an interactive plotly figure.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show definition: plot_surface3d()"
:  code_prompt_hide: "Hide definition: plot_surface3d()"

def plot_surface3d(X, *, title="", colorscale="Viridis", apex_up=True):
    """Plot a ``(n, n_phi, 3)`` surface array with plotly.

    With ``apex_up=True`` the form is rotated 180° about a horizontal axis so
    the apex points up (the conventional orientation). This is a rotation, not a
    ``z`` reflection, so the coiling direction is preserved.
    """
    X = np.asarray(X, dtype=float)
    if apex_up:
        X = X * np.array([-1.0, 1.0, -1.0])
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    fig = go.Figure(
        data=[go.Surface(x=x, y=y, z=z, colorscale=colorscale, showscale=False)]
    )
    fig.update_layout(
        title=title,
        width=700,
        height=500,
        scene=dict(aspectmode="data"),
    )
    return fig
```

```{code-cell} ipython3
fig = plot_surface3d(X, title="Raup's model (w_r=1.3, t_r=1.5, d_r=0.2)")
fig.show()
```

That's a coiled shell built from three parameters.

```{note}
Raup's model coils along the coiling axis (`+z`), so the apex sits at the bottom
of the raw output. `plot_surface3d` shows shells apex-up by default
(`apex_up=True`): it rotates the form 180° about a horizontal axis (a rotation,
not a `z`-flip, so the coiling direction is preserved). Pass `apex_up=False` for
the raw orientation.
```

## Explore the parameters

To see what each parameter does, we vary one at a time while holding the other
two fixed.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show definition: compare_raup()"
:  code_prompt_hide: "Hide definition: compare_raup()"

def compare_raup(values, *, vary, title, **fixed):
    """Generate Raup shells varying one parameter and show them side by side."""
    theta = np.linspace(0.0, 2.0 * np.pi * 4.0, 180)  # four whorls
    phi = np.linspace(0.0, 2.0 * np.pi, 60)  # aperture sampling
    fig = make_subplots(
        rows=1,
        cols=len(values),
        specs=[[{"type": "surface"}] * len(values)],
        subplot_titles=[f"{vary}={v}" for v in values],
        horizontal_spacing=0.01,
    )
    for i, v in enumerate(values, start=1):
        Xi = raup(theta_range=theta, phi_range=phi, **{**fixed, vary: v})
        Xi = Xi * np.array([-1.0, 1.0, -1.0])  # apex-up (see plot_surface3d note)
        fig.add_trace(
            go.Surface(
                x=Xi[..., 0],
                y=Xi[..., 1],
                z=Xi[..., 2],
                showscale=False,
                colorscale="Viridis",
            ),
            row=1,
            col=i,
        )
    scene_names = ["scene"] + [f"scene{k}" for k in range(2, len(values) + 1)]
    fig.update_layout(
        title=title,
        width=900,
        height=350,
        margin=dict(l=0, r=0, t=60, b=0),
        **{name: dict(aspectmode="data") for name in scene_names},
    )
    return fig
```

### Whorl expansion rate `w_r`

```{code-cell} ipython3
compare_raup([1.1, 2.0, 10.0], vary="w_r", title="Whorl expansion rate",
             t_r=1.0, d_r=0.0).show()
```

### Translation rate `t_r`

With `t_r = 0` the whorls stay in a single plane (a planispiral); increasing it
stacks the whorls into a high-spired form.

```{code-cell} ipython3
compare_raup([0.0, 1.0, 4.0], vary="t_r", title="Translation rate",
             w_r=2.0, d_r=0.0).show()
```

### Relative distance from the coiling axis `d_r`

`d_r` (in `(-1, 1)`) sets how far the generating curve sits from the coiling
axis.

```{code-cell} ipython3
compare_raup([-0.4, 0.0, 0.4], vary="d_r", title="Distance from the axis",
             w_r=2.0, t_r=1.0).show()
```

## Generate shells in batch

`RaupModel` provides the same model in a scikit-learn-style estimator. 
Its `inverse_transform` is the generative map from parameters to form.

```{code-cell} ipython3
model = RaupModel()
params = np.array(
    [
        [1.2, 0.5, 0.1],
        [1.5, 1.5, 0.2],
        [2.0, 0.0, 0.05],
    ]
)
surfaces = model.inverse_transform(params, theta_range=theta, phi_range=phi)
surfaces.shape
```

The leading axis indexes the three parameter rows; the rest is the familiar
`(n_theta, n_phi, 3)` surface.

```{code-cell} ipython3
plot_surface3d(surfaces[1], title="Second parameter set").show()
```

For analysis pipelines, pass `as_frame=True` to get a tidy long-format
`DataFrame` indexed by `(specimen_id, trajectory_id, phi_id)`.

```{code-cell} ipython3
df = model.inverse_transform(
    params, theta_range=theta, phi_range=phi, as_frame=True
)
df.head()
```

## Sample evenly along the shell

By default the surface is sampled at equal steps in the coiling angle `theta`.
Because the shell is a logarithmic spiral, equal `theta` steps pack many points
near the tight apex and stretch them apart at the wide aperture. 
For even meshes, sample at equal steps of arc length along the shell instead.

`l_r` and `theta_r` convert between the coiling angle and the trajectory arc
length `l`. To space `n_theta` samples evenly in arc length, take a uniform grid
in `l` and map it back to `theta` with `theta_r`:

```{code-cell} ipython3
w_r, t_r, d_r = 2.0, 0.7, 0.05
n_whorls = 4.0
n_theta = 48
phi = np.linspace(0.0, 2.0 * np.pi, 40)

theta_uniform = np.linspace(0.0, 2.0 * np.pi * n_whorls, n_theta)

l_max = l_r(2.0 * np.pi * n_whorls, w_r, t_r, d_r)
theta_arclength = theta_r(np.linspace(0.0, l_max, n_theta), w_r, t_r, d_r)
```

```{code-cell} ipython3
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=["equal steps in theta", "equal steps in arc length"],
    horizontal_spacing=0.01,
)
for col, theta in enumerate([theta_uniform, theta_arclength], start=1):
    Xi = raup(w_r, t_r, d_r, theta_range=theta, phi_range=phi)
    Xi = Xi * np.array([-1.0, 1.0, -1.0])  # apex-up (see plot_surface3d note)
    fig.add_trace(
        go.Surface(
            x=Xi[..., 0],
            y=Xi[..., 1],
            z=Xi[..., 2],
            showscale=False,
            colorscale="Viridis",
        ),
        row=1,
        col=col,
    )
fig.update_layout(
    width=900,
    height=450,
    margin=dict(l=0, r=0, t=40, b=0),
    scene=dict(aspectmode="data"),
    scene2=dict(aspectmode="data"),
)
fig.show()
```

Notice how the mesh on the right is spaced much more evenly along the coil. 

```{seealso}
- {doc}`growing_tube_model` to generate shells with the growing tube model,
  including heteromorph (non-constant) growth.
- {doc}`../../explanation/coiling` for the theory behind Raup's model.
```

## References

- Raup, D.M., Michelson, A., 1965. Theoretical Morphology of the Coiled Shell.
  Science 147, 1294–1295.
- Raup, D.M., 1966. Geometric analysis of shell coiling: general problems.
  Journal of Paleontology 40, 1178–1190.
