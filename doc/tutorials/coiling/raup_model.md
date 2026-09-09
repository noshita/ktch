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

In this tutorial, you will use Raup's model (Raup & Michelson, 1965; Raup, 1966)
in both directions: generate coiled shells from parameters and estimate the parameters from a shell.

Raup's model describes a shell as a trajectory of a generating curve, which
mimics the aperture shape, that expands, rotates, and translates along a fixed
coiling axis. Three parameters control the geometry:

- whorl expansion rate `w_r`: how fast the tube widens per revolution
- translation rate `t_r`: how fast the coil moves down along the coiling axis
- relative distance of the generating curve from the coiling axis `d_r`

## Setup

```{code-cell} ipython3
# Uncomment if needed
# %pip install "ktch[plot]"
```

```{code-cell} ipython3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ktch.coiling import RaupModel
from ktch.plot import morphospace_plot
```

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is only required for the Sphinx documentation build.
# You do not need this setting when running in Jupyter.
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
```

## Generate your first shell

`RaupModel` provides the model. Its `inverse_transform` takes the parameters
and returns a sampled surface.
We also pass a sampling grid (`theta_range` along the growth trajectory, `phi_range` around
the aperture).

```{code-cell} ipython3
theta = np.linspace(0.0, 2.0 * np.pi * 4.0, 240)  # four whorls
phi = np.linspace(0.0, 2.0 * np.pi, 60)

raup_model = RaupModel()
X = raup_model.inverse_transform([1.3, 1.5, 0.2], theta_range=theta, phi_range=phi)
X.shape
```

You should see a `(n_theta, n_phi, 3)` array: `n_theta` samples along the coil,
`n_phi` samples around the tube, and the last axis holds the `(x, y, z)`
coordinates of each surface point.

## Visualize in 3D

Let's define a small helper that turns a surface array into an interactive
plotly figure.

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

## Orientation of the aperture

The full parameter vector of `RaupModel` has five entries. Ask the model for
their names:

```{code-cell} ipython3
raup_model.get_feature_names_out()
```

The last two, `delta_r` and `gamma_r`, set the orientation of the aperture
(Noshita, 2014). With `(0, 0)` the aperture lies in the radial-axial plane, as
in the classical model; `delta_r` tilts it about the radial direction and
`gamma_r` turns it about the coiling axis. A 3-entry input, as above, sets both
to zero.

To compare several shells at once, we use a second helper that lays out a
batch of surfaces in a grid of subplots.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show definition: plot_surface_grid()"
:  code_prompt_hide: "Hide definition: plot_surface_grid()"

def plot_surface_grid(surfaces, subplot_titles, *, ncols=None, title="",
                      colorscale="Viridis", apex_up=True):
    """Lay out ``(n_theta, n_phi, 3)`` surface arrays in a grid of subplots.

    With ``apex_up=True`` each form is rotated 180° about a horizontal axis so
    the apex points up (the conventional orientation). This is a rotation, not a
    ``z`` reflection, so the coiling direction is preserved.
    """
    n = len(surfaces)
    ncols = n if ncols is None else ncols
    nrows = int(np.ceil(n / ncols))
    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=[[{"type": "surface"}] * ncols] * nrows,
        subplot_titles=list(subplot_titles),
        horizontal_spacing=0.01,
        vertical_spacing=0.05,
    )
    scenes = {}
    for i, Xi in enumerate(surfaces):
        Xi = np.asarray(Xi, dtype=float)
        if apex_up:
            Xi = Xi * np.array([-1.0, 1.0, -1.0])
        row, col = divmod(i, ncols)
        fig.add_trace(
            go.Surface(
                x=Xi[..., 0],
                y=Xi[..., 1],
                z=Xi[..., 2],
                showscale=False,
                colorscale=colorscale,
            ),
            row=row + 1,
            col=col + 1,
        )
        scenes["scene" if i == 0 else f"scene{i + 1}"] = dict(aspectmode="data")
    fig.update_layout(
        title=title,
        width=900,
        height=350 * nrows,
        margin=dict(l=0, r=0, t=60, b=0),
        **scenes,
    )
    return fig
```

Passing a `(n_samples, 5)` array to `inverse_transform` generates one surface
per row, so a comparison is a single call:

```{code-cell} ipython3
params = np.array(
    [
        [1.3, 1.5, 0.2, 0.0, 0.0],
        [1.3, 1.5, 0.2, 0.4, 0.0],
        [1.3, 1.5, 0.2, 0.0, 0.4],
    ]
)
surfaces = raup_model.inverse_transform(params, theta_range=theta, phi_range=phi)
surfaces.shape
```

```{code-cell} ipython3
plot_surface_grid(
    surfaces,
    ["delta_r=0, gamma_r=0", "delta_r=0.4, gamma_r=0", "delta_r=0, gamma_r=0.4"],
    title="Orientation of the aperture",
).show()
```

The leading axis indexes the parameter rows; the rest is the familiar
`(n_theta, n_phi, 3)` surface.

## Explore the parameters

We vary one parameter while holding the other two constant to see each effect.

### Whorl expansion rate `w_r`

```{code-cell} ipython3
params = np.array([[1.1, 1.0, 0.0], [2.0, 1.0, 0.0], [10.0, 1.0, 0.0]])
plot_surface_grid(
    raup_model.inverse_transform(params, theta_range=theta, phi_range=phi),
    ["w_r=1.1", "w_r=2.0", "w_r=10.0"],
    title="Whorl expansion rate",
).show()
```

### Translation rate `t_r`

With `t_r = 0` the whorls stay in a single plane (a planispiral); increasing it
stacks the whorls into a high-spired form.

```{code-cell} ipython3
params = np.array([[2.0, 0.0, 0.0], [2.0, 1.0, 0.0], [2.0, 4.0, 0.0]])
plot_surface_grid(
    raup_model.inverse_transform(params, theta_range=theta, phi_range=phi),
    ["t_r=0.0", "t_r=1.0", "t_r=4.0"],
    title="Translation rate",
).show()
```

### Relative distance from the coiling axis `d_r`

`d_r` (in `(-1, 1)`) sets how far the generating curve sits from the coiling
axis.

```{code-cell} ipython3
params = np.array([[2.0, 1.0, -0.4], [2.0, 1.0, 0.0], [2.0, 1.0, 0.4]])
plot_surface_grid(
    raup_model.inverse_transform(params, theta_range=theta, phi_range=phi),
    ["d_r=-0.4", "d_r=0.0", "d_r=0.4"],
    title="Distance from the axis",
).show()
```

## Estimate the parameters

Now, the reverse direction. `transform` takes a shell surface and estimates the
parameters from it. The `"surface"` method fits the model directly
to a structured surface.

```{note}
The default `estimator="ml_2d"` expects lateral and umbilical measurements of a
specimen, the conventional measurements of Raup's model, and is not covered here.
```

We generate a shell with a known parameter vector, including the orientation,
and then try to estimate it.

```{code-cell} ipython3
params_true = np.array([1.3, 1.5, 0.2, 0.3, -0.2])
X = raup_model.inverse_transform(params_true, theta_range=theta, phi_range=phi)

raup_model.set_params(estimator="surface")
params_est = raup_model.transform(X)
params_est.shape
```

`set_params` selects the estimation method; `inverse_transform` is not
affected. `transform` works on a panel of surfaces. The result has one row
per surface. A single `(n_theta, n_phi, 3)` array is accepted as a panel of one.

```{code-cell} ipython3
pd.DataFrame(
    {"true": params_true, "estimated": params_est[0]},
    index=raup_model.get_feature_names_out(),
)
```

You should see the five parameters recovered to numerical precision. The
`"surface"` method fits the rigid pose and the scale of the surface together
with the parameters. It does not require the shell to sit in any particular
position or orientation.

To close the loop, generate a shell from the estimate and compare it with the
original:

```{code-cell} ipython3
X_rec = raup_model.inverse_transform(params_est[0], theta_range=theta, phi_range=phi)

rmsd = np.sqrt(np.mean(np.sum((X_rec - X) ** 2, axis=-1)))
print(f"root-mean-square distance: {rmsd:.2e}")
```

```{code-cell} ipython3
X_up = X * np.array([-1.0, 1.0, -1.0])  # apex-up, as in plot_surface3d
X_rec_up = X_rec * np.array([-1.0, 1.0, -1.0])

fig = go.Figure(
    data=[
        go.Surface(
            x=X_up[..., 0], y=X_up[..., 1], z=X_up[..., 2],
            colorscale="Viridis", opacity=0.8, showscale=False,
        ),
        go.Surface(
            x=X_rec_up[..., 0], y=X_rec_up[..., 1], z=X_rec_up[..., 2],
            colorscale="Reds", opacity=0.4, showscale=False,
        ),
    ]
)
fig.update_layout(
    title="Original (Viridis) and reconstruction (Reds)",
    width=700,
    height=500,
    scene=dict(aspectmode="data"),
)
fig.show()
```

The two surfaces coincide. This is the sense in which `transform` and
`inverse_transform` are inverses of each other: a parameter vector, the shell it
generates, and the parameters estimated from that shell describe the same form.

```{note}
The `"surface"` method assumes that rows run along the coil and columns run
once around the tube, as in the output of `inverse_transform`. It uses the
coordinates only: the grid values `theta_range` and `phi_range` are not needed.
```

## Many shells at once

Both directions work on batches. A `(n_samples, 5)` parameter array gives a
`(n_samples, n_theta, n_phi, 3)` panel, and `transform` on that panel gives the
parameters back row by row.

```{code-cell} ipython3
params_batch = np.array(
    [
        [1.2, 0.5, 0.1, 0.0, 0.0],
        [1.5, 1.5, 0.2, 0.2, 0.1],
        [2.0, 0.0, 0.05, -0.1, 0.3],
    ]
)
surfaces = raup_model.inverse_transform(params_batch, theta_range=theta, phi_range=phi)
estimates = raup_model.transform(surfaces)

pd.DataFrame(estimates, columns=raup_model.get_feature_names_out())
```

```{code-cell} ipython3
print(f"largest absolute error: {np.max(np.abs(estimates - params_batch)):.2e}")
```

For analysis pipelines, pass `as_frame=True` to get a tidy long-format
`DataFrame` indexed by `(specimen_id, trajectory_id, phi_id)`. `transform`
accepts this frame as well.

```{code-cell} ipython3
df = raup_model.inverse_transform(
    params_batch, theta_range=theta, phi_range=phi, as_frame=True
)
df.head()
```

## Morphospace

A theoretical morphospace is the parameter space viewed through the shells it
generates. `morphospace_plot` from `ktch.plot` draws a scatter plot in a
two-dimensional space and places generated shapes at a regular grid of
positions. It expects a reducer (such as PCA) between the axes and the
descriptor; here the axes are the parameters `w_r` and `t_r` themselves, so
the reducer is the identity, and the descriptor is `inverse_transform` with
`d_r` fixed.

```{code-cell} ipython3
d_fixed = 0.1


def shells_at(w_t):
    """Generate one shell per ``(w_r, t_r)`` row at ``d_r = d_fixed``, apex-up."""
    w_t = np.asarray(w_t)
    params = np.column_stack([w_t, np.full(len(w_t), d_fixed)])
    X = raup_model.inverse_transform(
        params,
        theta_range=np.linspace(0.0, 2.0 * np.pi * 4.0, 120),
        phi_range=np.linspace(0.0, 2.0 * np.pi, 30),
    )
    return X * np.array([-1.0, 1.0, -1.0])  # apex-up, as in plot_surface3d


fig, ax = plt.subplots(figsize=(7, 6))
ax.set_xlim(1.1, 4.5)
ax.set_ylim(-0.5, 3.5)
morphospace_plot(
    reducer_inverse_transform=lambda scores: scores,  # axes are the parameters
    n_components=2,
    descriptor_inverse_transform=shells_at,
    n_shapes=3,
    shape_scale=0.9,
    ax=ax,
)
ax.scatter(estimates[:, 0], estimates[:, 1], zorder=2)
for i, (w, t) in enumerate(estimates[:, :2]):
    ax.annotate(f"shell {i}", (w, t), xytext=(5, 5), textcoords="offset points")
ax.set_xlabel("w_r (whorl expansion rate)")
ax.set_ylabel("t_r (translation rate)")
ax.set_title("Morphospace over w_r and t_r (d_r=0.1)");
```

The insets show the shells generated at the grid positions; the points are
the shells estimated in the previous section. Generation places a shell at a
point of the morphospace; estimation reads the point back from the shell. Use
the first to design shells and the second to locate specimens.

```{seealso}
- {doc}`growing_tube_model` to do the same with the growing tube model,
  including heteromorph (non-constant) growth.
- {doc}`../../how-to/analysis/sample_evenly_along_shell` to space the samples
  evenly along the shell instead of evenly in `theta`.
- {doc}`../../explanation/coiling` for the theory behind Raup's model.
```

## References

- Noshita, K., 2014. Quantification and geometric analysis of coiling patterns
  in gastropod shells based on 3D and 2D image data. Journal of Theoretical
  Biology 363, 93–104.
- Raup, D.M., Michelson, A., 1965. Theoretical Morphology of the Coiled Shell.
  Science 147, 1294–1295.
- Raup, D.M., 1966. Geometric analysis of shell coiling: general problems.
  Journal of Paleontology 40, 1178–1190.
