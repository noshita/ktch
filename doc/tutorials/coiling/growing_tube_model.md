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

# Growing tube model

In this tutorial, you will generate shells with the growing tube model
(Okamoto, 1988).

The growing tube model describes a coiling pattern using a differential-geometric framework. 
The tube radius and the trajectory's local geometry are set by three parameters 
at each growth stage `s`:

- expansion rate `e_g`: how fast the tube radius grows
- standardized curvature `c_g`: how tightly the trajectory bends (`c_g = 0` is
  a straight tube)
- standardized torsion `t_g`: how much the trajectory twists out of a plane
  (`t_g = 0` is a planispiral)

## Setup

```{code-cell} ipython3
# Uncomment if needed
# %pip install "ktch[plot]"
```

```{code-cell} ipython3
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ktch.coiling import GrowingTubeModel, growing_tube, l_g, s_g
```

```{code-cell} ipython3
:tags: [remove-cell]

# This cell is only required for the Sphinx documentation build.
# You do not need this setting when running in Jupyter.
import plotly.io as pio

pio.renderers.default = "sphinx_gallery"
```

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show definition: plot_surface3d()"
:  code_prompt_hide: "Hide definition: plot_surface3d()"

def plot_surface3d(X, *, title="", colorscale="Viridis"):
    """Plot a ``(n, n_phi, 3)`` surface array with plotly."""
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

## Generate a shell with the growing tube model

`growing_tube` takes the three parameters and returns a sampled surface. 

```{code-cell} ipython3
s = np.linspace(0.0, 60.0, 200)
phi = np.linspace(0.0, 2.0 * np.pi, 60)

X = growing_tube(e_g=0.02, c_g=0.4, t_g=0.06, s_range=s, phi_range=phi)
X.shape
```

The result is a `(n_s, n_phi, 3)` array: `n_s` samples along the growth stage
`s`, `n_phi` samples around the tube, and the `(x, y, z)` coordinates.

```{code-cell} ipython3
fig = plot_surface3d(X, title="Growing tube (e_g=0.02, c_g=0.4, t_g=0.06)")
fig.show()
```

## Explore the parameters

We vary one parameter at a time.

```{code-cell} ipython3
:tags: [hide-input]
:mystnb:
:  code_prompt_show: "Show definition: compare_growing_tube()"
:  code_prompt_hide: "Hide definition: compare_growing_tube()"

def compare_growing_tube(values, *, vary, title, **fixed):
    """Generate growing tubes varying one parameter and show them side by side."""
    phi = np.linspace(0.0, 2.0 * np.pi, 40)
    fig = make_subplots(
        rows=1,
        cols=len(values),
        specs=[[{"type": "surface"}] * len(values)],
        subplot_titles=[f"{vary}={v}" for v in values],
        horizontal_spacing=0.01,
    )
    for i, v in enumerate(values, start=1):
        params = {**fixed, vary: v}
        if params["c_g"] > 0:  # coiled: span four whorls
            d_g = np.hypot(params["c_g"], params["t_g"])
            s = np.linspace(0.0, 2.0 * np.pi * 4.0 / d_g, 240)
        else:  # straight tube (c_g = 0): no whorls to count, use a fixed span
            s = np.linspace(0.0, 60.0, 240)
        Xi = growing_tube(
            params["e_g"], params["c_g"], params["t_g"], s_range=s, phi_range=phi
        )
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

### Expansion rate `e_g`

With `e_g = 0` the tube keeps a constant radius; larger values open the shell up quickly.

```{code-cell} ipython3
compare_growing_tube([0.0, 0.04, 0.2], vary="e_g", title="Expansion rate",
                     c_g=0.4, t_g=0.06).show()
```

### Standardized curvature `c_g`

Larger values give a tighter coil. 
(At `c_g = 0` the trajectory becomes straight.)

```{code-cell} ipython3
compare_growing_tube([0.0, 0.4, 0.8], vary="c_g", title="Standardized curvature",
                     e_g=0.04, t_g=0.06).show()
```

### Standardized torsion `t_g`

With `t_g = 0` the coil stays planar (a planispiral); increasing it lifts it into a helix.

```{code-cell} ipython3
compare_growing_tube([0.0, 0.06, 0.2], vary="t_g", title="Standardized torsion",
                     e_g=0.04, c_g=0.4).show()
```

## Sample evenly along the shell

By default the surface is sampled at equal steps in the growth stage `s`.
Because the tube radius grows exponentially with `s`, equal `s` steps trace
little arc length near the tight apex and much more near the wide aperture.
For even meshes, sample at equal steps of arc length along the shell instead.

`l_g` and `s_g` convert between the growth stage and the trajectory arc length
`l`. The arc length depends only on the expansion rate `e_g` (and `r0`), not on
the curvature `c_g` or torsion `t_g`. To space `n_s` samples evenly in arc
length, take a uniform grid in `l` and map it back to `s` with `s_g`:

```{code-cell} ipython3
e_g, c_g, t_g = 0.04, 0.4, 0.06
d_g = np.hypot(c_g, t_g)
n_whorls = 4.0
n_s = 60
phi = np.linspace(0.0, 2.0 * np.pi, 40)

s_uniform = np.linspace(0.0, 2.0 * np.pi * n_whorls / d_g, n_s)

l_max = l_g(s_uniform[-1], e_g)
s_arclength = s_g(np.linspace(0.0, l_max, n_s), e_g)
```

```{code-cell} ipython3
fig = make_subplots(
    rows=1,
    cols=2,
    specs=[[{"type": "surface"}, {"type": "surface"}]],
    subplot_titles=["equal steps in s", "equal steps in arc length"],
    horizontal_spacing=0.01,
)
for col, s in enumerate([s_uniform, s_arclength], start=1):
    Xi = growing_tube(e_g, c_g, t_g, s_range=s, phi_range=phi)
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

## Heteromorph

So far the parameters were constant, giving regular spirals. The growing tube
model naturally allows its parameters (`e_g`, `c_g`, and `t_g`) to change during growth. 
Pass each as a function of the growth stage `s` (or an array aligned to
`s_range`). This produces heteromorph (irregularly coiled) shells.

The example below mimics the heteromorph ammonite *Nipponites* (Okamoto, 1988):
a constant expansion rate with curvature and torsion that oscillate along growth.

```{code-cell} ipython3
e_g = np.log(1.028)

def c_g(s):
    return 0.2 * np.sin(2.0 * np.pi * s / 7.0 - np.pi / 2.0) + 0.5

def t_g(s):
    return 0.55 * np.cos(2.0 * np.pi * s / 14.0)
```

Let's look at the two varying parameters as functions of `s`:

```{code-cell} ipython3
s = np.linspace(0.0, 45.0, 451)

fig = go.Figure()
fig.add_scatter(x=s, y=c_g(s), mode="lines", name="c_g(s)")
fig.add_scatter(x=s, y=t_g(s), mode="lines", name="t_g(s)")
fig.update_layout(width=700, height=300, xaxis_title="s", yaxis_title="value")
fig.show()
```

Non-constant parameters require `method="ode"` (the closed form only handles
constant parameters) and an explicit `s_range`.

```{code-cell} ipython3
X = growing_tube(e_g, c_g, t_g, s_range=s, method="ode",
                 phi_range=np.linspace(0.0, 2.0 * np.pi, 40))
plot_surface3d(X, title="Nipponites-like heteromorph", colorscale="Purp").show()
```

Try changing the oscillation periods (`7` and `14`) or amplitudes to explore other irregular forms.

## Generate shells in batch

`GrowingTubeModel` provides a scikit-learn-style estimator whose 
`inverse_transform` generates forms from parameters. 
The `method` argument selects the solver (`"ode"` or the constant-parameter
`"closed"` form).

```{code-cell} ipython3
model = GrowingTubeModel()
params = np.array(
    [
        [0.05, 0.24, 0.05],
        [0.03, 0.20, 0.00],
        [0.10, 0.40, 0.15],
    ]
)
surfaces = model.inverse_transform(
    params, s_range=np.linspace(0.0, 50.0, 180), phi_range=phi
)
surfaces.shape
```

```{code-cell} ipython3
plot_surface3d(surfaces[2], title="Third parameter set").show()
```

```{seealso}
- {doc}`raup_model` for Raup's model.
- {doc}`../../explanation/coiling` for the theory behind the growing tube model,
  including the differential geometry of heteromorph growth.
```

## References

- Okamoto, T., 1988. Analysis of heteromorph ammonoids by differential
  geometry. Palaeontology 31, 35–52.
