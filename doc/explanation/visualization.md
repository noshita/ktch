(visualization)=

# Morphometric Visualization

Morphometric analysis typically ends with dimensionality reduction (PCA, etc.) applied to shape descriptors (EFA coefficients, GPA-aligned coordinates, etc.).
Interpreting the morphospace requires reconstructing and visualizing shapes at various positions in that space.
ktch provides several visualization functions that handle this reconstruction pipeline for any combination of dimensionality reduction method (reducer) and morphometric descriptor.

## The Reconstruction Pipeline

Visualizing shapes in a low-dimensional space requires reversing two transformations:

1. Dimensionality reduction (PCA, KernelPCA, etc.), called `reducer`, maps coefficients to low-dimensional scores. Its inverse maps scores back to coefficients.
2. Morphometric description (EFA, SHA, etc.), called `descriptor`, maps shape coordinates to coefficients. Its inverse maps coefficients back to coordinates.

These two inverse transforms, followed by rendering, form a three-stage pipeline:

```{mermaid}
flowchart LR
    scores[/"scores"/] -- "reducer<br/>inverse_transform" --> coefficients[/"coefficients"/] -- "descriptor<br/>inverse_transform" --> coordinates[/"coordinates"/] -- "render" --> plot(["plot"])
```

For landmark-based methods (GPA), the descriptor stage is an identity operation: the reducer output is already shape coordinates (after reshaping), so no second inverse transform is needed.

This separation is the core abstraction behind `shape_variation_plot` and `morphospace_plot`.
Both functions accept a `reducer` and a `descriptor` (or their inverse transforms directly),
and the user does not need to write the reconstruction loop manually.

## Convenience and Override Parameters

Each pipeline stage follows the same two-level pattern:

- A convenience parameter (`reducer`, `descriptor`) accepts a fitted estimator object and auto-extracts the necessary callable and metadata via duck-typing.
- Override parameters (`reducer_inverse_transform`, `descriptor_inverse_transform`, `explained_variance`, `n_components`) allow direct control when the estimator does not follow the standard interface.

For example, `reducer=pca` extracts `pca.inverse_transform`, `pca.explained_variance_`, and `pca.n_components_` automatically.
For `KernelPCA`, which stores eigenvalues differently and may lack some attributes, the override parameters provide a clean escape hatch:

```python
shape_variation_plot(
    reducer_inverse_transform=kpca.inverse_transform,
    explained_variance=kpca.eigenvalues_,
    n_components=kpca.n_components,
    descriptor=efa,
)
```

This design avoids both PCA-specific naming (the parameter is `reducer`, not `pca`) and over-generalization (the convenience path targets the common case while overrides handle the rest).

## Shape Types

Morphometric data comes in several geometric forms. The `shape_type` parameter controls how reconstructed coordinates are interpreted and rendered:

| `shape_type` | Geometry | Rendering | Typical use |
|---|---|---|---|
| `curve_2d` | Parametric curve, 2D display | `ax.plot(x, y)` | EFA 2D outlines |
| `curve_3d` | Parametric curve, 3D display | `ax.plot(x, y, z)` | EFA 3D outlines |
| `surface_3d` | Structured grid, 3D display | `ax.plot_surface` | SHA surfaces |
| `landmarks_2d` | Discrete points, 2D display | `ax.scatter` + links | GPA 2D landmarks |
| `landmarks_3d` | Discrete points, 3D display | `ax.scatter` + links | GPA 3D landmarks |

The naming follows a `{geometry}_{display_dimension}` convention.
`curve` replaces the more specific term "outline" to accommodate arbitrary parametric curves.
`surface` covers any structured-grid surface (spherical harmonics, torus, annulus, disk harmonics).

### Auto-detection

When `shape_type="auto"` (the default), the type is inferred from the output of the descriptor inverse transform:

- 4D array -> `surface_3d`
- 3D array with last dimension 2 -> `curve_2d`
- 3D array with last dimension 3 -> `curve_3d`
- No descriptor (identity/GPA case) with `n_dim=2` -> `landmarks_2d`
- No descriptor (identity/GPA case) with `n_dim=3` -> `landmarks_3d`

Users can always override auto-detection by specifying `shape_type` explicitly.
This is necessary when the auto-detection rule does not match the intended display, for example when projecting 3D landmarks onto 2D.

### Spatial Dimensionality vs Display Mode

`n_dim` and `shape_type` are independent.
`n_dim` specifies the spatial dimensionality of the data (used for reshaping in the GPA identity case).
`shape_type` specifies the display mode (2D or 3D axes).
Combining them enables projection:

| `n_dim` | `shape_type`   | Result                             |
| ------- | -------------- | ---------------------------------- |
| 3       | `landmarks_3d` | Full 3D rendering                  |
| 3       | `landmarks_2d` | XY projection (first 2 dimensions) |
| 2       | `landmarks_2d` | Standard 2D rendering              |

For projections other than XY, a custom `render_fn` can select arbitrary dimension pairs.

## Axes-level and Figure-level Functions

Following seaborn's convention, ktch distinguishes two API levels for plot functions:

- Axes-level functions accept an `ax` parameter and return `matplotlib.axes.Axes`. They operate on a single axes and are composable with user-managed subplot layouts, seaborn's `FacetGrid`, and other axes-level functions. `morphospace_plot` is axes-level.

- Figure-level functions accept a `fig` parameter and return `matplotlib.figure.Figure`. They manage their own multi-axes layout. `shape_variation_plot` is figure-level because it creates a grid of subplots (components x SD values).

This distinction determines composability.
An axes-level function can be dropped into any existing subplot:

```python
fig, axes = plt.subplots(2, 2)
for ax, (i, j) in zip(axes.flat[:3], [(0, 1), (1, 2), (2, 0)]):
    morphospace_plot(
        data=df, x=f"PC{i+1}", y=f"PC{j+1}", hue="genus",
        reducer=pca, descriptor=efa, components=(i, j), ax=ax,
    )
explained_variance_ratio_plot(pca, ax=axes[1, 1])
```

A figure-level function controls the entire figure and is called standalone.

## Custom Rendering

Built-in renderers cover common cases, but the `render_fn` parameter accepts any callable with the signature:

```python
def render_fn(coords, ax, *, color="gray", alpha=1.0, links=None, **kwargs):
    ...
```

This enables filled shapes, custom projections, annotation overlays, or any other matplotlib drawing without modifying the reconstruction pipeline.
Additional keyword arguments can be passed via `**render_kw` and are forwarded to the renderer.

## Naming Conventions

Plot functions in ktch use a `<noun>_plot` suffix: `morphospace_plot`, `shape_variation_plot`, `explained_variance_ratio_plot`, `tps_grid_2d_plot`.
This is consistent with seaborn's `<noun>plot` style and avoids the `plot_<noun>` prefix, which can create misleading autocompletions when users type `plot.` expecting the module's namespace.

Component indices are 0-indexed throughout, consistent with sklearn's `components_[i]` attribute and Python convention.

```{seealso}
- {doc}`harmonic` for the theory behind EFA and spherical harmonics
- {doc}`landmark` for Procrustes methods and shape coordinates
- {doc}`../how-to/visualization/morphospace` for practical morphospace plotting
- {doc}`../tutorials/harmonic/elliptic_Fourier_analysis` for a complete EFA workflow including visualization
```
