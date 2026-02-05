(semilandmarks)=

# Semilandmarks

Semilandmarks are points placed along curves or surfaces where discrete anatomical landmarks cannot be defined.
Unlike fixed landmarks, semilandmarks are allowed to slide along the structure during Procrustes superimposition to minimize bending energy or Procrustes distance.

## Why Semilandmarks?

Landmark-based morphometrics requires discrete, biologically/geometrically homologous points.
However, many biological structures have smooth curves or surfaces without clear anatomical features:

- Leaf edges
- Shell apertures
- Cranial vaults
- Tooth crowns

Semilandmarks extend landmark methods to capture shape information from these structures.

## Types of Semilandmarks

### Curve Semilandmarks

Points placed along a 2D or 3D curve.
Each point is constrained to slide along the tangent direction of the curve.

### Surface Semilandmarks

Points placed on a 3D surface.
Each point is constrained to slide within the tangent plane of the surface at that point.

```{note}
Surface semilandmarks are not yet implemented in ktch.
The API accepts a `surfaces` parameter but will raise `NotImplementedError`.
```

## The Sliding Algorithm

During Generalized Procrustes Analysis, semilandmarks are iteratively repositioned to minimize a criterion:

1. Bending energy minimization: Minimizes the thin-plate spline bending energy between each specimen and the reference (Bookstein, 1997).
2. Procrustes distance minimization: Minimizes the Procrustes distance to the reference.

### GPA Iteration with Semilandmarks

The full GPA loop with semilandmark sliding proceeds as follows (Gunz et al., 2005):

1. Center and scale all configurations
2. Compute the mean shape as a reference
3. Align each configuration to the reference via Procrustes superimposition
4. Slide semilandmarks along their tangent directions
5. Re-project the slid semilandmarks back onto the original curve
6. Re-center and re-scale
7. Update the mean shape
8. Repeat steps 3--7 until convergence

Steps 4--5 are the key additions for semilandmark analysis.

## Re-projection onto curves

After sliding, each semilandmark may drift off its original curve because the tangent direction is only a local linear approximation.
To maintain geometric consistency, slid points are re-projected onto their original curve (Gunz et al., 2005).

Gunz et al. (2005) describe this step as: "Replace each slid semilandmark by its nearest point on the (curving) surface."
In ktch, the original curve is approximated by piecewise-linear segments connecting the original (unslid) landmark positions, and this approximation is stored at the start of GPA and maintained throughout the iterative process.

### Original curve geometry

At the beginning of GPA, the original curve geometry is stored for each specimen.
During GPA iterations, this stored geometry undergoes the same similarity transformations (centering, scaling, rotation) as the sliding configurations, but is never modified by the sliding step itself. This ensures a stable projection target throughout the iterative process.

## Combining Landmarks and Semilandmarks

A typical configuration includes:

- Fixed landmarks: Anatomically defined points that do not slide
- Curve semilandmarks: Points along outline curves
- Surface semilandmarks: Points on 3D surfaces (if applicable)

All points are analyzed together in GPA, but only semilandmarks are allowed to slide.
The topology of each curve is specified as a matrix of `[before, slider, after]` index triplets,
where the first and last points of an open curve serve as fixed anchors.

## TPS Bending Energy

The thin-plate spline (TPS) bending energy measures the amount of non-affine deformation between two configurations.
Minimizing this quantity during sliding produces semilandmark positions that require the least local deformation to match the reference.

### TPS Kernel

The TPS kernel function depends on the dimensionality:

2D (biharmonic):

$$
U(r) = r^2 \log r
$$

3D (triharmonic):

$$
U(r) = -|r|
$$

where $r$ is the Euclidean distance between two points. ktch automatically selects the appropriate kernel based on the `n_dim` parameter.

### Closed-form Sliding Solution

For bending energy minimization, the optimal sliding displacements are computed using the closed-form weighted least squares solution (Bookstein 1997):

$$
T = -(U^\top L_k^{-1} U)^{-1} U^\top L_k^{-1} Y^0
$$

where $L_k^{-1}$ is the bending energy matrix (upper-left $k \times k$ block of the TPS system matrix inverse), $U$ encodes the tangent sliding directions, and $Y^0$ are the initial landmark positions.

## Considerations

### Advantages

- Captures shape information from smooth structures
- Compatible with standard GPA workflow
- Enables analysis of structures lacking discrete landmarks
- Works in both 2D and 3D

### Limitations

- Placement is somewhat arbitrary (no biological homology)
- Results can depend on sliding criterion choice
- Requires careful initialization of point positions
- Surface semilandmarks require mesh or surface parameterization (not yet supported)

## References

- Bookstein, F.L., 1997. Landmark methods for forms without landmarks: morphometrics of group differences in outline shape. Medical image analysis 1, 225--243. [https://doi.org/10.1016/s1361-8415(97)85012-8](https://doi.org/10.1016/s1361-8415(97)85012-8)
- Gunz, P., Mitteroecker, P., Bookstein, F.L., 2005. Semilandmarks in three dimensions. In: Slice, D.E. (Ed.), Modern Morphometrics in Physical Anthropology. Kluwer Academic/Plenum Publishers, New York, pp. 73--98. [https://doi.org/10.1007/0-387-27614-9_3](https://doi.org/10.1007/0-387-27614-9_3)
- Gunz, P., Mitteroecker, P., 2013. Semilandmarks: a method for quantifying curves and surfaces. Hystrix, the Italian Journal of Mammalogy 24, 103--109. [https://doi.org/10.4404/hystrix-24.1-6292](https://doi.org/10.4404/hystrix-24.1-6292)

```{seealso}
- {doc}`landmark` for standard landmark-based methods
- {doc}`../tutorials/landmark/generalized_Procrustes_analysis` for GPA tutorial
- {doc}`../tutorials/landmark/semilandmarks_gpa` for semilandmark analysis tutorial
```
