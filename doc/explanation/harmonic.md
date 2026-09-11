(harmonic)=

# Harmonic-based Morphometrics

Harmonic-based morphometrics describes shape using mathematical functions that decompose outlines or surfaces into frequency components. Unlike landmark methods, harmonic approaches do not require explicit point-to-point correspondence between specimens, though specimens should represent homologous structures.

## Elliptic Fourier Analysis (EFA)

Elliptic Fourier Analysis represents closed 2D outlines as a linear combination of sine and cosine functions at various frequencies.

### Mathematical Foundation

A closed 2D outline can be parameterized by arc length $t$ as:

$$
x(t) = A_0 + \sum_{n=1}^{N} \left( a_n \cos\frac{2\pi nt}{T} + b_n \sin\frac{2\pi nt}{T} \right)
$$

$$
y(t) = C_0 + \sum_{n=1}^{N} \left( c_n \cos\frac{2\pi nt}{T} + d_n \sin\frac{2\pi nt}{T} \right)
$$

where:

- $T$ is the total perimeter
- $N$ is the number of harmonics
- $a_n, b_n, c_n, d_n$ are Fourier coefficients

Each harmonic $n$ contributes four coefficients that together define an ellipse.
The first harmonic ($n=1$) captures the overall elliptical shape, while higher harmonics capture finer details.

### Normalization

Raw EFA coefficients contain information about size, rotation, and starting point. For shape analysis, coefficients are typically normalized to remove these effects:

- Size normalization: Scale so the first harmonic has unit semi-major axis
- Rotation normalization: Rotate so the first harmonic is aligned with a reference axis
- Starting point normalization: Adjust phase to a standard starting point

In ktch, normalization is handled automatically:

```python
from ktch.harmonic import EllipticFourierAnalysis

efa = EllipticFourierAnalysis(n_harmonics=20, norm=True)
coefficients = efa.fit_transform(outlines)
```

### Choosing the Number of Harmonics

The number of harmonics determines the level of detail captured:

| Harmonics | Detail Level | Use Case |
|-----------|--------------|----------|
| 1-5 | Coarse | Overall shape, major features |
| 10-20 | Moderate | Most biological applications |
| 30+ | Fine | Complex outlines, high resolution |

#### Practical guidelines

- Start with 20 harmonics for most biological shapes
- Examine reconstructed outlines to assess fit
- More harmonics = more coefficients = higher dimensionality
- Diminishing returns for very high harmonics

### Cumulative Fourier Power

The cumulative Fourier power indicates how much shape variance is captured by the first $n$ harmonics. When the cumulative power exceeds 0.99, the harmonics capture 99% of the shape information.

### EFA for 3D Curves

ktch extends EFA to 3D closed curves by adding a third coordinate function $z(t)$. This yields six coefficients per harmonic instead of four.

```python
# 3D outline analysis
efa_3d = EllipticFourierAnalysis(n_harmonics=20, n_dim=3)
coefficients_3d = efa_3d.fit_transform(outlines_3d)
```

3D normalization follows the algorithm described in Godefroy et al. (2012), removing size, orientation (via Euler ZXZ decomposition), and starting point variation from the first-harmonic ellipse.

### General Codomain Dimension

The same expansion applies to data of any codomain dimension $D$ with `norm=False`. Here `t` plays the role of the parameterization, analogous to `r_theta` in DHA and `theta_phi` in SPHARM. For $D \in \{2, 3\}$ (spatial shape coordinates) `t` defaults to automatic arc-length parameterization. For $D=1$ or $D>3$ the codomain holds non-shape data, `t` is required and should be derived from the corresponding shape, onto which the data is then mapped.

## Spherical Harmonic Analysis

For 3D closed surfaces, spherical harmonics provide an analogous decomposition. A closed surface can be represented using spherical harmonic basis functions $Y_l^m$, where $l$ is the degree (analogous to harmonic number) and $m$ is the order.

### Usage in ktch

ktch provides `SphericalHarmonicAnalysis` for working with spherical harmonic coefficients. Note that coefficient estimation requires pre-estimated surface parameterization; direct estimation from surface coordinates alone is not currently supported.

```python
from ktch.harmonic import SphericalHarmonicAnalysis

sha = SphericalHarmonicAnalysis(n_harmonics=15)
coefficients = sha.fit_transform(parameterized_surfaces)
```

The codomain dimension is set by `n_dim` (default 3): use `n_dim=1` for a scalar field on the sphere, or any $D$ for an $\mathbb{R}^D$-valued field.

### Registration

SPHARM coefficients include position, orientation, and size, which will be removed for shape comparison. `SphericalHarmonicAnalysis` removes them through its `registration` parameter (e.g., `"first_order"` uses the degree-1 ellipsoid to canonicalize orientation, the parameter sphere, and scale). The same registration is available as a standalone transformer, `SphericalHarmonicRegistration`, for coefficients computed elsewhere (for example, SPHARM-PDM `.coef` files read with `read_spharmpdm_coef`):

```python
from ktch.harmonic import SphericalHarmonicRegistration

reg = SphericalHarmonicRegistration(method="first_order", scale=False)
registered = reg.fit_transform(coefficients)
```

It maps coefficients to coefficients and composes in a scikit-learn `Pipeline` before PCA on a morphospace.

#### What degree 1 leaves undetermined

The degree-1 ellipsoid fixes the principal axes only as lines: a half turn about any of its axes maps it onto itself. What remains is a sign on each axis of the parameter sphere, eight patterns that describe the same surface with different correspondences between parameter values and surface points. `first_order` resolves it with the third moment of the surface, which is invariant to rotation and reparameterization but ill-conditioned for near-symmetric shapes. SPHARM-PDM has no intrinsic rule: the signs come from its eigensolver, a right-handedness constraint, and optionally a template or an explicit flip index. The two agree on clearly triaxial shapes and may settle on different frames of the same shape near symmetry; per-degree amplitude spectra agree either way. SPHARM-PDM's flip indices are the eight patterns (x is the longest axis in both programs):

| flip index | signs on (x, y, z) | proper rotation |
|---|---|---|
| 0 | (+, +, +) | yes |
| 1 | (+, -, -) | yes |
| 2 | (-, +, -) | yes |
| 3 | (-, -, +) | yes |
| 4 | (+, -, +) | no |
| 5 | (+, +, -) | no |
| 6 | (-, -, -) | no |
| 7 | (-, +, +) | no |

An odd number of minus signs reverses the handedness of the parameter sphere. That is a reparameterization, not a mirror image of the shape, and unrelated to the `reflect` parameter.

To keep a frame the input already carries, register with `align_parameter=False`: the coordinates are rotated onto the parameter axes and the parameter sphere is left as it is. This presupposes an upstream parameter alignment, which is detected from the orthogonality of the degree-1 columns. SPHARM-PDM's frame differs from ktch's by a half turn about z for every specimen, a sign convention of its basis functions; a rotation shared by all specimens leaves a shape analysis unchanged. See {doc}`../how-to/data/load_spharm`.

#### Correcting a specimen after registration

A specimen that registration placed on the wrong sign pattern looks rotated by a half turn about a principal axis, which is indistinguishable by eye from a specimen that is merely posed differently. The two need different corrections. A representative mismatch is a coupled discrepancy, in the coordinates and on the parameter sphere together; rotating the coordinates alone aligns the pose and leaves the correspondence broken, which no rendering shows and every coefficient-space comparison detects. `rotate_spharm_coeffs` therefore takes the domain explicitly. Use `domain="coupled"` to correct one specimen. Use `domain="codomain"` or `domain="parameter"` for a convention applied to every specimen alike, such as a display up-axis or a pole placed at an anatomical landmark; applied uniformly, each leaves coefficient-space distances unchanged. Per specimen, a single domain is sound only when the other half is supplied elsewhere: a field defined on the same parameter sphere receives the geometry's correction with `domain="parameter"`, and a codomain-only rotation of one specimen is safe only when its parameterization already carries the correspondence. Do not register again after a correction; registration recomputes the frame and discards it.

### Applications

Spherical harmonic analysis is used for:

- Fruit shape analysis
- Grain morphology
- Organ shape quantification

## Disk Harmonic Analysis

For surfaces or scalar fields parameterized on a unit disk, disk harmonics provide a decomposition using Fourier–Bessel basis functions. `DiskHarmonicAnalysis` expands an $\mathbb{R}^D$-valued function on the disk, with `n_dim` setting the codomain dimension (`n_dim=1` for a scalar field, `2`/`3` for planar/surface mappings). It requires a pre-estimated disk parameterization $(r, \theta)$.

```python
from ktch.harmonic import DiskHarmonicAnalysis

dha = DiskHarmonicAnalysis(n_harmonics=10, n_dim=3)
coefficients = dha.fit_transform(surfaces, r_theta=r_theta)
```

## Statistical Analysis

After obtaining harmonic coefficients, standard multivariate methods apply:

### Principal Component Analysis

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pc_scores = pca.fit_transform(coefficients)
```

### Shape Reconstruction

Coefficients can be transformed back to outlines/surfaces for visualization:

```python
# Reconstruct outline from coefficients
reconstructed = efa.inverse_transform(coefficients)

# Reconstruct from modified PC scores
modified_coef = pca.inverse_transform(modified_scores)
reconstructed_shape = efa.inverse_transform(modified_coef)
```

ktch provides built-in plot functions for these visualizations; see {doc}`visualization` for details.

## Limitations

- Global description may miss local features
- Sensitive to outline/surface quality and sampling
- Normalization choices affect results

```{seealso}
- {doc}`morphometrics` for comparison with landmark methods
- {doc}`visualization` for the visualization pipeline design
- {doc}`../tutorials/harmonic/elliptic_Fourier_analysis` for practical examples
```

## References

- Kuhl, F. P., & Giardina, C. R. (1982). Elliptic Fourier features of a closed contour. Computer Graphics and Image Processing, 18(3), 236-258.
- Crampton, J. S. (1995). Elliptic Fourier shape analysis of fossil bivalves. Lethaia, 28(2), 147-158.
- Godefroy, J. E., Bornert, F., Gros, C. I., & Constantinesco, A. (2012). Elliptical Fourier descriptors for contours in three dimensions: A new tool for morphometrical analysis in biology. Comptes Rendus Biologies, 335(3), 205-213. https://doi.org/10.1016/j.crvi.2011.12.004
- Shen, L., & Makedon, F. (2006). Spherical mapping for processing of 3D closed surfaces. Image and Vision Computing, 24(7), 743-761.
- Verrall, S. C., & Kakarala, R. (1998). Disk-harmonic coefficients for invariant pattern recognition. Journal of the Optical Society of America A, 15(2), 389.
- Shaqfa, M., Choi, G. P. T., Anciaux, G., & Beyer, K. (2025). Disk harmonics for analysing curved and flat self-affine rough surfaces and the topological reconstruction of open surfaces. Journal of Computational Physics, 522, 113578.
