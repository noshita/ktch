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

## Spherical Harmonic Analysis

For 3D closed surfaces, spherical harmonics provide an analogous decomposition. A closed surface can be represented using spherical harmonic basis functions $Y_l^m$, where $l$ is the degree (analogous to harmonic number) and $m$ is the order.

### Usage in ktch

ktch provides `SphericalHarmonicAnalysis` for working with spherical harmonic coefficients. Note that coefficient estimation requires pre-estimated surface parameterization; direct estimation from surface coordinates alone is not currently supported.

```python
from ktch.harmonic import SphericalHarmonicAnalysis

sha = SphericalHarmonicAnalysis(n_harmonics=15)
coefficients = sha.fit_transform(parameterized_surfaces)
```

### Applications

Spherical harmonic analysis is used for:

- Fruit shape analysis
- Grain morphology
- Organ shape quantification

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

This enables visualization of shape variation along PC axes or between groups.

## Limitations

- Global description may miss local features
- Sensitive to outline/surface quality and sampling
- Normalization choices affect results

```{seealso}
- {doc}`morphometrics` for comparison with landmark methods
- {doc}`../tutorials/harmonic/elliptic_Fourier_analysis` for practical examples
```

## References

- Kuhl, F. P., & Giardina, C. R. (1982). Elliptic Fourier features of a closed contour. Computer Graphics and Image Processing, 18(3), 236-258.
- Crampton, J. S. (1995). Elliptic Fourier shape analysis of fossil bivalves. Lethaia, 28(2), 147-158.
- Shen, L., & Makedon, F. (2006). Spherical mapping for processing of 3D closed surfaces. Image and Vision Computing, 24(7), 743-761.
