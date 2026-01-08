(morphometrics)=

# What is Morphometrics?

Morphometrics provides quantitative representation of morphological traits by extracting shape information as geometric invariants. It offers mathematical tools for modeling morphological properties, bridging the gap between raw digitized data and biologically meaningful phenotypic values.

## Form and Shape

In geometric morphometrics, form and shape have precise mathematical definitions:

- Form: Geometric properties invariant to translation and rotation
- Shape: Geometric properties invariant to translation, rotation, and scaling

This distinction is fundamental in geometric morphometrics: shape analysis removes size information, allowing comparison of organisms regardless of their absolute dimensions.

## Approaches in Morphometrics

ktch implements two major approaches to morphometric analysis:

### Landmark-based Morphometrics

Landmark-based methods model morphological properties as sets of corresponding points among specimens. Shape is extracted via Generalized Procrustes Analysis (GPA), which removes position, size, and orientation.

Key characteristics:

- Requires identification of corresponding points across specimens
- Suitable for structures with clearly identifiable anatomical features
- Captures local shape information at specific points

Applications:

- Grass phytoliths
- Leaf shape analysis
- Flower morphology
- Skeletal morphology

```{seealso}
{doc}`landmark` for details on Procrustes methods
```

### Harmonic-based Morphometrics

Harmonic-based methods describe shape using mathematical functions that capture the outline or surface geometry without requiring point-to-point correspondence between specimens.

Elliptic Fourier Analysis (EFA):
Models closed outlines by approximating x and y coordinates as Fourier series, quantifying shapes through normalized Fourier coefficients.

Spherical Harmonic Analysis:
Models closed 3D surfaces using spherical harmonic functions.

Applications:

- Seed morphology
- Leaf outlines
- Petal shapes
- Fruit shapes

```{seealso}
{doc}`harmonic` for details on harmonic methods
```

## Choosing an Approach

| Consideration | Landmark-based | Harmonic-based |
|---------------|----------------|----------------|
| Data type | Discrete corresponding points | Continuous outlines/surfaces |
| Homology | Requires explicit point correspondence | No explicit point correspondence required |
| Automation | Partially manual | Highly automatic |

Use landmark-based methods when:

- Clear, identifiable anatomical points exist
- Biological homology between points is established

Use harmonic-based methods when:

- No clear landmarks are available
- The outline or surface itself is the feature of interest

## The scikit-learn Compatible Workflow

ktch follows the scikit-learn API design:

```python
from ktch.landmark import GeneralizedProcrustesAnalysis
from ktch.harmonic import EllipticFourierAnalysis
from sklearn.decomposition import PCA

# Landmark workflow
gpa = GeneralizedProcrustesAnalysis()
shapes = gpa.fit_transform(landmarks)

# Harmonic workflow
efa = EllipticFourierAnalysis(n_harmonics=20)
coefficients = efa.fit_transform(outlines)

# Combine with PCA
pca = PCA(n_components=3)
scores = pca.fit_transform(shapes)  # or coefficients
```

```{seealso}
{doc}`../how-to/analysis/use_with_pipeline` for practical examples
```

## References

- Noshita, K. (2022). Model-based phenotyping for plant morphometrics. Breeding Science, 72(1), 3-13.
