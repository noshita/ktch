(landmark)=

# Landmark-based Morphometrics

Landmark-based morphometrics analyzes shape using discrete, biologically meaningful points called landmarks. This approach is implemented in ktch through Generalized Procrustes Analysis (GPA).

## Landmarks

A landmark is a point of correspondence on each object that matches between and within populations.

## Configuration and Centroid Size

A configuration is the complete set of landmarks for a single specimen, represented as a matrix of coordinates.

### Centroid Size

Centroid size is the standard measure of size in geometric morphometrics, defined as the square root of the sum of squared distances from each landmark to the centroid.

In ktch:

```python
from ktch.landmark import centroid_size

cs = centroid_size(configurations)
```

## Generalized Procrustes Analysis (GPA)

GPA is the standard method for extracting shape information from landmark configurations. It removes variation due to:

1. Translation (position) - by centering configurations
2. Scale (size) - by normalizing to unit centroid size
3. Rotation (orientation) - by optimal rotation alignment

### GPA Algorithm

For a sample of configurations, GPA iteratively:

1. Center each configuration at the origin
2. Scale each configuration to unit centroid size
3. Rotate configurations to minimize distances to a reference
4. Compute the mean shape
5. Repeat until convergence

The result is a set of Procrustes shape coordinates.

In ktch:

```python
from ktch.landmark import GeneralizedProcrustesAnalysis

gpa = GeneralizedProcrustesAnalysis()
shapes = gpa.fit_transform(configurations)
```

## Pre-shape Space and Shape Space

After centering and scaling, configurations lie on a pre-shape space—a high-dimensional hypersphere.

After GPA removes rotation, specimens occupy Kendall's shape space. The Procrustes distance between shapes corresponds to the great-circle distance on this space.

### Tangent Space Approximation

For practical analysis, data are projected onto a tangent space—a linear approximation at the mean shape. This enables standard multivariate statistics (PCA, regression, etc.).

## Statistical Analysis of Shape

### Principal Component Analysis

```python
from sklearn.decomposition import PCA

pca = PCA()
pc_scores = pca.fit_transform(shapes)
```

## Limitations

- Requires homologous landmarks across all specimens
- Not suitable for structures lacking clear landmarks

```{seealso}
- {doc}`morphometrics` for comparison with harmonic methods
- {doc}`../tutorials/landmark/generalized_Procrustes_analysis` for practical examples
```

## References

- Noshita, K. (2022). Model-based phenotyping for plant morphometrics. Breeding Science, 72(1), 3-13.
