(landmark)=

# Landmark-based Morphometrics

Landmark-based morphometrics analyzes shape using discrete, biologically meaningful points called landmarks. This approach is implemented in ktch through Generalized Procrustes Analysis (GPA).

## Landmarks

A landmark is a point of correspondence on each object that matches between and within populations.
Bookstein (1997) classifies landmarks into three types based on the evidence supporting their homology:

- Type I: discrete juxtapositions of tissues or structures
  (e.g., intersection of two sutures on a skull)
- Type II: local geometric features such as maxima of curvature
  (e.g., the tip of a leaf lobe)
- Type III: extremal points defined relative to other landmarks or
  an external axis (e.g., the point farthest from the centroid)

When curves or surfaces lack sufficient discrete landmarks,
semilandmarks can be placed along the outline and allowed to slide
during Procrustes superimposition. See {doc}`semilandmarks` for details.

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

The result is a set of shape coordinates.

In ktch:

```python
from ktch.landmark import GeneralizedProcrustesAnalysis

gpa = GeneralizedProcrustesAnalysis()
shapes = gpa.fit_transform(configurations)
```

## Pre-shape Space and Shape Space

After centering and scaling, configurations lie on a pre-shape space, a high-dimensional hypersphere.

After GPA removes orientation information, the specimens occupy Kendall's shape space. The Procrustes distance between shapes corresponds to the great-circle distance on this space.

### Tangent Space Approximation

For practical analysis, data are projected onto a tangent space, which is a linear approximation at the mean shape. This enables standard multivariate statistics (PCA, regression, etc.).

## Statistical Analysis of Shape

### Principal Component Analysis

```python
from sklearn.decomposition import PCA

pca = PCA()
pc_scores = pca.fit_transform(shapes)
```

## Limitations

- Requires homologous landmarks across all specimens
- Not suitable for structures lacking clear landmarks (see {doc}`semilandmarks` for extending GPA to curves and surfaces)

## Digitization

Digitization (landmarking) is the process of recording landmark
coordinates from images or 3D scans.

### Manual digitization

For small to moderate datasets, manual digitization with GUI tools
remains the most reliable approach:

- 2D: [tpsDig2](https://www.sbmorphometrics.org/) (outputs TPS format natively),
  [ImageJ / FIJI](https://imagej.net/) (with point-picker plugins),
  [geomorph](https://cran.r-project.org/package=geomorph) (R)
- 3D: [SlicerMorph](https://slicermorph.github.io/) (extension for 3D Slicer; landmarking on CT/micro-CT volumes and surface meshes)

### Automated landmarking

For large datasets, machine learning tools can predict landmarks from images after training on a manually annotated subset:

- [ML-morph](https://github.com/agporto/ml-morph) (Porto & Voje, 2020):
  designed for morphometrics, accepts and outputs TPS format natively.
  Note: the repository has not been updated since 2020.
- [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut) (Mathis et
  al., 2018): actively maintained, large community, originally designed
  for animal pose estimation but applicable to landmark detection on
  static images. Outputs CSV/HDF5; conversion to TPS format is needed.

These methods require 50--200 manually annotated images for training
and are most worthwhile when the dataset exceeds a few hundred specimens.

| Criterion | Manual | Automated |
|-----------|--------|-----------|
| Setup cost | Low | High (training data needed) |
| Per-specimen cost | High | Low |
| Reproducibility | Operator-dependent | Consistent after training |

See {doc}`../tutorials/preprocessing/landmark_digitization_2d` for a hands-on tutorial on manual digitization.

```{seealso}
- {doc}`semilandmarks` for analyzing curves and surfaces using semilandmarks
- {doc}`morphometrics` for comparison with harmonic methods
- {doc}`../tutorials/preprocessing/landmark_digitization_2d` for digitizing landmarks from images
- {doc}`../tutorials/landmark/generalized_Procrustes_analysis` for practical examples
```

## References

- Bookstein, F.L., 1997. Morphometric tools for landmark data: geometry and biology, Cambridge University Press. Cambridge University Press.
- Claude, J., 2008. Morphometrics with R, Springer Science & Business Media. Springer Science & Business Media. <https://doi.org/10.1007/978-0-387-77789-4>
- Dryden, I.L., Mardia, K.V., 2016. Statistical Shape Analysis: With Applications in R, John Wiley & Sons. John Wiley & Sons.
- Mathis, A. et al., 2018. DeepLabCut: markerless pose estimation of user-defined body parts with deep learning. Nature Neuroscience, 21(9), 1281-1289. <https://doi.org/10.1038/s41593-018-0209-y>
- Porto, A., Voje, K.L., 2020. ML-morph: A fast, accurate and general approach for automated detection and landmarking of biological structures in images. Methods in Ecology and Evolution, 11(4), 500-512. <https://doi.org/10.1111/2041-210X.13373>
