# ktch - A Python package for model-based morphometrics

[![PyPI version](https://badge.fury.io/py/ktch.svg)](https://pypi.org/project/ktch/) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ktch/badges/version.svg)](https://anaconda.org/conda-forge/ktch) [![codecov](https://codecov.io/gh/noshita/ktch/branch/main/graph/badge.svg?token=SJN66K7KJY)](https://codecov.io/gh/noshita/ktch)

ktch is a Python package for model-based morphometrics with scikit-learn compatible APIs.

## Overview

ktch provides implementations of various morphometric analysis methods:

- Landmark-based methods: Generalized Procrustes Analysis (GPA) with semilandmark sliding
- Harmonic-based methods: Elliptic Fourier Analysis (EFA) for 2D/3D closed curves, spherical harmonic analysis for 3D closed surfaces
- File I/O: Support for standard morphometric file formats (TPS, CHC, SPHARM-PDM)
- Datasets: Example datasets for testing and learning

All analysis classes follow the scikit-learn API (`fit`, `transform`, `fit_transform`), making them easy to integrate into existing data analysis pipelines.

## Installation

Python >= 3.11 is required.

### From PyPI

```sh
pip install ktch
```

### From conda-forge

```sh
conda install -c conda-forge ktch
```

### Development Installation

```sh
git clone https://github.com/noshita/ktch.git
cd ktch
uv sync
```

## Quick Start

```python
from sklearn.decomposition import PCA

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis

# Load outline data (126 specimens, 100 points, 2D)
data = load_outline_mosquito_wings()
coords = data.coords.reshape(-1, 100, 2)

# Elliptic Fourier Analysis
efa = EllipticFourierAnalysis(n_harmonics=20)
coeffs = efa.fit_transform(coords)

# PCA on EFA coefficients
pca = PCA(n_components=5)
pc_scores = pca.fit_transform(coeffs)
```

## Documentation

See [doc.ktch.dev](https://doc.ktch.dev) for full documentation:

- Tutorials: Step-by-step guides for GPA, EFA, spherical harmonics, and more
- How-to guides: Task-oriented recipes for data loading, visualization, and pipeline integration
- Explanation: Theoretical background on morphometric methods
- API reference: Complete API documentation

## License

ktch is licensed under the Apache License, Version 2.0
