# ktch - A python package for model-based morphometrics

[![PyPI version](https://badge.fury.io/py/ktch.svg)](https://pypi.org/project/ktch/) [![Anaconda-Server Badge](https://anaconda.org/conda-forge/ktch/badges/version.svg)](https://anaconda.org/conda-forge/ktch) [![codecov](https://codecov.io/gh/noshita/ktch/branch/main/graph/badge.svg?token=SJN66K7KJY)](https://codecov.io/gh/noshita/ktch)

**ktch** is a Python package for model-based morphometrics, through scikit-learn compatible APIs.

## Overview

ktch provides implementations of various morphometric analysis methods:

- **Landmark-based methods**: Generalized Procrustes Analysis (GPA) for shape analysis
- **Harmonic-based methods**: Elliptic Fourier Analysis (EFA) for 2D and 3D closed curves, spherical harmonic analysis for 3D closed surfaces.
- **File I/O**: Support for standard morphometric file formats (TPS, CHC)
- **Datasets**: Example datasets for testing and learning

All methods follow the scikit-learn APIs, making them easy to integrate into existing data analysis pipelines.

## Installation

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
pip install -e .
```

## License

ktch is licensed under the Apache License, Version2.0
