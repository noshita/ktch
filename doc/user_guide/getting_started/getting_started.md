(getting-sterated)=

# Getting Started

ktch allows you to conduct model-based morphometrics via scikit-learn compatible API efficiently.

## Install

### From PyPI or conda-forge

ktch is currently available on [PyPI](https://pypi.org/project/ktch/) and [conda-forge](https://anaconda.org/conda-forge/ktch).
You can install it with pip::

    pip install ktch

or with conda::

    conda install -c conda-forge ktch

## Quick start

This example loads the mosquito wing outline dataset (Rohlf and Archie 1984), and calculates elliptic Fourier descriptors (EFDs).
Finally, it performs principal component analysis (PCA) on the EFDs.

```python

from sklearn.decomposition import PCA

from ktch.datasets import load_outline_mosquito_wings
from ktch.harmonic import EllipticFourierAnalysis

# load data
data_outline_mosquito_wings = load_outline_mosquito_wings()
X = data_outline_mosquito_wings.coords.to_numpy().reshape(-1,100,2)

# EFD
efa = EllipticFourierAnalysis(n_components=20)
coef = efa.transform(X)

# PCA
pca = PCA(n_components=3)
pcscores = pca.fit_transform(coef)

```
