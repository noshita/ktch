(installation)=

# Installation

ktch is available on [PyPI](https://pypi.org/project/ktch/) and [conda-forge](https://anaconda.org/conda-forge/ktch).

## From PyPI

```bash
pip install ktch
```

## From conda-forge

```bash
conda install -c conda-forge ktch
```

## Development Installation

For development, ktch uses [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
git clone https://github.com/noshita/ktch.git
cd ktch
uv sync
```

To run tests:

```bash
uv run pytest --benchmark-skip
```

To build documentation:

```bash
cd doc
uv run make html
```

## Dependencies

ktch requires:

- Python >= 3.9
- NumPy
- SciPy
- scikit-learn
- pandas

Optional dependencies for visualization:

- matplotlib
- seaborn
