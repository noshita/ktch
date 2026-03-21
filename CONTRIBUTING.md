# Contributing to ktch

Thank you for your interest in contributing to ktch!

Currently, this project is maintained by a small team and
is not actively accepting pull requests from external contributors.
If you find a bug, have a feature request, or want to suggest an improvement,
please [open an issue](https://github.com/noshita/ktch/issues).

The rest of this document describes the development workflow
and conventions used in this project.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) (package manager)
- Git

## Development Setup

```bash
git clone https://github.com/noshita/ktch.git
cd ktch
uv sync
```

### Running Tests

```bash
uv run pytest --benchmark-skip
```

Tests are co-located with source code at `ktch/<module>/tests/test_<name>.py`.
The CI matrix runs on Ubuntu, macOS, and Windows with Python 3.11, 3.12, and 3.13.

### Code Style

ktch uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.

```bash
uv run ruff check ktch/
uv run ruff format ktch/
```

### Building Documentation

```bash
cd doc
uv run make html
```

The documentation follows the [Diataxis](https://diataxis.fr/) framework
(`tutorials/`, `how-to/`, `explanation/`, `api/`).

## How to Contribute

The preferred way to contribute is through
[GitHub Issues](https://github.com/noshita/ktch/issues):

- Bug reports: describe the problem, steps to reproduce, and expected behavior
- Feature requests: describe the use case and expected behavior
- Questions: ask about usage, design decisions, or implementation details

## Development Workflow

This section documents the internal development workflow for reference.

1. Create a feature branch from `main`
2. Make your changes
3. Run `uv run ruff check ktch/` and `uv run ruff format ktch/` to ensure style compliance
4. Run `uv run pytest --benchmark-skip` to verify nothing is broken
5. Commit with a [Conventional Commits](#commit-messages) message
6. Submit a pull request against `main`

### Commit Messages

This project uses [Conventional Commits](https://www.conventionalcommits.org/)
and [release-please](https://github.com/googleapis/release-please) for
automated changelog generation and versioning.

Format:

```txt
<type>: <description>

[optional body]
```

#### Common types

| Type | Description | Changelog section |
|------|-------------|-------------------|
| `feat` | New feature | Features |
| `fix` | Bug fix | Bug Fixes |
| `docs` | Documentation only | Documentation |
| `refactor` | Code change that neither fixes a bug nor adds a feature | — |
| `test` | Adding or updating tests | — |
| `perf` | Performance improvement | Performance Improvements |
| `chore` | Maintenance tasks | Miscellaneous Chores |

#### Examples

```txt
feat: add 3D EFA normalization
fix: correct phase shift in harmonic reconstruction
docs: update CONTRIBUTING.md
```

## Code Conventions

- License: Apache License 2.0
- Code style: ruff (see [Code Style](#code-style))
- Docstrings: NumPy-style with reStructuredText math directives
- Tests: co-located at `ktch/<module>/tests/test_<name>.py`
- API design: scikit-learn compatible (`fit`, `transform`, `fit_transform`)

### Optional Dependencies

ktch splits optional dependencies into extras so that the core package
stays lightweight:

| Extra | Packages | conda-forge package |
|-------|----------|---------------------|
| `plot` | matplotlib, seaborn, plotly | `ktch-plot` |
| `data` | pooch | `ktch-data` |

Because `ktch/__init__.py` eagerly imports all submodules (including `plot`),
optional packages must never be imported at module level. Doing so would
break installations that do not have the extra installed (e.g., `ktch-data`
without matplotlib).

#### Pattern: `plot` module

The `plot` module centralizes optional dependency checks in
`ktch/plot/_base.py`. Other modules in the package use
`require_dependencies` as a guard, then import directly inside the
function body:

```python
# ktch/plot/_new_module.py
from ._base import require_dependencies

def my_plot_function(data, ax=None):
    require_dependencies("matplotlib")
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    # ... plot logic ...
    return ax
```

Key rules:

1. Import `require_dependencies` at the top of the file
2. Call `require_dependencies()` at the start of every public function
   that needs optional packages — this gives users a clear error message
   with install instructions
3. Import the library directly inside the function body after the guard
   (e.g., `import matplotlib.pyplot as plt`). Python caches modules in
   `sys.modules`, so repeated imports are effectively free (~100 ns).
   This pattern gives full IDE autocomplete and type-checker support.

To add a new optional dependency to `_base.py`, add its name to
`_VALID_DEPS`.

#### Pattern: `datasets` module

The `datasets` module uses a simpler try/except since it only has one
optional dependency:

```python
# ktch/datasets/_base.py
try:
    import pooch
except ImportError:
    pooch = None
```

Functions check `if pooch is None` and raise `ImportError` with install
instructions.

#### General guidelines

- Never add optional packages to the top-level imports in `__init__.py`
- Always provide an actionable error message showing both `pip` and `conda`
  install commands
- Test that `import ktch` succeeds without any optional dependency installed

## Code Organization

### Subpackage Structure

The codebase uses domain-driven subpackage organization. Each morphometric
method domain has its own subpackage:

- `ktch/landmark/` - Landmark-based morphometrics (GPA, TPS)
- `ktch/harmonic/` - Harmonic-based morphometrics (EFA, SPHARM)
- `ktch/io/` - File format readers/writers
- `ktch/datasets/` - Built-in example datasets
- `ktch/plot/` - Visualization functions
- `ktch/motion/` - Motion analysis utilities

### Module Naming

- Private implementation files use `_` prefix: `_procrustes_analysis.py`
- Public API is re-exported through `__init__.py`
- I/O module: one file per format (`_tps.py`, `_chc.py`, `_spharm_pdm.py`)
- Dataset loaders: `load_<type>_<name>()` functions in `_base.py`

### Import Conventions

```python
# Within a subpackage: relative imports
from ._kernels import tps_bending_energy

# Cross-subpackage: absolute imports
from ktch.datasets import load_outline_mosquito_wings
```

### Parallelization

Analysis classes that support `n_jobs` use `sklearn.utils.parallel`:

```python
from sklearn.utils.parallel import Parallel, delayed

results = Parallel(n_jobs=self.n_jobs)(
    delayed(self._process_single)(x) for x in X
)
```

### Deprecation

Renamed or moved APIs use `__getattr__` hooks with `DeprecationWarning`:

```python
# ktch/outline/__init__.py (deprecated -> ktch/harmonic/)
def __getattr__(name):
    if name == "EllipticFourierAnalysis":
        warnings.warn("...", DeprecationWarning, stacklevel=2)
        from ktch.harmonic import EllipticFourierAnalysis
        return EllipticFourierAnalysis
    raise AttributeError(...)
```

## Maintaining

For release procedures (Release Please, PyPI, conda-forge) and remote
dataset management (Cloudflare R2, registry), see
[MAINTAINING.md](MAINTAINING.md).
