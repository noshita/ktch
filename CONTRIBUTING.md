# Contributing to ktch

Thank you for your interest in contributing to ktch!

ktch is maintained by a single researcher. Pull requests are not accepted
at this time because a single maintainer cannot reliably promise a review
turnaround. Instead, change proposals are welcome via
[GitHub Issues](https://github.com/noshita/ktch/issues), including bug
reports, feature requests, usage questions, and improvement suggestions.

The rest of this document describes the development workflow
and conventions used in this project.

## Prerequisites

- Python 3.11 or later
- [uv](https://docs.astral.sh/uv/) (package manager)
- Git

## Development setup

```bash
git clone https://github.com/noshita/ktch.git
cd ktch
uv sync
```

### Running tests

```bash
uv run pytest --benchmark-skip
```

Tests are co-located with source code at `ktch/<module>/tests/test_<name>.py`.

CI reads the Python versions from `pyproject.toml`: `requires-python` gives the
floor and the classifiers give the range. Linux runs every version in that
range, macOS and Windows run the newest, and Windows also runs the floor. Two
further jobs run on every push: one resolves the minimum versions declared for
each dependency instead of the lockfile, and one checks that an install without
the extras still imports every subpackage. A weekly run adds two more that
report without blocking, resolving the newest releases of every dependency and
installing ktch on top of the versions Google Colab ships.

To install the test tools without the notebook and lint groups:

```bash
uv sync --no-default-groups --group test
```

### Code style

ktch uses [ruff](https://docs.astral.sh/ruff/) for linting and formatting.
The ruff version is pinned in `uv.lock` (the `lint` dependency group).

Run it directly:

```bash
uv run ruff check ktch/
uv run ruff format ktch/
```

Or install the pre-commit hook so it runs automatically on changed files at
commit:

```bash
uv run pre-commit install
```

### Building documentation

```bash
cd doc
uv run make html
```

The documentation follows the [Diataxis](https://diataxis.fr/) framework
(`tutorials/`, `how-to/`, `explanation/`, `api/`).

## How to contribute

The preferred way to contribute is through
[GitHub Issues](https://github.com/noshita/ktch/issues):

- Bug reports: describe the problem, steps to reproduce, and expected behavior
- Feature requests: describe the use case and expected behavior
- Questions: ask about usage, design decisions, or implementation details

## Getting help

The support channel is
[GitHub Issues](https://github.com/noshita/ktch/issues). Usage questions
are welcome, not only bug reports. If you have a question about how to use
a function, whether a behavior is intended, or how to apply ktch to your
data, open an issue.

## Development workflow

This section documents the internal development workflow for reference.

1. Create a feature branch from `main`
2. Make your changes
3. Run `uv run ruff check ktch/` and `uv run ruff format ktch/` to ensure style compliance
4. Run `uv run pytest --benchmark-skip` to verify nothing is broken
5. Commit with a [Conventional Commits](#commit-messages) message
6. Submit a pull request against `main`

### Commit messages

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
| `perf` | Performance improvement | Performance Improvements |
| `refactor` | Code change that neither fixes a bug nor adds a feature | Code Refactoring |
| `test` | Adding or updating tests | — |
| `chore` | Maintenance tasks | — |
| `ci` | CI and workflow configuration | — |

Types with no changelog section do not produce a release on their own.

#### Breaking changes and deprecations

The type says what kind of change it is. `!` after the type, or a
`BREAKING CHANGE:` footer, says whether users have to be told. The two are
independent. Renaming, moving or removing a public name, or changing a
signature or an exception type, is not a new feature. Such a change is
`refactor!`, not `feat`.

Announce a deprecation as `feat` while the old spelling still works: users
need to see it, and nothing is broken yet.

#### Examples

```txt
feat: add 3D EFA normalization
fix: correct phase shift in harmonic reconstruction
docs: update CONTRIBUTING.md
```

## Code conventions

- License: Apache License 2.0
- Code style: ruff (see [Code style](#code-style))
- Docstrings: NumPy-style with reStructuredText math directives
- Tests: co-located at `ktch/<module>/tests/test_<name>.py`
- API design: scikit-learn compatible (`fit`, `transform`, `fit_transform`)

### Optional dependencies

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

## Code organization

### Subpackage structure

The codebase uses domain-driven subpackage organization. Each morphometric
method domain has its own subpackage:

- `ktch/landmark/` - Landmark-based morphometrics (GPA, TPS)
- `ktch/harmonic/` - Harmonic-based morphometrics (EFA, SPHARM)
- `ktch/io/` - File format readers/writers
- `ktch/datasets/` - Built-in example datasets
- `ktch/plot/` - Visualization functions
- `ktch/motion/` - Placeholder for planned motion analysis utilities
  (not implemented; excluded from the wheel)

### Naming, imports, and patterns

Module naming:

- Private implementation files use `_` prefix: `_procrustes_analysis.py`
- Public API is re-exported through `__init__.py`
- I/O module: one file per format (`_tps.py`, `_chc.py`, `_spharm_pdm.py`)
- Dataset loaders: `load_<type>_<name>()` functions in `_base.py`
- Name a module at the level of what it covers: analysis modules by their
  representation or method (e.g. `harmonic`, `landmark`); theoretical morphological model
  modules by the phenomenon they model, so that structure terms stay available
  to analysis modules.

Parameter naming:

- When a parameter transcribes a published model's mathematical symbol, use the
  lowercased, snake_case form of that symbol (PEP 8). 
  Recommend to document the symbol and its source equation in the docstring.

```python
# Within a subpackage: relative imports
from ._kernels import tps_bending_energy

# Cross-subpackage: absolute imports
from ktch.datasets import load_outline_mosquito_wings
```

Analysis classes that support `n_jobs` use `sklearn.utils.parallel`:

```python
from sklearn.utils.parallel import Parallel, delayed

results = Parallel(n_jobs=self.n_jobs)(
    delayed(self._process_single)(x) for x in X
)
```

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
