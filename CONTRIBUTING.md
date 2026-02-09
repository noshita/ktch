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
The CI matrix runs on Ubuntu, macOS, and Windows with Python 3.11 and 3.12.

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

## Remote Datasets (Cloudflare R2)

Large datasets are hosted on Cloudflare R2 and downloaded on demand via
[pooch](https://www.fatiando.org/pooch/). R2 was chosen for its free egress
and easy public access. This section describes the infrastructure and
procedures for adding or updating remote datasets.

### R2 Configuration

| Item | Value |
|------|-------|
| Bucket name | `ktch-datasets` |
| Region | Automatic (R2 is regionless) |
| Public access | Public Development URL (`r2.dev`) |
| Base URL | `https://pub-c1d6dba6c94843f88f0fd096d19c0831.r2.dev` |

### Bucket Directory Structure

```txt
ktch-datasets/
└── releases/
    └── v0.7.0/
        ├── manifest.json
        └── image_passiflora_leaves.zip
```

Versions correspond to ktch package versions. When releasing a new version,
place files under `releases/vX.Y.Z/`.

### Zip Archive Layout

Each dataset zip should contain a top-level directory matching the dataset name:

```txt
image_passiflora_leaves.zip
└── image_passiflora_leaves/
    ├── metadata.csv
    └── images/
        ├── <image_id>.png
        └── ...
```

### manifest.json

Each version directory on R2 contains a `manifest.json` that maps dataset
filenames to their SHA256 hashes:

```json
{
  "image_passiflora_leaves.zip": "<sha256-hex-string>"
}
```

This file is used by `scripts/update_registry.py` to automatically update
the local registry without manual hash computation.

### Updating the Dataset Registry

Once the zip archive(s) and `manifest.json` are uploaded to R2 under
`releases/vX.Y.Z/`, update the local registry with the following steps:

#### 1. Run the registry update script

 ```bash
 uv run python scripts/update_registry.py <version>
 ```

 For example,

 ```bash
 uv run python scripts/update_registry.py 0.8.0
 ```

 This fetches `manifest.json` from R2, validates the SHA256 hashes, and
 updates `ktch/datasets/_registry.py` automatically.

 Use `--dry-run` to preview changes without modifying the file:

 ```bash
 uv run python scripts/update_registry.py --dry-run 0.8.0
 ```

#### 2. Run tests to verify the registry update

 ```bash
 uv run pytest --benchmark-skip
 ```

 `test_default_version_in_registry` will automatically verify the new entry.

#### 3. Commit the registry change

 ```bash
 git add ktch/datasets/_registry.py
 git commit -m "feat: update dataset registry for v0.8.0"
 ```

### pooch Dependency Policy

Following scikit-image's approach, pooch is **not** a core dependency:

```toml
[project.optional-dependencies]
data = ["pooch>=1.3"]
```

- Basic install (`pip install ktch`): no pooch
- Dataset download (`pip install ktch[data]`): pooch included
- When pooch is missing, `ImportError` with an informative message is raised

### Testing Strategy for Remote Datasets

| Test | Scope | Runs |
|------|-------|------|
| Function signature and pooch error handling | Unit | Always |
| Actual data download and loading | Integration | Skipped by default (`@pytest.mark.skip`) |
| Default version exists in registry | Registry integrity | Always |
| All registered versions have valid entries | Registry integrity | Always |
| Version detection format and consistency | Version logic | Always |

Tests are designed around the default version (auto-detected from the package version)
to avoid hardcoded version strings. When a new version is added to the registry,
`test_default_version_in_registry` catches any omissions.
