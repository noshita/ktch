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

## Releasing

This project uses [release-please](https://github.com/googleapis/release-please)
to automate versioning, changelog generation, and GitHub Release creation.
The release flow is driven by Conventional Commits on `main`.

### Overview

```txt
Conventional Commits on main
  -> Release Please creates/updates a release PR
    (bumps pyproject.toml, CHANGELOG.md, .release-please-manifest.json)
  -> Maintainer merges the PR (merge commit)
    -> Release Please creates v0.X.Y tag + GitHub Release
      -> documentation.yml triggers (release: published)
      -> sphinx-multiversion builds /stable/ and /dev/
  -> Maintainer publishes to PyPI
    -> conda-forge feedstock auto-creates a PR
      -> Maintainer reviews and merges feedstock PR(s)
```

### Version Numbering

Release Please determines the next version from Conventional Commits
automatically. The current config (`bump-patch-for-minor-pre-major: true`,
`bump-minor-pre-major: true`) produces the following bumps while
the version is below 1.0.0:

| Commit type | Version bump |
|-------------|-------------|
| `fix:` | patch (0.7.0 → 0.7.1) |
| `feat:` | patch (0.7.0 → 0.7.1) |
| `feat!:` / `BREAKING CHANGE` | minor (0.7.0 → 0.8.0) |
| `docs:`, `chore:`, etc. | no bump |

Since `feat:` only produces a patch bump in this configuration, a minor
version bump for feature releases requires explicit specification via
`Release-As`. The recommended workflow is:

1. Develop normally — `feat:` and `fix:` commits trigger Release Please
   to create a release PR with an automatic patch bump
2. Before merging the release PR, if a minor bump is intended,
   add a `Release-As` footer to a commit on `main`:

   ```txt
   feat: add some feature

   Release-As: 0.8.0
   ```

3. Release Please updates the existing PR to target the specified version

Adding `Release-As` just before merging (rather than immediately after the
previous release) keeps the version flexible — e.g., if an urgent patch
release is needed in the meantime, the automatic patch bump can be merged
without conflict.

Do not manually edit `pyproject.toml` version — let Release Please manage it.

### Pre-release Checklist

1. Verify CI is green on `main`
2. Update `doc/_static/versions.json` for the new stable version:

   ```bash
   uv run python scripts/update_versions_json.py <version>
   ```

   For example:

   ```bash
   uv run python scripts/update_versions_json.py 0.8.0
   ```

   Use `--dry-run` to preview changes without modifying the file.
   Commit this change to `main` before merging the release PR,
   so that the tagged commit includes the correct version switcher
   configuration.

3. If remote datasets were added or updated, ensure
   the [dataset registry](#updating-the-dataset-registry) is up to date.

### Merging the Release Please PR

1. Review the auto-generated CHANGELOG in the PR
2. Approve and merge with a merge commit:

   ```bash
   gh pr merge <PR_NUMBER> --merge
   ```

   After merge, Release Please automatically:
   - Creates a `v0.X.Y` tag
   - Creates a GitHub Release with the changelog
   - Triggers the documentation workflow via `release: published`

### Publishing to PyPI

After the GitHub Release is created:

```bash
uv build
uv publish
```

> Note: PyPI publishing could be automated using
> [Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
> with GitHub Actions in the future.

### Updating conda-forge

The [ktch-feedstock](https://github.com/conda-forge/ktch-feedstock) produces
the following packages:

| Package | Description |
|---------|-------------|
| `ktch` | Core package |
| `ktch-data` | `[data]` extra (pooch) |
| `ktch-plot` | `[plot]` extra (matplotlib, plotly, seaborn) |
| `ktch-all` | Metapackage depending on all extras |

See the conda-forge
[Maintaining packages](https://conda-forge.org/docs/maintainer/updating_pkgs/)
guide for general reference.

#### Version update (routine)

After the PyPI package is published:

1. The regro-cf-autotick-bot typically creates a version update PR
   in the feedstock repository within a few hours
2. The bot updates the source URL, version, and SHA256 hash in `meta.yaml`
   automatically
3. Review the PR — the bot does not update dependency version
   constraints, so check `pyproject.toml` against `meta.yaml` and
   fix any mismatches. To apply fixes, close the bot PR and create
   a new one from a personal fork:

   ```bash
   cd ktch-feedstock   # conda-forge/ktch-feedstock clone
   gh pr checkout <PR_NUMBER>
   # Edit and commit
   git checkout -b <new-branch-name>
   git push fork <new-branch-name>
   gh pr create --repo conda-forge/ktch-feedstock \
     --head noshita:<new-branch-name>
   gh pr close <PR_NUMBER> --repo conda-forge/ktch-feedstock
   ```

4. Merge the feedstock PR after CI passes

If the bot PR does not appear, check that the feedstock does not already
have 3+ open version update PRs (the bot stops after 3).

For changes independent of a bot PR (e.g., recipe-only fixes),
create a PR from a personal fork. Do not create branches directly
on the feedstock repository, as pushes to the main repo trigger CI
and may cause unintended package publishing.

#### Adding a new output

When a new optional dependency group is added to `pyproject.toml`
(e.g., `[data]`), a new output must be registered before the feedstock
can publish it.

1. Add the new output to the version update PR. Push the following
   changes to the bot's branch (or create a new feedstock PR if the
   version update is already merged — in that case, bump the build
   number):

   ```yaml
   - name: {{ name }}-data
     build:
       noarch: python
     requirements:
       run:
         - {{ pin_subpackage(name, exact=True) }}
         - pooch >=1.3
     test:
       imports:
         - ktch.datasets
   ```

   Update `ktch-all` to depend on the new output. This PR will not
   pass CI until step 3 is complete, but it serves as context for the
   registration request.

2. Create a PR to
   [conda-forge/admin-requests](https://github.com/conda-forge/admin-requests)
   to register the new output name. Fork the repository and add a YAML
   file in the `requests/` directory following the
   [example template](https://github.com/conda-forge/admin-requests/blob/main/examples/example-add-feedstock-output.yml):

   ```yaml
   action: add_feedstock_output
   feedstock_to_output_mapping:
     ktch:
       - ktch-data
   ```

   Link the feedstock PR from step 1 in the description.

3. After the admin-requests PR is merged, the feedstock PR's CI will
   pass. Review and merge it

### Post-release Verification

- [ ] <https://doc.ktch.dev/stable/> shows the new version
- [ ] Version switcher works correctly
- [ ] <https://pypi.org/project/ktch/> shows the new version
- [ ] conda-forge feedstock PR is created (may take a few hours)

### Troubleshooting

#### Documentation shows 404 at /stable/

`versions.json` was not updated before the release, or the version listed
does not match any tag. Verify that `doc/_static/versions.json` contains
the released version and that the corresponding `v0.X.Y` tag exists.
A `workflow_dispatch` run of the Docs workflow can rebuild without a new release.

#### Release Please PR not appearing

Ensure recent commits on `main` include at least one `feat:` or `fix:` commit.
Commits with types like `docs:`, `chore:`, or `refactor:` alone do not trigger
a version bump.

#### Release Please picks the wrong version

If the automatic version bump does not match the intended release version,
add `Release-As: X.Y.Z` to a commit footer on `main`. This overrides
the automatic calculation for the next release PR.

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
