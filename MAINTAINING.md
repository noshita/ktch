# Maintaining ktch

This document describes operational procedures for maintaining the ktch
project. For development setup, conventions, and contribution guidelines,
see [CONTRIBUTING.md](CONTRIBUTING.md).

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

### Version numbering

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

### Pre-release checklist

1. Verify CI is green on `main`
2. Update `doc/_static/versions.json` on the release-please branch.
   Push the change to the release-please branch (not `main`) so that
   it only reaches `main` when the release PR is merged. This avoids
   triggering a push-based Docs build before the release tag exists.

   ```bash
   git fetch origin release-please--branches--main
   git checkout release-please--branches--main
   # edit doc/_static/versions.json
   git add doc/_static/versions.json
   git commit -m "docs: update versions.json for vX.Y.Z"
   git push origin release-please--branches--main
   ```

   Edit the `"name"` and `"version"` fields in the stable entry:

   ```json
   [
     {
       "name": "0.8.1 (stable)",
       "version": "0.8.1",
       "url": "https://doc.ktch.dev/stable/",
       "preferred": true
     },
     {
       "name": "dev",
       "version": "dev",
       "url": "https://doc.ktch.dev/dev/"
     }
   ]
   ```

   For minor releases that use `Release-As`, add the `Release-As`
   footer to a commit on `main` first to let release-please set the
   target version, then update `versions.json` on the release-please
   branch as described above.

3. If remote datasets were added or changed, update
   [`registry.toml`](#registrytoml) and run the
   [registry update script](#updating-the-dataset-registry).
   When data is unchanged, no registry update is needed — the loader
   falls back to the latest compatible version automatically.

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
   a new one from a personal fork.

   Note: the bot's PR branch lives on the bot's own fork
   (`regro-cf-autotick-bot/ktch-feedstock`), not on
   `conda-forge/ktch-feedstock`. You cannot push commits to the
   bot's branch even though the PR description says "Feel free to
   push to the bot's branch." Always create a new PR from your
   personal fork instead.

   The local clone used here is `noshita/ktch-feedstock` (personal fork),
   with `origin` pointing to the personal fork. The `--repo` flag tells
   `gh` to fetch the PR from the upstream `conda-forge/ktch-feedstock`
   even though `origin` is the personal fork.

   ```bash
   cd ktch-feedstock   # local clone of noshita/ktch-feedstock (personal fork)
   gh pr checkout <PR_NUMBER> --repo conda-forge/ktch-feedstock
   # Edit meta.yaml and commit
   git checkout -b <new-branch-name>
   git push origin <new-branch-name>
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
can publish it. The process requires two separate PRs: one to
`conda-forge/admin-requests` to register the output name, and one to
the feedstock to add the output definition.

Note: you cannot push directly to the bot's branch. Instead, check out
the bot PR locally, add your changes on a new branch, and open a new PR
from your personal fork. The bot PR is then closed.

1. Register the new output name via
   [conda-forge/admin-requests](https://github.com/conda-forge/admin-requests).
   Fork the repository and add a YAML file in the `requests/` directory
   following the
   [example template](https://github.com/conda-forge/admin-requests/blob/main/examples/example-add-feedstock-output.yml):

   ```yaml
   action: add_feedstock_output
   feedstock_to_output_mapping:
     ktch:
       - ktch-data
   ```

   Link the feedstock PR (step 2) in the description. The feedstock PR's
   CI will not pass until this admin-requests PR is merged.

2. Add the new output to the feedstock. If a bot version-update PR
   exists, base your work on it (so the version bump is included).
   If the bot PR is already merged, create a standalone PR and bump
   the build number instead.

   ```bash
   cd ktch-feedstock   # local clone of noshita/ktch-feedstock (personal fork)

   # Check out the bot PR from the upstream conda-forge repo.
   # --repo is required because origin points to the personal fork.
   gh pr checkout <BOT_PR_NUMBER> --repo conda-forge/ktch-feedstock

   # Create a new branch on the personal fork
   git checkout -b add-ktch-data-output
   ```

   Edit `meta.yaml` to add the new output block and update `ktch-all`:

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

   Then commit, push, and open a PR:

   ```bash
   git add meta.yaml
   git commit -m "add ktch-data output"
   git push origin add-ktch-data-output

   gh pr create --repo conda-forge/ktch-feedstock \
     --head noshita:add-ktch-data-output \
     --title "add ktch-data output" \
     --body "Add ktch-data output for [data] extra. Closes #<BOT_PR_NUMBER>"

   # Close the original bot PR
   gh pr close <BOT_PR_NUMBER> --repo conda-forge/ktch-feedstock
   ```

3. After the admin-requests PR is merged, the feedstock PR's CI will
   pass. Review and merge it

### Post-release verification

- [ ] <https://doc.ktch.dev/stable/> shows the new version
- [ ] Version switcher works correctly
- [ ] <https://pypi.org/project/ktch/> shows the new version
- [ ] conda-forge feedstock PR is created (may take a few hours)
- [ ] (Minor releases only) Re-run the Docs workflow with cache disabled
  (see below)

#### Re-building documentation without cache

Before v1.0, minor version releases (e.g., 0.8.0 → 0.9.0) may introduce
API changes that affect notebook outputs. To ensure all notebooks are
re-executed from scratch, manually trigger the Docs workflow with the
"Disable notebook execution cache" option checked:

Actions → Docs → Run workflow → check "Disable notebook execution cache"

This is not automated because patch releases rarely need it and the
no-cache build takes significantly longer.

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

## Remote datasets (Cloudflare R2)

Large datasets are hosted on Cloudflare R2 and downloaded on demand via
[pooch](https://www.fatiando.org/pooch/). R2 was chosen for its free egress
and easy public access. This section describes the infrastructure and
procedures for adding or updating remote datasets.

### R2 configuration

| Item | Value |
|------|-------|
| Bucket name | `ktch-datasets` |
| Region | Automatic (R2 is regionless) |
| Public access | Public Development URL (`r2.dev`) |
| Base URL | `https://pub-c1d6dba6c94843f88f0fd096d19c0831.r2.dev` |

### Bucket directory structure

```txt
ktch-datasets/
└── datasets/
    └── image_passiflora_leaves/
        ├── v1/
        │   ├── manifest.json
        │   └── image_passiflora_leaves.zip
        └── v2/
            ├── manifest.json
            └── image_passiflora_leaves.zip
```

Each dataset has its own numeric version sequence (v1, v2, ...) independent
of the ktch package version. When adding or updating a dataset, place files
under `datasets/{dataset_name}/v{N}/`.

### Zip archive layout

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

### registry.toml

`ktch/datasets/registry.toml` is the source of truth for dataset
configuration. It defines which datasets and versions the registry
update script should fetch manifests for:

```toml
[image_passiflora_leaves]
default = "2"
versions = ["1", "2"]
```

When adding a new dataset or version, update this file first.

### Updating the dataset registry

Once the zip archive(s) and `manifest.json` are uploaded to R2 under
`datasets/{dataset_name}/v{N}/`, update the local registry with the
following steps:

#### 1. Update `registry.toml`

 Add the new version to `ktch/datasets/registry.toml`. For example,
 to add version 3 of `image_passiflora_leaves`:

 ```toml
 [image_passiflora_leaves]
 default = "3"
 versions = ["1", "2", "3"]
 ```

#### 2. Run the registry update script

 ```bash
 uv run python scripts/update_registry.py
 ```

 This reads `registry.toml`, fetches `manifest.json` for each
 dataset/version from R2, validates the SHA256 hashes, and updates
 `ktch/datasets/_registry.py` automatically.

 Use `--dry-run` to preview changes without modifying the file:

 ```bash
 uv run python scripts/update_registry.py --dry-run
 ```

#### 3. Run tests to verify the registry update

 ```bash
 uv run pytest --benchmark-skip
 ```

 `test_default_version_in_registry` will automatically verify the new entry.

#### 4. Commit the registry change

 ```bash
 git add ktch/datasets/registry.toml ktch/datasets/_registry.py
 git commit -m "feat: update dataset registry"
 ```

### pooch dependency policy

Following scikit-image's approach, pooch is not a core dependency:

```toml
[project.optional-dependencies]
data = ["pooch>=1.3"]
```

- Basic install (`pip install ktch`): no pooch
- Dataset download (`pip install ktch[data]`): pooch included
- When pooch is missing, `ImportError` with an informative message is raised

### Testing strategy for remote datasets

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
