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
      -> Release Please dispatches documentation.yml (workflow_dispatch)
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
| `fix:`, `feat:`, `docs:`, `perf:`, `refactor:`, `revert:` | patch (0.7.0 → 0.7.1) |
| any type with `!` or a `BREAKING CHANGE` footer | minor (0.7.0 → 0.8.0) |
| `style:`, `chore:`, `test:`, `build:`, `ci:` | no release PR |

Whether a release PR appears depends not on the version impact of the type, but
on whether the changelog would come out empty. Types marked `hidden` in
`release-please-config.json` produce no changelog entry, and commits of only
those types leave nothing to release. A `!` or a `BREAKING CHANGE` footer is
rendered even for a hidden type.

`refactor:` is visible on purpose while the API is still moving. Hide it at the
1.0 transition, once `refactor` reliably means a change no user can observe.

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

   Do this once the last commit of the release has landed on `main`.
   Release Please rebuilds its branch from `main` on every push and
   force-pushes the result, discarding a `versions.json` commit made
   before that.

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

4. Verify what the source distribution ships:

   ```bash
   rm -rf dist
   uv build
   tar -tzf dist/ktch-*.tar.gz
   ```

   The archive should contain `ktch/` plus `CHANGELOG.md`, `LICENSE`,
   `README.md`, `pyproject.toml`, `PKG-INFO`, and `.gitignore`, and nothing
   else. `[tool.hatch.build.targets.sdist]` selects the contents by
   allowlist, which replaced a denylist that had shipped untracked working
   directories in earlier releases. A new top-level file that belongs in the
   distribution has to be added to that list; anything else stays out.

5. Confirm that the supported-version ends have not moved, per
   [Supported versions policy](#supported-versions-policy):

   ```bash
   curl -s https://raw.githubusercontent.com/conda-forge/conda-forge-pinning-feedstock/main/recipe/conda_build_config.yaml | grep -A3 '^python_min'
   curl -s https://raw.githubusercontent.com/googlecolab/backend-info/main/os-info.txt | grep Python
   ```

   A change in either value does not block the release. It schedules the
   floor update for the next minor release.

### Merging the Release Please PR

1. Review the auto-generated CHANGELOG in the PR
2. Approve and merge with a merge commit:

   ```bash
   gh pr merge <PR_NUMBER> --merge
   ```

   After merge, Release Please automatically:
   - Creates a `v0.X.Y` tag
   - Creates a GitHub Release with the changelog
   - Dispatches the Docs workflow, which rebuilds `/stable/` and `/dev/`

   The dispatch is deliberate rather than event-driven. A release created with
   `GITHUB_TOKEN` does not start a workflow. The `release: published`
   trigger on `documentation.yml` never fires; `workflow_dispatch` is the
   documented exception. The Docs workflow also skips its own push build for
   this merge commit, because that build would race the tag it needs.

### Publishing to PyPI

After the GitHub Release is created:

```bash
rm -rf dist
uv build
uv publish
```

`uv publish` uploads every file in `dist/`. Clear the directory first;
artifacts built for an earlier release would otherwise be uploaded again.

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
3. Review the PR. The bot updates the source URL, the version, and the
   hash, and leaves everything else alone. Check `pyproject.toml` against
   `meta.yaml` for dependency floors that moved under
   [Supported versions policy](#supported-versions-policy), and confirm the
   recipe does not set `python_min`. That value comes from the conda-forge
   pinning.

   Fixes go on the bot's branch. The PR description invites this and the
   branch allows edits by maintainers, which a `maintainerCanModify` of
   `true` on the PR confirms. The local clone has `bot` pointing at the
   bot's fork, `origin` at the personal fork, and `upstream` at
   `conda-forge/ktch-feedstock`.

   ```bash
   cd ktch-feedstock
   git fetch bot <BOT_BRANCH>   # e.g. 0.11.1_hdf33bd, shown on the PR page
   git switch --track bot/<BOT_BRANCH>
   # edit recipe/meta.yaml
   git commit -am "<what changed>"
   git push bot HEAD
   gh pr checks <PR_NUMBER> --repo conda-forge/ktch-feedstock --watch
   ```

   Pushing re-runs both the build and the linting service. If the push is
   rejected, fall back to a PR from the personal fork and close the bot PR:

   ```bash
   gh pr checkout <PR_NUMBER> --repo conda-forge/ktch-feedstock
   git switch -c <new-branch-name>
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

  The first item is the only detector for a dispatch that never ran. A dispatch
  that fails turns the release-please run red, but a step that is removed or
  whose condition stops matching produces no build and no failed run, and the
  push build that used to cover this case by accident is now skipped.
- [ ] <https://pypi.org/project/ktch/> shows the new version
- [ ] conda-forge feedstock PR is created (may take a few hours)
- [ ] (Minor releases only) Re-run the Docs workflow with cache disabled
  (see below)
- [ ] Google Search Console: sitemap (`/stable/sitemap.xml`) is detected
  and page count is correct

`/stable/` is rebuilt from the release tag, so any documentation *source* fix
merged to `main` after the previous release first appears on `/stable/` with
this release (it was only on `/dev/` before). `conf.py` and workflow changes, by
contrast, apply to `/stable/` on every build regardless of the tag. When a
release is expected to carry such a docs fix, confirm it on `/stable/` here.

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

Ensure recent commits on `main` include at least one commit of a type that
appears in the changelog (see [Version numbering](#version-numbering)).
`style:`, `chore:`, `test:`, `build:` and `ci:` are hidden; commits of only
those types leave the changelog empty and produce no release PR.

#### Release Please picks the wrong version

If the automatic version bump does not match the intended release version,
add `Release-As: X.Y.Z` to a commit footer on `main`. This overrides
the automatic calculation for the next release PR.

#### Pages not indexed by Google

SEO configuration of the documentation site:

- `robots.txt`: allows `/stable/`, blocks `/dev/` and versioned paths
- `sphinx-sitemap`: generates `/stable/sitemap.xml` with clean `/page/`
  (dirhtml) URLs; `sitemap_excludes` keeps builder-generated pages
  (`genindex/`, `py-modindex/`, `search/`, `opensearch/`) out
- `noindex_utilities` (custom Sphinx extension): adds
  `<meta name="robots" content="noindex">` to low-value pages
  (`_modules/*`, `genindex`, `py-modindex`, `search`, `opensearch`)
- `sphinxext-opengraph`: generates Open Graph meta tags
- `scripts/gen_redirects.py`: redirect stubs for legacy and pre-dirhtml URLs
- `scripts/stamp_sitemap_lastmod.py`: dates each sitemap entry from the last
  commit that touched the page's source. It runs in `documentation.yml` after
  the build, because sphinx-multiversion builds each ref from a `git archive`
  where no git lookup works. Pages with no source in the repository, the
  generated API reference among them, are left without a date

The technical crawl configuration above was audited in July 2026 and is
healthy: the sitemap is valid, `robots.txt` allows `/stable/`, canonical tags
are correct, and the property is verified. Repeatedly re-checking these has not
moved indexing, because they are not the bottleneck.

When GSC reports "Crawled - currently not indexed", that is a quality/authority
judgment on pages Google fetched successfully, not a crawl block. The dominant
lever is inbound authority: dofollow, in-content links from indexed third-party
pages. Note that PyPI and GitHub links to the docs carry `rel="nofollow"` and
transfer no ranking signal. Higher-value actions than any further technical
tweak: a JOSS paper or other citation, and links from pages other people
control.

Only if a specific page is genuinely missing from the index, first rule out the
mechanical causes: the URL is in `/stable/sitemap.xml`, returns 200 (not a
redirect stub or 404), and has no unintended `noindex`; then request indexing
via GSC's URL Inspection tool.

## Supported versions policy

ktch declares a minimum Python version and a minimum version for each
dependency. This section fixes how those minimums move, so that the reasoning
does not have to be reconstructed at every release.

### Python

The floor equals conda-forge's `python_min`, as long as that value stays at or
below the Python that Google Colab runs. Both ends are external. Check them
rather than assume them.

conda-forge sets the lower end. CFEP-25 pins `python_min` in the global
pinning file, and conda-forge does not ship new `noarch: python` packages
below it. Declaring a lower floor in `pyproject.toml` therefore does not
produce a conda package that installs on an older Python.

Colab sets the upper end. Its Python version cannot be changed from inside a
session, and the Colab FAQ names it as an example of a core dependency that
cannot be changed back. Libraries are a separate matter: `pip install`
upgrades them at the cost of a session restart. Colab therefore constrains the
Python floor only. Colab also lets a notebook pin a past runtime for one year,
and workshop material depends on that. Keep the floor at or below the Python
of the oldest pinnable runtime.

When conda-forge raises `python_min`, raise `requires-python` to match in the
next minor release. If `python_min` ever rises above the Python that Colab
runs, keep `requires-python` at the Colab version and let the conda package's
floor move on its own. CFEP-25 allows the two ecosystems to differ, and states
the reason: conda-forge will not ship new packages for Python versions it no
longer supports.

The feedstock recipe must not redefine `python_min`. Without a redefinition
the conda side follows the global pinning by itself, which is what keeps this
rule self-maintaining.

```bash
curl -s https://raw.githubusercontent.com/conda-forge/conda-forge-pinning-feedstock/main/recipe/conda_build_config.yaml | grep -A3 '^python_min'
curl -s https://raw.githubusercontent.com/googlecolab/backend-info/main/os-info.txt | grep Python
```

The `googlecolab/backend-info` README also lists the pinnable past runtimes
with their Python versions.

### Dependency tiers

| Tier | Packages | Floor policy |
|---|---|---|
| 1 | numpy, scipy, pandas, scikit-learn | Track the newest major version |
| 2 | matplotlib, plotly, seaborn, pillow, pyarrow, pooch | Whatever the APIs in use require |

Tier 1 is the layer the public API is built on. Keeping current there is what
keeps the code free of deprecated usage. Tier 2 reaches users through extras,
where raising a floor excludes environments for little gain. polars is Tier 2
and is not a dependency today.

Being compatible with a new release and requiring it are separate decisions.
The first is routine and breaks nothing. The second excludes environments and
needs the conditions below.

A new upstream release should be green in CI within one month for a major
version, and by the next ktch release for a minor version. The weekly CI job
that resolves the newest compatible releases is the detector.

### Raising a floor

Raise a floor in the next minor release once all four conditions hold:

1. The version is available on conda-forge's main channel
2. CI passes with the newest resolution
3. Three months have passed since that version was released
4. The invariant below still holds

Record the reason in the changelog entry, such as the API the new floor makes
available or the compatibility code it lets us delete.

[SPEC 0](https://scientific-python.org/specs/spec-0000/) is the backstop at
the other end. It is the Scientific Python ecosystem's shared recommendation
for support windows: drop a Python version 3 years after its release and a
core package 2 years after its release. A floor should not be older than
that.

### The Python floor caps the dependency floors

No dependency floor may require a Python newer than ktch's own Python floor.

Core packages drop old Python versions on their own schedule, and the newest
release of one of them regularly requires a Python newer than the floor. The
floor therefore caps every dependency floor, and the cap moves on someone
else's timetable. Breaking the invariant also breaks the conda-forge build,
because the recipe builds and tests in a host environment pinned to the
minimum Python.

Operations therefore run in one order: conda-forge raises `python_min`, then
ktch raises `requires-python`, then the dependency floors can move.

```bash
curl -s https://pypi.org/pypi/<package>/<version>/json \
  | python -c "import json,sys; print(json.load(sys.stdin)['info']['requires_python'])"
```

### Where the versions are declared

| Location | What it carries |
|---|---|
| `pyproject.toml` `requires-python` | The Python floor |
| `pyproject.toml` classifiers | The supported Python versions |
| `pyproject.toml` dependencies, optional-dependencies | The dependency floors |
| `.github/workflows/test-codecov.yml` | The jobs that test them; the versions are read from `pyproject.toml` |
| `README.md`, installation section | The Python floor, in prose |
| `doc/installation.md`, dependencies section | The Python floor and the dependency floors |

The feedstock recipe follows the Python floor on its own. Dependency floors in
`recipe/meta.yaml` do not: update them by hand in the next version-bump PR, as
described under [Version update (routine)](#version-update-routine).

### How CI covers the range

The `versions` job reads `requires-python` and the Python classifiers, and
every other job takes its Python version from that output. Adding a version to
the classifiers therefore adds it to the matrix.

ktch is pure Python, and what differs between operating systems is file IO and
the plotting backends rather than the language minor version. Linux runs the
whole supported range. macOS and Windows run the newest version, and Windows
also runs the floor, since the file parsers meet different path and newline
handling there.

Three jobs stand outside the matrix. `floors` resolves the declared minimums
instead of the lockfile, which is what keeps them honest. `core` checks that
an install without the extras imports every subpackage. The weekly `latest`
and `colab` jobs report without blocking: one resolves the newest compatible
releases, the other installs ktch on top of the versions Colab ships.

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
├── datasets/
│   └── image_passiflora_leaves/
│       ├── v1/
│       │   ├── manifest.json
│       │   └── image_passiflora_leaves.zip
│       └── v2/
│           ├── manifest.json
│           └── image_passiflora_leaves.zip
└── examples/
    └── danshaku_08_allSegments_para/
        └── v1/
            ├── manifest.json
            └── danshaku_08_allSegments_para.vtp
```

The bucket has two top-level prefixes:

- `datasets/` -- curated datasets loaded by `load_*` functions (typically
  zip archives containing multiple files).
- `examples/` -- individual sample files fetched by `ktch.datasets.fetch()`
  for tutorials and demonstrations (typically single files, not zipped).

Each entry has its own numeric version sequence (v1, v2, ...) independent
of the ktch package version. When adding or updating a dataset, place files
under `datasets/{dataset_name}/v{N}/`. For example data, use
`examples/{stem}/v{N}/` where `{stem}` is the filename without extension.

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
