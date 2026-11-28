# Documentation

## Building locally

```bash
cd doc
uv run make html
```

The output is generated in `doc/_build/html/`.

The published site is built with the `dirhtml` builder, which serves each
page at `page/` instead of `page.html`. To reproduce the published URL form
locally:

```bash
cd doc
uv run make dirhtml   # output in doc/_build/dirhtml/
```

For live-reload during editing:

```bash
cd doc
uv run sphinx-autobuild . _build/html
```

A notebook that raises an error fails the build. Tag a cell that is meant to
raise with `raises-exception`.

## Versions

CI builds each version from its own git ref, in the environment that ref's
`uv.lock` records (`.github/workflows/docs-build.yml`). A released version is
built once, and the build is stored on its GitHub release.
`.github/workflows/documentation.yml` builds `dev` from `main` and assembles the
site from it and the stored builds of the versions listed in
`_static/versions.json` (`scripts/assemble_site.py`). A change under `doc/` on
`main` therefore reaches `/dev/` only. To change a released version's pages,
see [Documentation builds](../MAINTAINING.md#documentation-builds).

The build sets `KTCH_DOCS_VERSION` (`main` or `vX.Y.Z`). `conf.py` reads it to
set the version switcher's entry and the canonical URLs.

## Redirects

`scripts/gen_redirects.py` writes redirect stubs into the built site so that old
URLs keep working; `scripts/assemble_site.py` runs it on every deployment and
`.github/workflows/docs-release-check.yml` on the release PR. It
has three modes: `stubs` maps the legacy pre-`/stable/` layout URLs listed in
`doc/_redirects.toml` onto current pages (and fails the build if a target is
missing); `html-aliases` keeps every old `page.html` URL working after the
dirhtml switch by walking the built tree; `cloudflare` emits a `_redirects` file
of real 301s for a future move off GitHub Pages. See
`scripts/gen_redirects.py` for details.

## Release workflow

See the [Releasing](../MAINTAINING.md#releasing) section in MAINTAINING.md.

## License check

[LicenseCheck](https://github.com/FHPythonUtils/LicenseCheck) verifies that
all dependency licenses are compatible with the project license (Apache-2.0).
Licenses to reject are configured in `pyproject.toml` under `[tool.licensecheck]`.

```bash
uvx licensecheck -r pyproject.toml --zero
```

`--zero` returns a non-zero exit code if an incompatible license is found,
suitable for CI use.

## License

The documentation is licensed under a
Creative Commons Attribution 4.0 International (CC BY 4.0) License.

The documentation system may also include some components licensed under
open source licenses.

### scikit-learn

The following items forked from
[scikit-learn](https://github.com/scikit-learn/scikit-learn)
are licensed under the BSD 3-Clause License.

- `sphinxext/override_pst_pagetoc.py`
- `_templates/base.rst`
