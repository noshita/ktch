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

## Multi-version build

The CI workflow (`.github/workflows/documentation.yml`) uses
[sphinx-multiversion](https://github.com/sphinx-contrib/multiversion)
to build documentation for all versions listed in `_static/versions.json`,
with `-b dirhtml`.

sphinx-multiversion builds every version with the *current* checkout's
`conf.py` (it passes `-c` pointing at the working tree's `doc/`), while taking
each version's *source documents* from that version's git ref. So `conf.py`
changes and workflow changes reach `/stable/` on the next build, but changes to
source files under `doc/` only appear once a release tag carries them. `conf.py`
branches on `SPHINX_MULTIVERSION_NAME` to render each version correctly.

Because older versions' *source* is built with today's toolchain, CI must
install any build-time dependency an older version's source still needs (not its
`conf.py`, which is not used). When removing a version from `versions.json`,
drop any dependency it alone required from
`.github/workflows/documentation.yml`.

## Redirects

`scripts/gen_redirects.py` writes redirect stubs into the built site so that old
URLs keep working; the CI workflow runs it after the version builds. It has
three modes: `stubs` maps the legacy pre-`/stable/` layout URLs listed in
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
uv run licensecheck -r pyproject.toml --zero
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
