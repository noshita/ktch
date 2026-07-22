"""Tests for scripts/gen_redirects.py."""

import importlib.util
import tomllib
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the script module (not a package, so use importlib)
# ---------------------------------------------------------------------------

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
_SCRIPT_PATH = _SCRIPTS_DIR / "gen_redirects.py"

spec = importlib.util.spec_from_file_location("gen_redirects", _SCRIPT_PATH)
gen_redirects = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen_redirects)

resolve = gen_redirects.resolve
plan = gen_redirects.plan
emit_stubs = gen_redirects.emit_stubs
emit_cloudflare = gen_redirects.emit_cloudflare
emit_html_aliases = gen_redirects.emit_html_aliases


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BUILDERS = ["html", "dirhtml"]


def _make_site(root, docnames, builder="html"):
    """Create a minimal built site holding `docnames` under a stable/ directory.

    Mirrors how each Sphinx builder lays out its output: both write a directory
    index to ``<name>/index.html``, and they differ only for leaf pages.
    """
    stable = root / "stable"
    for doc in docnames:
        if doc == "index" or doc.endswith("/index"):
            path = stable / f"{doc}.html"
        elif builder == "dirhtml":
            path = stable / doc / "index.html"
        else:
            path = stable / f"{doc}.html"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("page", encoding="utf-8")
    return stable


# ---------------------------------------------------------------------------
# resolve()
# ---------------------------------------------------------------------------


class TestResolve:
    def test_html_builder_keeps_the_extension_on_leaf_pages(self, tmp_path):
        stable = _make_site(tmp_path, ["api/datasets"], builder="html")
        assert resolve("api/datasets", stable, "/stable") == "/stable/api/datasets.html"

    def test_dirhtml_builder_uses_a_directory_url_on_leaf_pages(self, tmp_path):
        stable = _make_site(tmp_path, ["api/datasets"], builder="dirhtml")
        assert resolve("api/datasets", stable, "/stable") == "/stable/api/datasets/"

    @pytest.mark.parametrize("builder", BUILDERS)
    def test_root_index_resolves_to_the_version_root(self, tmp_path, builder):
        stable = _make_site(tmp_path, ["index"], builder=builder)
        assert resolve("index", stable, "/stable") == "/stable/"

    @pytest.mark.parametrize("builder", BUILDERS)
    def test_nested_index_resolves_to_its_directory(self, tmp_path, builder):
        stable = _make_site(tmp_path, ["api/index"], builder=builder)
        assert resolve("api/index", stable, "/stable") == "/stable/api/"

    def test_missing_docname_raises(self, tmp_path):
        stable = _make_site(tmp_path, ["api/datasets"])
        with pytest.raises(LookupError):
            resolve("api/gone", stable, "/stable")


# ---------------------------------------------------------------------------
# plan()
# ---------------------------------------------------------------------------


class TestPlan:
    def test_resolves_every_entry(self, tmp_path):
        _make_site(tmp_path, ["api/datasets", "api/index"])
        resolved = plan(
            {"/old/data.html": "api/datasets", "/old/index.html": "api/index"},
            tmp_path,
            "stable",
        )
        assert dict(resolved) == {
            "/old/data.html": "/stable/api/datasets.html",
            "/old/index.html": "/stable/api/",
        }

    def test_missing_target_fails_the_build(self, tmp_path):
        _make_site(tmp_path, ["api/datasets"])
        with pytest.raises(SystemExit) as excinfo:
            plan({"/old/data.html": "api/renamed"}, tmp_path, "stable")
        assert "api/renamed" in str(excinfo.value)

    def test_source_that_would_overwrite_a_real_page_fails_the_build(self, tmp_path):
        _make_site(tmp_path, ["index"])
        (tmp_path / "index.html").write_text("root redirect", encoding="utf-8")
        with pytest.raises(SystemExit) as excinfo:
            plan({"/index.html": "index"}, tmp_path, "stable")
        assert "overwrite" in str(excinfo.value)

    def test_rerunning_over_its_own_output_is_allowed(self, tmp_path):
        # Generating twice into the same tree must not report the first run's
        # stubs as real pages, or a local rebuild loop breaks on the second pass.
        _make_site(tmp_path, ["api/datasets"])
        mapping = {"/old/data.html": "api/datasets"}
        emit_stubs(plan(mapping, tmp_path, "stable"), tmp_path, "https://doc.ktch.dev")
        assert plan(mapping, tmp_path, "stable") == [
            ("/old/data.html", "/stable/api/datasets.html")
        ]

    def test_relative_source_is_rejected(self, tmp_path):
        _make_site(tmp_path, ["index"])
        with pytest.raises(SystemExit):
            plan({"old/index.html": "index"}, tmp_path, "stable")

    def test_missing_version_directory_fails(self, tmp_path):
        with pytest.raises(SystemExit):
            plan({"/old.html": "index"}, tmp_path, "stable")


# ---------------------------------------------------------------------------
# emit_stubs() / emit_cloudflare()
# ---------------------------------------------------------------------------


class TestEmit:
    def test_stub_carries_a_meta_refresh_and_a_canonical_link(self, tmp_path):
        emit_stubs(
            [("/notebooks/index.html", "/stable/tutorials/")],
            tmp_path,
            "https://doc.ktch.dev",
        )
        stub = (tmp_path / "notebooks/index.html").read_text(encoding="utf-8")
        assert 'http-equiv="refresh"' in stub
        assert "0; url=/stable/tutorials/" in stub
        assert 'rel="canonical" href="https://doc.ktch.dev/stable/tutorials/"' in stub

    def test_stub_is_never_noindex(self, tmp_path):
        # A redirect stub exists to hand the old URL's ranking signals to the new
        # one; `noindex` would sever exactly that, so guard against it.
        emit_stubs([("/old.html", "/stable/")], tmp_path, "https://doc.ktch.dev")
        assert "noindex" not in (tmp_path / "old.html").read_text(encoding="utf-8")

    def test_cloudflare_emits_301_rules(self, tmp_path):
        emit_cloudflare(
            [("/a.html", "/stable/a/"), ("/b.html", "/stable/b/")], tmp_path
        )
        lines = (tmp_path / "_redirects").read_text(encoding="utf-8").splitlines()
        assert lines == ["/a.html  /stable/a/  301", "/b.html  /stable/b/  301"]


# ---------------------------------------------------------------------------
# emit_html_aliases() -- keep old foo.html URLs alive after html -> dirhtml
# ---------------------------------------------------------------------------


def _make_dirhtml_version(root, version, docnames):
    """Lay out a dirhtml version directory: each docname as <docname>/index.html."""
    vdir = root / version
    for doc in docnames:
        page = vdir / doc / "index.html" if doc else vdir / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text("page", encoding="utf-8")
    return vdir


class TestHtmlAliases:
    def test_leaf_page_gets_an_alias_to_its_directory_url(self, tmp_path):
        _make_dirhtml_version(tmp_path, "stable", ["installation"])
        emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        alias = tmp_path / "stable/installation.html"
        assert alias.exists()
        text = alias.read_text(encoding="utf-8")
        # Relative refresh (version-independent) + absolute canonical.
        assert "0; url=installation/" in text
        assert (
            'rel="canonical" href="https://doc.ktch.dev/stable/installation/"' in text
        )

    def test_nested_leaf_alias_is_written_next_to_its_directory(self, tmp_path):
        _make_dirhtml_version(tmp_path, "stable", ["tutorials/harmonic/spharm"])
        emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        alias = tmp_path / "stable/tutorials/harmonic/spharm.html"
        assert alias.exists()
        # Refresh target is the last segment only, so it resolves relative to
        # the alias's own directory.
        assert "0; url=spharm/" in alias.read_text(encoding="utf-8")

    def test_version_root_gets_no_alias(self, tmp_path):
        _make_dirhtml_version(tmp_path, "stable", [""])
        emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        assert not (tmp_path / "stable.html").exists()

    def test_asset_directories_are_skipped(self, tmp_path):
        _make_dirhtml_version(tmp_path, "stable", ["_modules/ktch/io/_tps"])
        emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        assert not (tmp_path / "stable/_modules/ktch/io/_tps.html").exists()

    def test_every_version_directory_is_processed(self, tmp_path):
        _make_dirhtml_version(tmp_path, "stable", ["installation"])
        _make_dirhtml_version(tmp_path, "dev", ["installation"])
        written = emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        assert (tmp_path / "stable/installation.html").exists()
        assert (tmp_path / "dev/installation.html").exists()
        assert written == 2

    def test_canonical_carries_the_version_prefix(self, tmp_path):
        # The workflow rewrites <version>/ -> stable/ in the copied stable dir,
        # so the alias canonical must carry the version prefix to be rewritten.
        _make_dirhtml_version(tmp_path, "0.10.1", ["installation"])
        emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        text = (tmp_path / "0.10.1/installation.html").read_text(encoding="utf-8")
        assert "https://doc.ktch.dev/0.10.1/installation/" in text

    def test_aliases_are_never_noindex(self, tmp_path):
        _make_dirhtml_version(tmp_path, "stable", ["installation"])
        emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        text = (tmp_path / "stable/installation.html").read_text(encoding="utf-8")
        assert "noindex" not in text

    def test_does_not_overwrite_a_real_html_file(self, tmp_path):
        vdir = _make_dirhtml_version(tmp_path, "stable", ["guide"])
        real = vdir / "guide.html"
        real.write_text("REAL PAGE", encoding="utf-8")
        emit_html_aliases(tmp_path, "https://doc.ktch.dev")
        assert real.read_text(encoding="utf-8") == "REAL PAGE"


# ---------------------------------------------------------------------------
# The map that actually ships
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mapping():
    path = _PROJECT_ROOT / "doc" / "_redirects.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))["redirects"]


class TestShippedMap:
    def test_sources_are_site_root_relative(self, mapping):
        assert all(source.startswith("/") for source in mapping)

    def test_targets_are_docnames_without_an_extension(self, mapping):
        # Storing docnames rather than URLs is what keeps the map valid across a
        # builder change; an extension here would silently pin it to one builder.
        assert not [target for target in mapping.values() if target.endswith(".html")]

    def test_root_index_is_not_redirected(self, mapping):
        # The site root already serves its own redirect to the stable version.
        assert "/index.html" not in mapping
