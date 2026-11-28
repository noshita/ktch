"""Tests for scripts/assemble_site.py."""

import importlib.util
import io
import json
import subprocess
import tarfile
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_SCRIPT_PATH = _SCRIPTS_DIR / "assemble_site.py"

spec = importlib.util.spec_from_file_location("assemble_site", _SCRIPT_PATH)
assemble_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assemble_site)

BASE = "https://doc.ktch.dev"


def _page(version, path=""):
    url = f"{BASE}/{version}/{path}"
    return f'<html><head><link rel="canonical" href="{url}"></head></html>'


def _sitemap(version, pages):
    urls = "".join(f"<url><loc>{BASE}/{version}/{p}</loc></url>" for p in pages)
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        f'<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{urls}</urlset>'
    )


def make_build(path, version, sha="0" * 40, files=None, manifest_version=None):
    """Write a build tarball as docs-build.yml packages it."""
    if files is None:
        files = {
            "index.html": _page(version),
            "guide/index.html": _page(version, "guide/"),
            "sitemap.xml": _sitemap(version, ["", "guide/"]),
        }
    manifest = {
        "version": manifest_version or version,
        "ref": f"v{version}",
        "sha": sha,
    }
    with tarfile.open(path, "w:gz") as tar:
        members = {"manifest.json": json.dumps(manifest)}
        members.update({f"html/{name}": text for name, text in files.items()})
        for name, text in members.items():
            data = text.encode()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return path


def write_versions(path, entries):
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


STABLE_AND_DEV = [
    {"version": "0.2.0", "preferred": True},
    {"version": "0.1.0"},
    {"version": "dev"},
]


def test_read_versions_returns_released_and_preferred(tmp_path):
    path = write_versions(tmp_path / "versions.json", STABLE_AND_DEV)
    assert assemble_site.read_versions(path) == (["0.2.0", "0.1.0"], "0.2.0")


@pytest.mark.parametrize(
    "entries",
    [
        [{"version": "0.1.0"}, {"version": "dev"}],
        [
            {"version": "0.1.0", "preferred": True},
            {"version": "0.2.0", "preferred": True},
        ],
        [{"version": "0.1.0"}, {"version": "dev", "preferred": True}],
    ],
    ids=["none", "two", "dev"],
)
def test_read_versions_requires_one_released_preferred(tmp_path, entries):
    path = write_versions(tmp_path / "versions.json", entries)
    with pytest.raises(SystemExit, match="exactly one released version"):
        assemble_site.read_versions(path)


def test_unpack_places_the_tree_and_returns_the_manifest(tmp_path):
    tarball = make_build(tmp_path / "b.tar.gz", "0.1.0", sha="a" * 40)
    manifest = assemble_site.unpack(tarball, "0.1.0", tmp_path / "0.1.0")
    assert manifest["sha"] == "a" * 40
    assert (tmp_path / "0.1.0" / "guide" / "index.html").is_file()
    assert not (tmp_path / "0.1.0" / "manifest.json").exists()


def test_unpack_fails_without_the_tarball(tmp_path):
    with pytest.raises(SystemExit, match="no stored build of 0.1.0"):
        assemble_site.unpack(tmp_path / "missing.tar.gz", "0.1.0", tmp_path / "out")


def test_unpack_fails_on_another_version(tmp_path):
    tarball = make_build(tmp_path / "b.tar.gz", "0.1.0", manifest_version="0.1.1")
    with pytest.raises(SystemExit, match="holds version '0.1.1'"):
        assemble_site.unpack(tarball, "0.1.0", tmp_path / "out")


def test_unpack_fails_without_index(tmp_path):
    tarball = make_build(tmp_path / "b.tar.gz", "0.1.0", files={"a.html": ""})
    with pytest.raises(SystemExit, match="no html/index.html"):
        assemble_site.unpack(tarball, "0.1.0", tmp_path / "out")


def test_unpack_rejects_paths_outside_the_tree(tmp_path):
    tarball = make_build(
        tmp_path / "b.tar.gz", "0.1.0", files={"index.html": "", "../../evil": ""}
    )
    with pytest.raises(tarfile.OutsideDestinationError):
        assemble_site.unpack(tarball, "0.1.0", tmp_path / "out")


def test_make_stable_rewrites_absolute_urls(tmp_path):
    (tmp_path / "0.2.0" / "_static").mkdir(parents=True)
    (tmp_path / "0.2.0" / "index.html").write_text(_page("0.2.0"))
    (tmp_path / "0.2.0" / "sitemap.xml").write_text(_sitemap("0.2.0", [""]))
    (tmp_path / "0.2.0" / "_static" / "opensearch.xml").write_text(f"{BASE}/0.2.0/")
    (tmp_path / "0.2.0" / "_static" / "other.js").write_text(f"{BASE}/0.2.0/")

    assert assemble_site.make_stable(tmp_path, "0.2.0", BASE) == 3

    stable = tmp_path / "stable"
    assert f"{BASE}/stable/" in (stable / "index.html").read_text()
    assert f"{BASE}/stable/" in (stable / "sitemap.xml").read_text()
    assert f"{BASE}/stable/" in (stable / "_static" / "opensearch.xml").read_text()
    assert f"{BASE}/0.2.0/" in (stable / "_static" / "other.js").read_text()
    assert f"{BASE}/0.2.0/" in (tmp_path / "0.2.0" / "index.html").read_text()


def test_make_stable_fails_when_nothing_is_rewritten(tmp_path):
    (tmp_path / "0.2.0").mkdir()
    (tmp_path / "0.2.0" / "index.html").write_text("<html></html>")
    with pytest.raises(SystemExit, match="no https://doc.ktch.dev/0.2.0/ URL"):
        assemble_site.make_stable(tmp_path, "0.2.0", BASE)


def _git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A clone holding the sources the stable version was built from."""
    repo = tmp_path / "repo"
    (repo / "doc" / "guide").mkdir(parents=True)
    (repo / "doc" / "index.md").write_text("# Home\n")
    (repo / "doc" / "guide" / "index.md").write_text("# Guide\n")
    (repo / "doc" / "robots.txt").write_text("User-agent: *\n")
    (repo / "doc" / "_redirects.toml").write_text(
        '[redirects]\n"/old/guide.html" = "guide/index"\n'
    )
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c", "user.name=t", "-c", "user.email=t@example.invalid",
        "commit", "-q", "-m", "docs",
    )  # fmt: skip
    return repo


def _run(repo, tmp_path, monkeypatch, versions=STABLE_AND_DEV, stored=None, sha=None):
    """Store builds of ``stored`` (default: every released version), then assemble."""
    sha = sha or _git(repo, "rev-parse", "HEAD")
    if stored is None:
        stored = [e["version"] for e in versions if e["version"] != "dev"]
    assets = tmp_path / "assets"
    assets.mkdir()
    for version in stored:
        name = assemble_site.ASSET_NAME.format(version=version)
        make_build(assets / name, version, sha=sha)
    dev = make_build(tmp_path / "dev.tar.gz", "dev")
    write_versions(repo / "doc" / "versions.json", versions)
    monkeypatch.chdir(repo)
    out = tmp_path / "site"
    assemble_site.main(
        [
            "--dev", str(dev), "--assets-dir", str(assets), "--out", str(out),
            "--versions", "doc/versions.json",
        ]
    )  # fmt: skip
    return out


def test_main_assembles_the_site(repo, tmp_path, monkeypatch):
    out = _run(repo, tmp_path, monkeypatch)

    for directory in ("0.2.0", "0.1.0", "dev", "stable"):
        assert (out / directory / "guide" / "index.html").is_file()
        assert (out / directory / "guide.html").is_file()
    assert (
        f"{BASE}/stable/guide/" in (out / "stable" / "guide" / "index.html").read_text()
    )
    assert f"{BASE}/stable/guide/" in (out / "stable" / "guide.html").read_text()
    assert "<lastmod>" in (out / "stable" / "sitemap.xml").read_text()
    assert "<lastmod>" not in (out / "0.2.0" / "sitemap.xml").read_text()
    assert (out / "robots.txt").read_text() == "User-agent: *\n"
    assert 'url=stable/"' in (out / "index.html").read_text()
    assert "/stable/guide/" in (out / "old" / "guide.html").read_text()
    assert (out / "CNAME").read_text() == "doc.ktch.dev\n"


def test_main_fails_without_a_stored_version(repo, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="no stored build of 0.1.0"):
        _run(repo, tmp_path, monkeypatch, stored=["0.2.0"])


def test_main_fails_when_the_stable_commit_is_unknown(repo, tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="dating the stable sitemap failed"):
        _run(repo, tmp_path, monkeypatch, sha="f" * 40)


def test_main_fails_on_a_stale_redirect_map(repo, tmp_path, monkeypatch):
    (repo / "doc" / "_redirects.toml").write_text(
        '[redirects]\n"/old/gone.html" = "gone"\n'
    )
    with pytest.raises(SystemExit, match="target docnames absent"):
        _run(repo, tmp_path, monkeypatch)
