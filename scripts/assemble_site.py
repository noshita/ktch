#!/usr/bin/env python3
"""Assemble the documentation site from the stored versions and a dev build.

The release assets are downloaded beforehand. Run from the root of a full clone:
the stable sitemap is dated from git history.

Usage:
    python scripts/assemble_site.py --dev dev.tar.gz --assets-dir "$ASSETS" --out site
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from types import ModuleType

_SCRIPTS_DIR = Path(__file__).resolve().parent
ASSET_NAME = "ktch-docs-html-{version}.tar.gz"

ROOT_INDEX = """<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8">
    <meta http-equiv="refresh" content="0; url=stable/">
    <link rel="canonical" href="{base_url}/stable/">
    <title>Redirecting to stable documentation</title>
  </head>
  <body>
    <p>Redirecting to <a href="stable/">stable documentation</a>...</p>
  </body>
</html>
"""


def _load(name: str) -> ModuleType:
    """Import a sibling script, which is not part of a package."""
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_versions(path: Path) -> tuple[list[str], str]:
    """Return the released versions ``versions.json`` names, and the preferred one."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    released = [e["version"] for e in entries if e.get("version") not in (None, "dev")]
    preferred = [e["version"] for e in entries if e.get("preferred")]
    if len(preferred) != 1 or preferred[0] not in released:
        raise SystemExit(
            f"assemble_site: {path} must mark exactly one released version as "
            f"preferred, found {preferred}"
        )
    return released, preferred[0]


def unpack(tarball: Path, version: str, dest: Path) -> dict:
    """Extract one built version's tree to ``dest`` and return its manifest."""
    if not tarball.is_file():
        raise SystemExit(f"assemble_site: no stored build of {version}: {tarball}")
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        with tarfile.open(tarball) as tar:
            tar.extractall(tmp, filter="data")
        manifest = json.loads((Path(tmp) / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("version") != version:
            raise SystemExit(
                f"assemble_site: {tarball} holds version "
                f"{manifest.get('version')!r}, expected {version!r}"
            )
        tree = Path(tmp) / "html"
        if not (tree / "index.html").is_file():
            raise SystemExit(f"assemble_site: {tarball} has no html/index.html")
        shutil.move(tree, dest)
    return manifest


def make_stable(site: Path, version: str, base_url: str) -> int:
    """Copy ``version`` to ``stable/`` and point its absolute URLs there."""
    stable = site / "stable"
    shutil.copytree(site / version, stable)
    old = f"{base_url}/{version}/"
    new = f"{base_url}/stable/"
    count = 0
    for path in stable.rglob("*"):
        if path.suffix != ".html" and path.name not in (
            "sitemap.xml",
            "opensearch.xml",
        ):
            continue
        text = path.read_text(encoding="utf-8")
        found = text.count(old)
        if found:
            path.write_text(text.replace(old, new), encoding="utf-8")
            count += found
    if count == 0:
        raise SystemExit(f"assemble_site: no {old} URL found in the stable copy")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dev", type=Path, required=True, help="dev build tarball")
    parser.add_argument(
        "--assets-dir",
        type=Path,
        required=True,
        help=f"directory holding one {ASSET_NAME} per released version",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="site directory to create"
    )
    parser.add_argument(
        "--versions", type=Path, default=Path("doc/_static/versions.json")
    )
    parser.add_argument("--robots", type=Path, default=Path("doc/robots.txt"))
    parser.add_argument(
        "--redirect-map", type=Path, default=Path("doc/_redirects.toml")
    )
    parser.add_argument("--doc-dir", default="doc", help="documentation source root")
    parser.add_argument("--base-url", default="https://doc.ktch.dev")
    args = parser.parse_args(argv)
    base_url = args.base_url.rstrip("/")

    released, preferred = read_versions(args.versions)
    args.out.mkdir(parents=True)

    manifests = {}
    for version in released:
        tarball = args.assets_dir / ASSET_NAME.format(version=version)
        manifests[version] = unpack(tarball, version, args.out / version)
        print(f"assemble_site: {version} from {manifests[version]['ref']}")
    unpack(args.dev, "dev", args.out / "dev")

    gen_redirects = _load("gen_redirects")
    aliases = gen_redirects.emit_html_aliases(args.out, base_url)
    print(f"assemble_site: wrote {aliases} .html aliases")

    rewritten = make_stable(args.out, preferred, base_url)
    print(f"assemble_site: stable/ is {preferred}, {rewritten} URLs rewritten")

    stamp = _load("stamp_sitemap_lastmod")
    if stamp.main(
        [
            "--sitemap",
            str(args.out / "stable" / "sitemap.xml"),
            "--ref",
            manifests[preferred]["sha"],
            "--base-url",
            f"{base_url}/stable/",
            "--doc-dir",
            args.doc_dir,
        ]
    ):
        raise SystemExit("assemble_site: dating the stable sitemap failed")

    shutil.copyfile(args.robots, args.out / "robots.txt")
    (args.out / "index.html").write_text(
        ROOT_INDEX.format(base_url=base_url), encoding="utf-8"
    )
    gen_redirects.main(
        [
            "--map",
            str(args.redirect_map),
            "--site-dir",
            str(args.out),
            "--base-url",
            base_url,
        ]
    )
    (args.out / "CNAME").write_text(
        base_url.split("://", 1)[1] + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
