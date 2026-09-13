#!/usr/bin/env python3
"""Date sitemap entries from the git history of the page each one was built from.

Google ignores ``<priority>`` and ``<changefreq>`` and uses ``<lastmod>`` only
when it is, in its own words, "consistently and verifiably accurate". A single
build or release date across every URL fails that test as soon as one page in
the sitemap has not changed, which is why the dates here come per page from the
last commit that touched its source.

``sphinx-last-updated-by-git`` does this inside Sphinx and cannot be used:
sphinx-multiversion materialises each ref with ``git archive`` into a plain
directory, so every git lookup inside the build fails with "not a git
repository". This script runs after the build instead, against the real clone.

Pages whose source is not in the repository, the generated API reference among
them, get no ``<lastmod>``. The field is optional, and omitting it says less
than guessing would.
"""

from __future__ import annotations

import argparse
import subprocess  # nosec B404
import sys
import xml.etree.ElementTree as ElementTree
from pathlib import PurePosixPath

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
SOURCE_SUFFIXES = (".md", ".rst", ".ipynb")


def git(*args: str) -> str:
    """Run git and return stdout, or an empty string when it fails."""
    result = subprocess.run(  # nosec B603 B607
        ["git", *args], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


def sources_at(ref: str, doc_dir: str) -> set[str]:
    listing = git("ls-tree", "-r", "--name-only", ref, "--", doc_dir)
    return set(listing.splitlines())


def source_for(page: str, doc_dir: str, tracked: set[str]) -> str | None:
    """Map a dirhtml page path onto the source file that produced it."""
    stem = page.strip("/")
    base = PurePosixPath(doc_dir)
    candidates = []
    for suffix in SOURCE_SUFFIXES:
        if stem:
            candidates.append(str(base / f"{stem}{suffix}"))
            candidates.append(str(base / stem / f"index{suffix}"))
        else:
            candidates.append(str(base / f"index{suffix}"))
    for candidate in candidates:
        if candidate in tracked:
            return candidate
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sitemap", required=True, help="sitemap.xml to rewrite")
    parser.add_argument("--ref", required=True, help="git ref the site was built from")
    parser.add_argument(
        "--base-url", required=True, help="URL prefix the sitemap entries carry"
    )
    parser.add_argument("--doc-dir", default="doc", help="documentation source root")
    args = parser.parse_args(argv)

    if not git("rev-parse", "--verify", f"{args.ref}^{{commit}}"):
        print(f"warning: ref {args.ref} not found; sitemap left without lastmod")
        return 0

    tracked = sources_at(args.ref, args.doc_dir)
    if not tracked:
        print(f"warning: no sources under {args.doc_dir} at {args.ref}")
        return 0

    ElementTree.register_namespace("", SITEMAP_NS)
    tree = ElementTree.parse(args.sitemap)
    base = args.base_url if args.base_url.endswith("/") else args.base_url + "/"

    dated = 0
    undated = []
    dates: dict[str, str] = {}
    for url in tree.getroot().findall(f"{{{SITEMAP_NS}}}url"):
        loc = url.find(f"{{{SITEMAP_NS}}}loc")
        if loc is None or not loc.text or not loc.text.startswith(base):
            continue
        page = loc.text[len(base) :]
        source = source_for(page, args.doc_dir, tracked)
        if source is None:
            undated.append(page or "/")
            continue
        if source not in dates:
            dates[source] = git("log", "-1", "--format=%cI", args.ref, "--", source)
        if not dates[source]:
            undated.append(page or "/")
            continue
        lastmod = ElementTree.Element(f"{{{SITEMAP_NS}}}lastmod")
        lastmod.text = dates[source]
        url.insert(list(url).index(loc) + 1, lastmod)
        dated += 1

    tree.write(args.sitemap, encoding="utf-8", xml_declaration=True)
    print(f"stamped {dated} sitemap entries from {args.ref}; {len(undated)} left bare")
    return 0


if __name__ == "__main__":
    sys.exit(main())
