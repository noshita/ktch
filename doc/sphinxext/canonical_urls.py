"""Sphinx extension to emit clean directory-style canonical URLs.

Sphinx derives the ``<link rel="canonical">`` tag (and the ``pageurl``
context variable) from ``html_baseurl`` plus the page's target URI, which
yields ``.../index.html`` for directory-index pages.  Search engines prefer
the clean directory form (``.../``) and treat ``.../`` and ``.../index.html``
as separate URLs.  When the two forms disagree across the site (for example a
redirect that points at ``.../stable/`` while the target page canonicalizes to
``.../stable/index.html``), the canonical signal is diluted.

This extension strips a trailing ``index.html`` from ``pageurl`` so that the
canonical tag uses the directory form consistently.  Non-index pages (for
example ``installation.html``) are left untouched.
"""

from __future__ import annotations

from typing import Any

from sphinx.application import Sphinx

_INDEX_SUFFIX = "index.html"


def _clean_canonical(
    app: Sphinx,
    pagename: str,
    templatename: str,
    context: dict[str, Any],
    doctree: Any,
) -> None:
    """``html-page-context`` callback — drop ``index.html`` from ``pageurl``."""
    pageurl = context.get("pageurl")
    if pageurl and pageurl.endswith("/" + _INDEX_SUFFIX):
        context["pageurl"] = pageurl[: -len(_INDEX_SUFFIX)]


def setup(app: Sphinx) -> dict[str, Any]:
    app.connect("html-page-context", _clean_canonical)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
