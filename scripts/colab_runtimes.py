#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""List the Google Colab runtimes and check that installing ktch leaves them intact.

Colab publishes the Python version and the pip freeze of its runtimes in the
googlecolab/backend-info repository. The files on ``main`` describe the
current runtime. The README lists the past runtimes a notebook can pin, each
with a link to the commit that holds its files.

Installing ``ktch[data,plot]`` on any of these runtimes should leave every
installed package at its version. ``check`` resolves ktch's dependencies with
the runtime's freeze as the preferred versions and reports each package the
resolution moves. It exits 1 when one moves and 2 when the data cannot be read
or the dependencies cannot be resolved.

Usage:
    uv run scripts/colab_runtimes.py list [--json] [PACKAGE ...]
    uv run scripts/colab_runtimes.py check [RUNTIME ...] [--preinstall FILE]

RUNTIME is ``current``, ``oldest``, or a pinnable version such as ``2026.07``.
``check`` without a runtime checks every one of them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

_BASE_URL = "https://raw.githubusercontent.com/googlecolab/backend-info"
_ROOT = Path(__file__).resolve().parent.parent
# The extras the tutorials tell Colab users to install.
_EXTRAS = ("data", "plot")
# Colab runs Ubuntu on x86_64. The tag decides which wheels the resolver may pick.
_PLATFORM = "x86_64-manylinux_2_28"


class Runtime(NamedTuple):
    """A Colab runtime and the git ref that holds its files in backend-info."""

    name: str
    ref: str


def fetch(ref: str, path: str) -> str:
    url = f"{_BASE_URL}/{ref}/{path}"
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode()


def parse_pinnable(readme: str) -> list[Runtime]:
    """Return the pinnable past runtimes the README lists, newest first."""
    runtimes = []
    for section in re.split(r"^### ", readme, flags=re.MULTILINE)[1:]:
        heading, _, body = section.partition("\n")
        name = heading.strip()
        if not re.fullmatch(r"\d{4}\.\d{2}", name):
            continue
        commit = re.search(r"/tree/([0-9a-f]{40})", body)
        if commit is None:
            raise ValueError(f"runtime {name} has no commit link in the README")
        runtimes.append(Runtime(name, commit.group(1)))
    if not runtimes:
        raise ValueError("no pinnable runtime in the README; has its format changed?")
    return sorted(runtimes, key=lambda runtime: runtime.name, reverse=True)


def list_runtimes() -> list[Runtime]:
    """The current runtime first, then the pinnable ones from newest to oldest."""
    return [Runtime("current", "main"), *parse_pinnable(fetch("main", "README.md"))]


def select(runtimes: list[Runtime], names: list[str]) -> list[Runtime]:
    if not names:
        return runtimes
    by_name = {runtime.name: runtime for runtime in runtimes}
    selected = []
    for name in names:
        if name == "oldest":
            selected.append(runtimes[-1])
        elif name in by_name:
            selected.append(by_name[name])
        else:
            choices = ", ".join([*by_name, "oldest"])
            raise ValueError(f"unknown runtime {name!r}; choose from {choices}")
    return selected


def python_version(os_info: str) -> str:
    match = re.search(r"^Python (\d+\.\d+\.\d+)", os_info, flags=re.MULTILINE)
    if match is None:
        raise ValueError("os-info.txt names no Python version")
    return match.group(1)


def normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def parse_freeze(freeze: str) -> dict[str, str]:
    """Map each pinned distribution to its version.

    Lines that install from a URL or a local path carry no version to pin and
    are skipped.
    """
    pins = {}
    for line in freeze.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;@]+)", line.strip())
        if match:
            pins[normalize(match.group(1))] = match.group(2)
    return pins


def dependency_names(pyproject: dict) -> list[str]:
    """ktch's direct dependencies, including those of the extras Colab users install."""
    project = pyproject["project"]
    requirements = list(project["dependencies"])
    for extra in _EXTRAS:
        requirements += project["optional-dependencies"][extra]
    names = []
    for requirement in requirements:
        name = normalize(re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement).group(0))
        if name not in names:
            names.append(name)
    return names


def resolve(freeze: dict[str, str], python: str) -> dict[str, str]:
    """Resolve ktch's dependencies, preferring the versions in ``freeze``.

    uv keeps a version already in its output file whenever the requirements
    allow it, as pip leaves an installed package alone when it satisfies them.
    """
    uv = os.environ.get("UV", "uv")
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "resolved.txt"
        output.write_text(
            "".join(f"{name}=={version}\n" for name, version in freeze.items())
        )
        subprocess.run(
            [
                uv,
                "pip",
                "compile",
                str(_ROOT / "pyproject.toml"),
                *(f"--extra={extra}" for extra in _EXTRAS),
                f"--python-version={python}",
                f"--python-platform={_PLATFORM}",
                "--no-header",
                "--no-annotate",
                "--quiet",
                f"--output-file={output}",
            ],
            check=True,
        )
        return parse_freeze(output.read_text())


def compare(
    resolved: dict[str, str], freeze: dict[str, str]
) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """Split the resolution into packages it moves and packages it adds."""
    changed = {
        name: (freeze[name], version)
        for name, version in resolved.items()
        if name in freeze and freeze[name] != version
    }
    added = sorted(name for name in resolved if name not in freeze)
    return changed, added


def _list(args: argparse.Namespace) -> int:
    pyproject = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    packages = [normalize(name) for name in args.packages] or dependency_names(
        pyproject
    )
    rows = []
    for runtime in list_runtimes():
        freeze = parse_freeze(fetch(runtime.ref, "pip-freeze.txt"))
        rows.append(
            {
                "runtime": runtime.name,
                "ref": runtime.ref,
                "python": python_version(fetch(runtime.ref, "os-info.txt")),
                "packages": {name: freeze.get(name) for name in packages},
            }
        )
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    header = ["runtime", "python", *packages]
    table = [
        [
            row["runtime"],
            row["python"],
            *(row["packages"][name] or "-" for name in packages),
        ]
        for row in rows
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *table)]
    for line in [header, *table]:
        print(
            "  ".join(cell.ljust(width) for cell, width in zip(line, widths)).rstrip()
        )
    return 0


def _check(args: argparse.Namespace) -> int:
    runtimes = select(list_runtimes(), args.runtimes)
    if args.preinstall and len(runtimes) != 1:
        raise ValueError("--preinstall takes exactly one runtime")
    moved = False
    for runtime in runtimes:
        python = python_version(fetch(runtime.ref, "os-info.txt"))
        freeze = parse_freeze(fetch(runtime.ref, "pip-freeze.txt"))
        resolved = resolve(freeze, python)
        changed, added = compare(resolved, freeze)
        verdict = f"changes {len(changed)} package(s)" if changed else "changes nothing"
        print(f"{runtime.name} (Python {python}): {verdict}")
        for name, (old, new) in sorted(changed.items()):
            print(f"  {name} {old} -> {new}")
        if added:
            print(f"  adds {', '.join(added)}")
        moved = moved or bool(changed)
        if args.preinstall:
            # What the runtime already holds of everything ktch pulls in.
            args.preinstall.write_text(
                "".join(
                    f"{name}=={freeze[name]}\n"
                    for name in sorted(resolved)
                    if name in freeze
                )
            )
    return 1 if moved else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser(
        "list", help="show each runtime's Python and ktch's dependency versions"
    )
    list_parser.add_argument(
        "packages", nargs="*", metavar="PACKAGE", help="packages to show instead"
    )
    list_parser.add_argument("--json", action="store_true", help="print JSON")
    list_parser.set_defaults(func=_list)

    check_parser = commands.add_parser(
        "check", help="report the packages installing ktch[data,plot] would change"
    )
    check_parser.add_argument("runtimes", nargs="*", metavar="RUNTIME")
    check_parser.add_argument(
        "--preinstall",
        type=Path,
        metavar="FILE",
        help="write the runtime's versions of the packages ktch pulls in",
    )
    check_parser.set_defaults(func=_check)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (urllib.error.URLError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
