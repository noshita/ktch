"""Tests for scripts/colab_runtimes.py."""

import importlib.util
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Import the script module (not a package, so use importlib)
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "colab_runtimes.py"

spec = importlib.util.spec_from_file_location("colab_runtimes", _SCRIPT_PATH)
colab_runtimes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(colab_runtimes)

Runtime = colab_runtimes.Runtime

# ---------------------------------------------------------------------------
# Fixtures modelled on googlecolab/backend-info
# ---------------------------------------------------------------------------

COMMIT_A = "a" * 40
COMMIT_B = "b" * 40

README = f"""\
# Colab backend info

## Colab's Past Runtime Versions

Currently available past runtime versions are as follows:

### 2026.04

*   Python 3.12.13

Refer to the individual `*.txt` files at this
[GitHub commit](https://github.com/googlecolab/backend-info/tree/{COMMIT_A}).

### 2026.10

*   Python 3.13.16

Refer to the individual `*.txt` files at this
[GitHub commit](https://github.com/googlecolab/backend-info/tree/{COMMIT_B}).

### Notes

Not a runtime.
"""

OS_INFO = """\
#  $ (lsb_release -ds;python --version;) > os-info.txt
Ubuntu 24.04.5 LTS
Python 3.13.18
"""

FREEZE = """\
numpy==2.4.6
Pandas==2.2.3
scikit_learn==1.6.1
google-colab @ file:///colabtools/dist/google_colab-1.0.0.tar.gz
torch @ https://download.pytorch.org/whl/cpu/torch-2.14.1%2Bcpu-cp313-cp313-manylinux_2_28_x86_64.whl
"""


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_pinnable_reads_runtime_sections_newest_first():
    assert colab_runtimes.parse_pinnable(README) == [
        Runtime("2026.10", COMMIT_B),
        Runtime("2026.04", COMMIT_A),
    ]


def test_parse_pinnable_rejects_a_runtime_without_commit():
    with pytest.raises(ValueError, match="2026.07"):
        colab_runtimes.parse_pinnable("### 2026.07\n\n*   Python 3.12.13\n")


def test_parse_pinnable_rejects_a_readme_without_runtimes():
    with pytest.raises(ValueError, match="format"):
        colab_runtimes.parse_pinnable("# Colab backend info\n")


def test_python_version_skips_the_command_comment():
    assert colab_runtimes.python_version(OS_INFO) == "3.13.18"


def test_python_version_rejects_missing_version():
    with pytest.raises(ValueError):
        colab_runtimes.python_version("Ubuntu 24.04.5 LTS\n")


def test_parse_freeze_normalizes_names_and_skips_url_lines():
    assert colab_runtimes.parse_freeze(FREEZE) == {
        "numpy": "2.4.6",
        "pandas": "2.2.3",
        "scikit-learn": "1.6.1",
    }


def test_dependency_names_cover_the_colab_extras_once():
    pyproject = {
        "project": {
            "dependencies": ["numpy>=2.0", "pandas[pyarrow]>=3.0"],
            "optional-dependencies": {
                "data": ["pillow>=9.3", "pooch>=1.3"],
                "plot": ["matplotlib>=3.10", "NumPy"],
                "docs": ["sphinx"],
            },
        }
    }
    assert colab_runtimes.dependency_names(pyproject) == [
        "numpy",
        "pandas",
        "pillow",
        "pooch",
        "matplotlib",
    ]


# ---------------------------------------------------------------------------
# Selection and comparison
# ---------------------------------------------------------------------------

RUNTIMES = [
    Runtime("current", "main"),
    Runtime("2026.10", COMMIT_B),
    Runtime("2026.04", COMMIT_A),
]


def test_select_without_names_returns_every_runtime():
    assert colab_runtimes.select(RUNTIMES, []) == RUNTIMES


def test_select_resolves_oldest_and_named_runtimes():
    assert colab_runtimes.select(RUNTIMES, ["oldest", "current", "2026.10"]) == [
        RUNTIMES[2],
        RUNTIMES[0],
        RUNTIMES[1],
    ]


def test_select_rejects_unknown_runtime_and_lists_choices():
    with pytest.raises(ValueError, match="current, 2026.10, 2026.04, oldest"):
        colab_runtimes.select(RUNTIMES, ["2099.01"])


def test_compare_separates_moved_and_added_packages():
    freeze = {"numpy": "2.0.2", "pandas": "2.2.2"}
    resolved = {"numpy": "2.0.2", "pandas": "3.1.1", "narwhals": "2.28.0"}
    assert colab_runtimes.compare(resolved, freeze) == (
        {"pandas": ("2.2.2", "3.1.1")},
        ["narwhals"],
    )


# ---------------------------------------------------------------------------
# check, with the network and uv replaced
# ---------------------------------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    files = {
        ("main", "README.md"): README,
        ("main", "os-info.txt"): OS_INFO,
        ("main", "pip-freeze.txt"): "numpy==2.4.6\npandas==2.2.3\n",
        (COMMIT_B, "os-info.txt"): "Python 3.13.16\n",
        (COMMIT_B, "pip-freeze.txt"): "numpy==2.1.3\npandas==2.2.3\n",
        (COMMIT_A, "os-info.txt"): "Python 3.12.13\n",
        (COMMIT_A, "pip-freeze.txt"): "numpy==2.0.2\npandas==2.2.2\n",
    }
    monkeypatch.setattr(colab_runtimes, "fetch", lambda ref, path: files[ref, path])


def test_check_passes_when_the_resolution_keeps_every_version(
    offline, monkeypatch, capsys
):
    monkeypatch.setattr(colab_runtimes, "resolve", lambda freeze, python: dict(freeze))
    assert colab_runtimes.main(["check"]) == 0
    assert capsys.readouterr().out.count("changes nothing") == 3


def test_check_reports_a_moved_package_and_writes_the_preinstall_file(
    offline, monkeypatch, capsys, tmp_path
):
    def resolve(freeze, python):
        return {**freeze, "pandas": "3.1.1", "narwhals": "2.28.0"}

    monkeypatch.setattr(colab_runtimes, "resolve", resolve)
    preinstall = tmp_path / "preinstall.txt"
    assert (
        colab_runtimes.main(["check", "oldest", "--preinstall", str(preinstall)]) == 1
    )
    out = capsys.readouterr().out
    assert "2026.04 (Python 3.12.13): changes 1 package(s)" in out
    assert "pandas 2.2.2 -> 3.1.1" in out
    assert "adds narwhals" in out
    assert preinstall.read_text() == "numpy==2.0.2\npandas==2.2.2\n"


def test_check_refuses_preinstall_for_several_runtimes(offline, capsys):
    assert colab_runtimes.main(["check", "--preinstall", "x.txt"]) == 2
    assert "exactly one runtime" in capsys.readouterr().err
