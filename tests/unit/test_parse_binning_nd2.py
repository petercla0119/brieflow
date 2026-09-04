"""Unit tests for _parse_binning_from_nd2 (metadata-only ND2 binning parse).

The fix reads the ND2 text-info block via nd2.ND2File instead of nd2.imread,
which would materialize the full multi-GB pixel array. These tests pin the
parsing behavior and, critically, assert nd2.imread is never called.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nd2  # noqa: E402
from workflow.lib.preprocess.preprocess import _parse_binning_from_nd2  # noqa: E402


class FakeND2File:
    """Minimal nd2.ND2File stub: context manager exposing .text_info."""

    def __init__(self, text_info):
        self._text_info = text_info
        self.entered = False
        self.exited = False

    @property
    def text_info(self):
        return self._text_info

    def __enter__(self):
        self.entered = True
        return self

    def __exit__(self, *args):
        self.exited = True
        return False


@pytest.fixture(autouse=True)
def _forbid_imread(monkeypatch):
    """Every test: calling nd2.imread is a hard failure (would decode pixels)."""

    def _boom(*a, **k):
        raise AssertionError("nd2.imread called -- pixels must not be materialized")

    monkeypatch.setattr(nd2, "imread", _boom)


@pytest.mark.unit
@pytest.mark.parametrize(
    "desc,expected",
    [
        ("Binning: 2x2", "2x2"),
        ("prefix Binning: 1x1 suffix", "1x1"),
        ("Binning:  4x4", "4x4"),  # extra spaces
        ("multi\nline\nBinning: 3x3\nmore", "3x3"),
        ("no binning field here", None),
        ("", None),
    ],
)
def test_binning_parse(monkeypatch, desc, expected):
    fake = FakeND2File({"description": desc})
    monkeypatch.setattr(nd2, "ND2File", lambda *a, **k: fake)
    assert _parse_binning_from_nd2("dummy.nd2") == expected
    assert fake.entered and fake.exited, "ND2File not used as a context manager"


@pytest.mark.unit
def test_text_info_none(monkeypatch):
    monkeypatch.setattr(nd2, "ND2File", lambda *a, **k: FakeND2File(None))
    assert _parse_binning_from_nd2("dummy.nd2") is None


@pytest.mark.unit
def test_text_info_not_dict(monkeypatch):
    monkeypatch.setattr(nd2, "ND2File", lambda *a, **k: FakeND2File("not-a-dict"))
    assert _parse_binning_from_nd2("dummy.nd2") is None


@pytest.mark.unit
def test_nd2file_raises_returns_none(monkeypatch):
    def _raise(*a, **k):
        raise OSError("cannot open")

    monkeypatch.setattr(nd2, "ND2File", _raise)
    assert _parse_binning_from_nd2("dummy.nd2") is None


@pytest.mark.unit
def test_never_calls_imread(monkeypatch):
    """Critical gate: the whole point of the fix is to avoid nd2.imread."""
    fake = FakeND2File({"description": "Binning: 2x2"})
    monkeypatch.setattr(nd2, "ND2File", lambda *a, **k: fake)
    # _forbid_imread autouse fixture already makes imread explode; a regression
    # back to nd2.imread would raise AssertionError here instead of returning.
    assert _parse_binning_from_nd2("dummy.nd2") == "2x2"
