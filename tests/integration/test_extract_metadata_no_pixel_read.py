"""Integration guarantee: binning extraction never materializes ND2 pixels.

Encodes the behavioral contract that the extract-metadata path obtains camera
binning from the ND2 text-info block only. A regression to nd2.imread (full
pixel decode, ~92 GB/well) is caught here by making imread raise.
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


@pytest.mark.integration
def test_binning_extraction_reads_no_pixels(monkeypatch):
    fake = FakeND2File({"description": "Camera\nBinning: 2x2\nExposure: 100ms"})

    def _boom(*a, **k):
        raise AssertionError("nd2.imread called -- pixels materialized")

    monkeypatch.setattr(nd2, "imread", _boom)
    monkeypatch.setattr(nd2, "ND2File", lambda *a, **k: fake)

    assert _parse_binning_from_nd2("/fake/well.nd2") == "2x2"
    assert fake.entered and fake.exited, "ND2File must be used as a context manager"


@pytest.mark.integration
def test_binning_extraction_resilient_to_open_error(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("nd2.imread called -- pixels materialized")

    def _raise(*a, **k):
        raise OSError("corrupt or missing file")

    monkeypatch.setattr(nd2, "imread", _boom)
    monkeypatch.setattr(nd2, "ND2File", _raise)

    assert _parse_binning_from_nd2("/nonexistent.nd2") is None
