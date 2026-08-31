"""Self-check for the ported run_phenotype_direct.py niceties:
atomic_write_parquet (crash-safe write) and _merge_worker_cap (RAM-sized concurrency).

The runner's top-level `lib.shared.*` imports are stubbed so this test is fast and
env-independent (skimage/polars/tifffile are not needed to exercise these two helpers).

Run: python tests/direct/test_phenotype_direct_niceties.py
"""
import os
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


# stub the heavy lib.shared.* imports the runner does at module load
_stub("lib")
_stub("lib.shared")
_stub("lib.shared.file_utils", get_data_output_path=lambda *a, **k: "",
      get_image_output_path=lambda *a, **k: "", validate_dtypes=lambda df: df)
_stub("lib.shared.image_io", read_image=lambda *a, **k: None, save_image=lambda *a, **k: None)
_stub("lib.shared.illumination_correction", apply_ic_field=lambda *a, **k: None)
_stub("lib.shared.parquet_io", write_parquet=lambda df, p: df.to_parquet(p),
      read_parquets=lambda *a, **k: None)
_stub("lib.shared.rule_utils", get_alignment_params=lambda *a, **k: {},
      get_segmentation_params=lambda *a, **k: {})

_stub("lib.shared.resource_monitor", monitor_step=lambda *a, **k: __import__("contextlib").nullcontext(), set_benchmark_context=lambda *a, **k: None)  # ponytail: nullcontext stub; real context manager is only needed when the step actually runs

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "direct"))
import run_phenotype_direct as rpd  # noqa: E402


def test_atomic_write_parquet_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "sub" / "x.parquet"  # parent does not exist yet
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        rpd.atomic_write_parquet(df, out)
        assert out.exists() and out.stat().st_size > 0
        pd.testing.assert_frame_equal(pd.read_parquet(out), df)
        assert not list(out.parent.glob("*.tmp"))  # tmp cleaned by os.replace


def test_merge_worker_cap():
    orig = rpd._available_gb
    try:
        rpd._available_gb = lambda: 700.0   # 700 // 70 = 10
        assert rpd._merge_worker_cap(8) == 8      # requested < mem cap
        assert rpd._merge_worker_cap(20) == 10    # capped by mem
        rpd._available_gb = lambda: 140.0   # 140 // 70 = 2
        assert rpd._merge_worker_cap(8) == 2
        rpd._available_gb = lambda: None    # unknown -> conservative default 2
        assert rpd._merge_worker_cap(8) == 2
        assert rpd._merge_worker_cap(1) == 1      # never below 1
    finally:
        rpd._available_gb = orig


if __name__ == "__main__":
    test_atomic_write_parquet_roundtrip()
    test_merge_worker_cap()
    print("OK: atomic_write_parquet + _merge_worker_cap pass")
