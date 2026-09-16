"""Self-check for run_sbs_direct.combine_after: which --step values run combine.

Guards the contract that `--step post-seg` also runs combine (so merge's per-well
parquets exist after one CPU call), without combining a partial tile selection.

The runner's top-level `lib.shared.*` imports are stubbed so this test is fast and
env-independent.

Run: python tests/direct/test_sbs_direct_combine_after.py
"""
import sys
import types
from pathlib import Path


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
_stub("lib.shared.illumination_correction", apply_ic_field=lambda *a, **k: None,
      combine_ic_images=lambda *a, **k: None)
_stub("lib.shared.parquet_io", write_parquet=lambda *a, **k: None,
      read_parquets=lambda *a, **k: None, read_table=lambda *a, **k: None)
_stub("lib.shared.combine_dfs", combine_tile_dfs=lambda *a, **k: None)
_stub("lib.shared.rule_utils", get_call_cells_params=lambda *a, **k: {},
      get_segmentation_params=lambda *a, **k: {}, get_spot_detection_params=lambda *a, **k: {})
_stub("lib.shared.resource_monitor", monitor_step=lambda *a, **k: None,
      set_benchmark_context=lambda *a, **k: None)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "direct"))
import run_sbs_direct as rsd  # noqa: E402


def test_post_seg_runs_combine_on_full_tile_set():
    assert rsd.combine_after("post-seg", False) is True


def test_post_seg_skips_combine_on_partial_tile_set():
    # combining a shard writes a truncated well parquet
    assert rsd.combine_after("post-seg", True) is False


def test_combine_only_unchanged():
    # --step combine is called directly by run_sbs_combine_both.sh; keep it working
    assert rsd.combine_after("combine", False) is True
    assert rsd.combine_after("combine", True) is True


def test_all_unchanged():
    assert rsd.combine_after("all", False) is True


def test_other_steps_never_combine():
    for step in ("tiles", "pre-seg", "segment"):
        assert rsd.combine_after(step, False) is False
        assert rsd.combine_after(step, True) is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all checks passed")
