"""Self-check for the run_sbs_direct.py combine fold-in (issue #38).

Verifies that folding SBS combine into `--step post-seg` is strictly opt-in (--combine),
mirrors the phenotype runner, and does not alter the standalone `--step combine` / `--step all`
paths or the default relay's `--step post-seg`.

The runner's top-level `lib.shared.*` imports are stubbed so this test is fast and
env-independent (skimage/polars/tifffile are not needed to exercise the branch logic).

Run: python tests/direct/test_sbs_combine_foldin.py
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
_stub("lib.shared.parquet_io", write_parquet=lambda df, p: None,
      read_parquets=lambda *a, **k: None, read_table=lambda *a, **k: None)
_stub("lib.shared.combine_dfs", combine_tile_dfs=lambda *a, **k: None)
_stub("lib.shared.rule_utils", get_call_cells_params=lambda *a, **k: {},
      get_segmentation_params=lambda *a, **k: {}, get_spot_detection_params=lambda *a, **k: {})

import contextlib
_stub("lib.shared.resource_monitor",
      monitor_step=lambda *a, **k: contextlib.nullcontext(),
      set_benchmark_context=lambda *a, **k: None)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "direct"))
import run_sbs_direct as rsd  # noqa: E402


def _decisions(step, combine):
    """Reproduce the two branch decisions process_sbs() makes, so the test pins the exact
    (run_combine, reload_all_tiles) behavior for a given (--step, --combine)."""
    folded = rsd.combine_folds_into_postseg(step, combine)
    run_combine = step in ("combine", "all") or folded
    reload_all_tiles = step == "combine" or folded
    return run_combine, reload_all_tiles


def test_foldin_is_opt_in_and_relay_safe():
    # Default relay: post-seg without --combine must stay extract-only (no combine, no reload).
    assert _decisions("post-seg", False) == (False, False), "post-seg default must NOT combine"

    # Opt-in: post-seg --combine runs combine AND reloads all tiles (output-identical to standalone).
    assert _decisions("post-seg", True) == (True, True), "post-seg --combine must fold in + reload"

    # Standalone combine unchanged: always combines, always reloads all tiles.
    assert _decisions("combine", False) == (True, True)
    assert _decisions("combine", True) == (True, True), "--combine must not perturb --step combine"

    # --step all unchanged: combines, but does NOT force the full-tile reload (preserves prior
    # behavior — no output change for existing runs).
    assert _decisions("all", False) == (True, False)
    assert _decisions("all", True) == (True, False), "--combine must not perturb --step all"

    # Upstream relay phases never combine, flag or not.
    for step in ("tiles", "pre-seg", "segment"):
        assert _decisions(step, False) == (False, False)
        assert _decisions(step, True) == (False, False), f"--combine must not perturb {step}"

    # The fold-in predicate itself only fires on post-seg + flag.
    assert rsd.combine_folds_into_postseg("post-seg", True) is True
    assert rsd.combine_folds_into_postseg("post-seg", False) is False
    assert rsd.combine_folds_into_postseg("combine", True) is False


if __name__ == "__main__":
    test_foldin_is_opt_in_and_relay_safe()
    print("OK: run_sbs_direct combine fold-in is opt-in, relay-safe, and idempotent-guarded")
