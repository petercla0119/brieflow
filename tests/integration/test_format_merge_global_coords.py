"""Every merge table the small test wrote carries the 12 stage-coordinate columns.

Runs after the pipeline on every CI leg -- flat `P-1_W-A1__merge_formatted.parquet`
on tiff/direct, nested `1/A/1/merge_formatted.parquet` on zarr -- so both the
Snakemake rule and the direct runner are checked against real output, not a
synthetic frame. Skips with a reason only when no output exists.
"""

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_ANALYSIS = Path(__file__).resolve().parents[1] / "small_test_analysis"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

COORD_COLS = [
    f"{c}_{s}"
    for s in ("0", "1")
    for c in ("x_pos", "y_pos", "pixel_size_x", "pixel_size_y", "global_j", "global_i")
]


def _resolve_output_dir():
    canonical = _TEST_ANALYSIS / "brieflow_output"
    if canonical.exists():
        return canonical
    candidates = sorted(
        [
            p
            for p in _TEST_ANALYSIS.iterdir()
            if p.is_dir() and p.name.startswith("brieflow_output")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if (p / "preprocess" / "metadata").exists():
            return p
    pytest.skip("Brieflow output directory not found. Run run_brieflow.sh first.")


def _files(name):
    files = sorted(_resolve_output_dir().glob(f"merge/parquets/**/*{name}.parquet"))
    if not files:
        pytest.skip(f"No {name} parquet outputs found; run the merge module first.")
    return files


@pytest.mark.integration
def test_merge_formatted_has_non_null_global_coords():
    from workflow.lib.shared.parquet_io import read_parquet

    for fp in _files("merge_formatted"):
        df = read_parquet(fp)
        missing = [c for c in COORD_COLS if c not in df.columns]
        assert not missing, f"{fp}: missing {missing}"
        assert df[COORD_COLS].notna().all().all(), f"{fp}: null coordinates"
        assert (df[["pixel_size_x_0", "pixel_size_x_1"]] > 0).all().all(), fp
        for s in ("0", "1"):
            per_well = df.groupby(["plate", "well"])[
                [f"global_i_{s}", f"global_j_{s}"]
            ].min()
            assert (per_well >= 0).all().all(), f"{fp}: negative global_{s}"


@pytest.mark.integration
@pytest.mark.parametrize("name", ["merge_deduplicated", "merge_final"])
def test_downstream_merge_tables_keep_global_coords(name):
    from pyarrow.parquet import read_schema

    for fp in _files(name):
        names = read_schema(fp).names
        missing = [c for c in COORD_COLS if c not in names]
        assert not missing, f"{fp}: missing {missing}"
