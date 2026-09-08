"""Regression test for issue #36 — phenotype convert-round zarr collision.

Phenotype `combos` carry a `round` column that is NOT encoded in the output path
(only `cycle` is), so every round-row of a tile collapses to the same
`image_<plate>.zarr/<well>/<tile>` path. The direct runner used to build one
convert task per combos row, dispatching those duplicates concurrently to a
ProcessPoolExecutor -> non-atomic out_exists/create race -> ContainsArrayError.

The fix dedups convert tasks by output path before dispatch. Because all rounds
are gathered+stacked into one combined image (get_sample_fps(round_order=all)),
the duplicate rows produce byte-identical output, so dedup is lossless.

This test stubs the heavy lib.* imports and monkeypatches run_parallel/run_ic_step
to capture the task list the convert step would dispatch.

Run: python tests/direct/test_convert_dedup.py   (or under pytest, marked unit)
"""
import argparse
import sys
import types
from pathlib import Path

import pandas as pd

try:
    import pytest
    _mark = pytest.mark.unit
except ImportError:  # allow bare `python tests/direct/test_convert_dedup.py`
    def _mark(f):
        return f


def _stub(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m


def _out_path(loc, *a, **k):
    """Realistic HCS-style path: includes cycle when present (SBS), never round.

    Reproduces the collapse — phenotype loc has no cycle, so all rounds of a tile
    map to the same path; SBS loc has cycle, so paths stay unique per cycle.
    """
    well = loc.get("well") or loc.get("row", "")
    parts = [f"image_{loc['plate']}.zarr", str(well)]
    if "tile" in loc:
        parts.append(str(loc["tile"]))
    if "cycle" in loc:
        parts.append(f"c{loc['cycle']}")
    parts.append("zarr.json")
    return "/".join(parts)


# --- stub heavy lib.* imports the runner does at module load ------------------
_stub("lib")
_stub("lib.preprocess")
_stub("lib.shared")
_stub(
    "lib.preprocess.preprocess",
    convert_to_array=lambda *a, **k: None,
    extract_metadata=lambda *a, **k: None,
    get_data_config=lambda *a, **k: {
        "data_format": "nd2",
        "image_data_organization": "tile",
        "metadata_data_organization": "tile",
        "channel_order_flip": False,
        "n_z_planes": None,
        "channel_order": None,
    },
)
_stub(
    "lib.preprocess.file_utils",
    get_metadata_wildcard_combos=lambda s, m: pd.DataFrame(columns=["plate", "well", "tile"]),
    get_sample_fps=lambda df, **k: ["r1.nd2", "r2.nd2"],  # combined all-rounds file list
)
_stub(
    "lib.shared.file_utils",
    get_data_output_path=lambda *a, **k: "md.tsv",
    get_image_output_path=_out_path,
    validate_dtypes=lambda df: df,
)
_stub("lib.shared.illumination_correction", calculate_ic_field=lambda *a, **k: None)
_stub("lib.shared.image_io", save_image=lambda *a, **k: None)
_stub("lib.shared.parquet_io", write_parquet=lambda df, p: None)
_stub(
    "lib.shared.resource_monitor",
    monitor_step=lambda *a, **k: __import__("contextlib").nullcontext(),
    set_benchmark_context=lambda *a, **k: None,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "direct"))
import run_preprocess_direct as rpd  # noqa: E402


def _run(image_type, combos_df, tmp_path):
    """Drive process() with captured run_parallel/run_ic_step; return convert tasks."""
    samples = tmp_path / "samples.tsv"
    samples.write_text("plate\twell\ttile\n1\tA1\t0\n")
    combo_fp = tmp_path / "combos.tsv"
    combos_df.to_csv(combo_fp, sep="\t", index=False)

    config = {
        "all": {"root_fp": str(tmp_path), "image_format": "tiff"},
        "preprocess": {
            f"{image_type}_samples_fp": str(samples),
            f"{image_type}_combo_fp": str(combo_fp),
            f"{image_type}_round_order": [1, 2],
        },
    }
    args = argparse.Namespace(
        plate_filter=None, max_tiles=None, workers=4, image_type=image_type
    )

    calls = []
    orig_rp, orig_ic = rpd.run_parallel, rpd.run_ic_step
    rpd.run_parallel = lambda tasks, fn, workers, label, **k: (calls.append((label, list(tasks))) or 0)
    rpd.run_ic_step = lambda *a, **k: 0
    try:
        rpd.process(image_type, config, args)
    finally:
        rpd.run_parallel, rpd.run_ic_step = orig_rp, orig_ic

    convert = [t for lbl, t in calls if lbl.startswith("Convert images")]
    assert len(convert) == 1, f"expected one convert step, got labels {[c[0] for c in calls]}"
    return convert[0]


@_mark
def test_convert_dedups_phenotype_rounds(tmp_path=None):
    """4 round-rows across 2 tiles -> 2 convert tasks (not 4), file list preserved."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    combos = pd.DataFrame(
        {
            "plate": ["1", "1", "1", "1"],
            "well": ["A1", "A1", "A1", "A1"],
            "tile": ["0", "0", "1", "1"],
            "round": ["1", "2", "1", "2"],
        }
    )
    tasks = _run("phenotype", combos, tmp_path)

    out_paths = [t[-1] for t in tasks]  # output_path is last element of the task tuple
    assert len(tasks) == 2, f"dedup failed: {len(tasks)} tasks for 2 unique tiles"
    assert len(set(out_paths)) == 2, f"duplicate output paths dispatched: {out_paths}"
    # no data loss: each surviving task still carries the full combined-rounds file list
    for t in tasks:
        assert t[0] == ["r1.nd2", "r2.nd2"], f"file list altered by dedup: {t[0]}"


@_mark
def test_convert_keeps_distinct_sbs_cycles(tmp_path=None):
    """SBS: cycle is a real path dimension -> dedup must NOT collapse cycles."""
    tmp_path = tmp_path or Path(__import__("tempfile").mkdtemp())
    combos = pd.DataFrame(
        {
            "plate": ["1", "1", "1", "1"],
            "well": ["A1", "A1", "A1", "A1"],
            "tile": ["0", "0", "1", "1"],
            "cycle": ["1", "2", "1", "2"],
        }
    )
    tasks = _run("sbs", combos, tmp_path)

    out_paths = [t[-1] for t in tasks]
    assert len(tasks) == 4, f"dedup wrongly collapsed distinct cycles: {out_paths}"
    assert len(set(out_paths)) == 4, f"expected 4 unique paths, got {out_paths}"


if __name__ == "__main__":
    import tempfile

    test_convert_dedups_phenotype_rounds(Path(tempfile.mkdtemp()))
    test_convert_keeps_distinct_sbs_cycles(Path(tempfile.mkdtemp()))
    print("OK: convert dedup collapses phenotype rounds, preserves SBS cycles")
