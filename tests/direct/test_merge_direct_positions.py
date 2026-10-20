"""run_merge_direct runs the `positions` chain into the production nested layout (#86).

Production merges go through this runner, not the Snakemake rule. Until #86 it raised for
any approach but `fast`, so no plate could run `positions` or emit the persisted placement
(#83/#84). These tests use the synthetic well from tests/test_positions_merge.py written
in the nested {plate}/{row}/{col}/ layout the runner auto-detects, and the real lib (no
stubs -- the positions math is the thing under test).

Run: python -m pytest -q tests/direct/test_merge_direct_positions.py
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "workflow", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_positions_merge import TILE_PX, make_screen  # noqa: E402
from lib.merge.positions_merge import (  # noqa: E402
    MERGE_COLUMNS,
    placement_from_table,
    tile_pixel_to_um,
)

RUNNER = ROOT / "scripts" / "direct" / "run_merge_direct.py"
CFG = {
    "approach": "positions",
    "threshold": 2,
    "phenotype_dimensions": [TILE_PX, TILE_PX],
    "sbs_dimensions": [TILE_PX, TILE_PX],
    "flipud": True,
    "fliplr": False,
    "rot90": 1,
}


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_merge_direct", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _nested_well(root, ph, sbs):
    """Write one synthetic well (plate 1, well A1) in the nested production layout."""
    stem = Path("1") / "A" / "1"
    files = {
        "preprocess/metadata/phenotype": ("combined_metadata", ph[1]),
        "preprocess/metadata/sbs": ("combined_metadata", sbs[1]),
        "phenotype/parquets": ("phenotype_info", ph[0]),
        "sbs/parquets": ("sbs_info", sbs[0]),
    }
    for subdir, (name, df) in files.items():
        d = root / subdir / stem
        d.mkdir(parents=True)
        df.to_parquet(d / f"{name}.parquet", index=False)


def test_positions_step_writes_nested_merge_and_placement(tmp_path):
    runner = _load_runner()
    ph, sbs = make_screen(seed=1, orientation=(True, False, 1))
    _nested_well(tmp_path, ph, sbs)
    paths = runner.Paths(tmp_path, 1, "A1")
    runner._ensure_dirs(paths)

    runner.step_positions_merge(CFG, paths, 1, "A1", force=False)

    assert (
        paths.positions_merge
        == tmp_path / "merge/parquets/1/A/1/positions_merge.parquet"
    )
    assert (
        paths.positions_placement
        == tmp_path / "merge/parquets/1/A/1/positions_placement.parquet"
    )
    merged = pd.read_parquet(paths.positions_merge)
    assert list(merged.columns) == MERGE_COLUMNS
    assert len(merged) > 1000
    placement = placement_from_table(pd.read_parquet(paths.positions_placement))
    assert set(placement) == {"phenotype", "sbs"}
    row = merged.iloc[0]
    xy = tile_pixel_to_um(
        placement["phenotype"], int(row["tile"]), [[row["i_0"], row["j_0"]]]
    )
    assert xy.shape == (1, 2)

    # existing outputs are skipped without --force, like every other step
    before = paths.positions_merge.stat().st_mtime_ns
    runner.step_positions_merge(CFG, paths, 1, "A1", force=False)
    assert paths.positions_merge.stat().st_mtime_ns == before


def test_run_well_dispatches_positions_not_fast(tmp_path, monkeypatch):
    runner = _load_runner()
    ph, sbs = make_screen(seed=2, orientation=(True, False, 1))
    _nested_well(tmp_path, ph, sbs)

    def must_not_run(*args, **kwargs):
        raise AssertionError("fast step ran under approach=positions")

    monkeypatch.setattr(runner, "step_fast_alignment", must_not_run)
    monkeypatch.setattr(runner, "step_fast_merge", must_not_run)
    called = []
    for name in ("step_format_merge", "step_deduplicate_merge", "step_final_merge"):
        monkeypatch.setattr(
            runner, name, lambda cfg, paths, force, _n=name: called.append(_n)
        )

    runner.run_well(CFG, tmp_path, 1, "A1", workers=1, force=False)

    assert (tmp_path / "merge/parquets/1/A/1/positions_placement.parquet").exists()
    assert called == ["step_format_merge", "step_deduplicate_merge", "step_final_merge"]


def test_approach_guard_accepts_positions_rejects_stitch():
    runner = _load_runner()
    assert runner._check_approach({"approach": "positions"}) == "positions"
    assert runner._check_approach({}) == "fast"
    with pytest.raises(SystemExit):
        runner._check_approach({"approach": "stitch"})
