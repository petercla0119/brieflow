"""render_well wiring: config -> channels, and the direct runner's opt-in step (#88)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "workflow", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from test_positions_merge import TILE_PX, make_screen  # noqa: E402
from lib.merge.render_well import render_channels  # noqa: E402

RUNNER = ROOT / "scripts" / "direct" / "run_merge_direct.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_merge_direct", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_render_channels_default_dapi_and_configured():
    assert render_channels({}, "phenotype", 0) == {"DAPI": 0}
    assert render_channels({"render_well_channels": None}, "sbs", 4) == {"DAPI": 4}
    cfg = {"render_well_channels": {"phenotype": {"DAPI": 0, "WGA": 2}}}
    assert render_channels(cfg, "phenotype", 0) == {"DAPI": 0, "WGA": 2}
    assert render_channels(cfg, "sbs", 4) == {"DAPI": 4}


def test_run_well_renders_only_when_opted_in(tmp_path, monkeypatch):
    runner = _load_runner()
    ph, sbs = make_screen(seed=5, orientation=(True, False, 1))
    stem = Path("1") / "A" / "1"
    for subdir, name, df in (
        ("preprocess/metadata/phenotype", "combined_metadata", ph[1]),
        ("preprocess/metadata/sbs", "combined_metadata", sbs[1]),
        ("phenotype/parquets", "phenotype_info", ph[0]),
        ("sbs/parquets", "sbs_info", sbs[0]),
    ):
        d = tmp_path / subdir / stem
        d.mkdir(parents=True)
        df.to_parquet(d / f"{name}.parquet", index=False)
    for name in ("step_format_merge", "step_deduplicate_merge", "step_final_merge"):
        monkeypatch.setattr(runner, name, lambda cfg, paths, force: None)
    rendered = []
    monkeypatch.setattr(
        runner,
        "step_render_well",
        lambda cfg, config, paths, out_root, plate, well, force: rendered.append(
            (plate, well, paths.well_image("phenotype"))
        ),
    )
    cfg = {
        "approach": "positions",
        "threshold": 2,
        "phenotype_dimensions": [TILE_PX, TILE_PX],
        "sbs_dimensions": [TILE_PX, TILE_PX],
        "flipud": True,
        "rot90": 1,
    }
    runner.run_well(cfg, tmp_path, 1, "A1", workers=1, force=False, config={})
    assert rendered == []
    runner.run_well(
        dict(cfg, render_well=True),
        tmp_path,
        1,
        "A1",
        workers=1,
        force=False,
        config={},
    )
    assert rendered == [(1, "A1", tmp_path / "merge/images/1/A/1/phenotype_well.zarr")]
