"""run_merge_direct.step_format_merge writes the coordinate columns, fails loudly without
metadata, and agrees with the Snakemake script on identical inputs.

Real lib, nested {plate}/{row}/{col} layout, no stubs -- the join against preprocess
metadata is the thing under test. The 0-row metadata case is the exact input the
TDP-GWS recomb rerun had on 2026-10-08, when 55 wells of 36-column merge tables were
written without complaint.

Run: python -m pytest -q tests/direct/test_merge_direct_format_merge.py
"""

import importlib.util
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "workflow") not in sys.path:
    sys.path.insert(0, str(ROOT / "workflow"))

RUNNER = ROOT / "scripts" / "direct" / "run_merge_direct.py"
SMK_SCRIPT = ROOT / "workflow" / "scripts" / "merge" / "format_merge.py"
CFG = {
    "approach": "fast",
    "phenotype_dimensions": [100, 200],
    "sbs_dimensions": [50, 100],
}
COORD_COLS = [
    f"{c}_{s}"
    for s in ("0", "1")
    for c in ("x_pos", "y_pos", "pixel_size_x", "pixel_size_y", "global_j", "global_i")
]
STEM = Path("1") / "A" / "1"


def _load_runner():
    spec = importlib.util.spec_from_file_location("run_merge_direct", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(n_tiles, tile_px, px, origin=(1000.0, 2000.0), channels=2):
    rows = []
    for t in range(n_tiles):
        for ch in range(channels):
            rows.append(
                dict(
                    plate=1,
                    well="A1",
                    tile=t,
                    channel=ch,
                    x_pos=origin[0] + (t % 2) * tile_px[1] * px,
                    y_pos=origin[1] + (t // 2) * tile_px[0] * px,
                    pixel_size_x=px,
                    pixel_size_y=px,
                )
            )
    return pd.DataFrame(rows)


def _inputs(n_tiles=4, cells_per_tile=5, seed=0):
    """fast_merge + sbs cells + phenotype_cp_min + both metadata frames for one well."""
    rng = np.random.default_rng(seed)
    rows = []
    for t in range(n_tiles):
        for c in range(cells_per_tile):
            rows.append(
                dict(
                    plate=1,
                    well="A1",
                    tile=t,
                    cell_0=c + 1,
                    i_0=rng.uniform(0, 100),
                    j_0=rng.uniform(0, 200),
                    site=t,
                    cell_1=c + 1,
                    i_1=rng.uniform(0, 50),
                    j_1=rng.uniform(0, 100),
                    distance=rng.uniform(0, 2),
                )
            )
    fast_merge = pd.DataFrame(rows)
    cells = pd.DataFrame(
        dict(
            plate=1,
            well="A1",
            tile=fast_merge["site"],
            cell=fast_merge["cell_1"],
            cell_barcode_0=["ACGT"] * len(fast_merge),
            gene_symbol_0=["TARDBP"] * len(fast_merge),
            gene_symbol_1=[None] * len(fast_merge),
            prefix_recomb=["GGC"] * len(fast_merge),
        )
    )
    cp_min = pd.DataFrame(
        dict(
            plate=1,
            well="A1",
            tile=fast_merge["tile"],
            label=fast_merge["cell_0"],
            cell_DAPI_min=rng.uniform(0, 1, len(fast_merge)),
            cell_GFP_min=rng.uniform(0, 1, len(fast_merge)),
        )
    )
    ph_meta = _metadata(n_tiles, CFG["phenotype_dimensions"], 0.5)
    sbs_meta = _metadata(
        n_tiles, CFG["sbs_dimensions"], 1.0, origin=(1000.0, 2000.0), channels=3
    )
    return fast_merge, cells, cp_min, ph_meta, sbs_meta


def _nested_well(root, fast_merge, cells, cp_min, ph_meta, sbs_meta):
    files = {
        "preprocess/metadata/phenotype": ("combined_metadata", ph_meta),
        "preprocess/metadata/sbs": ("combined_metadata", sbs_meta),
        "merge/parquets": ("fast_merge", fast_merge),
        "sbs/parquets": ("cells", cells),
        "phenotype/parquets": ("phenotype_cp_min", cp_min),
    }
    for subdir, (name, df) in files.items():
        d = root / subdir / STEM
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"{name}.parquet", index=False)


def test_step_format_merge_writes_global_coords(tmp_path):
    runner = _load_runner()
    _nested_well(tmp_path, *_inputs())
    paths = runner.Paths(tmp_path, 1, "A1")
    runner._ensure_dirs(paths)

    runner.step_format_merge(CFG, paths, force=False)

    out = pd.read_parquet(paths.merge_formatted)
    assert (
        paths.merge_formatted
        == tmp_path / "merge/parquets/1/A/1/merge_formatted.parquet"
    )
    assert len(out) == 20
    names = list(out.columns)
    # the 12 columns sit directly after channels_min, in attach order
    assert names[names.index("channels_min") + 1 :] == COORD_COLS
    assert out[COORD_COLS].notna().all().all()
    # anchored at the well's top-left tile corner: no negative globals, max inside the 2x2 mosaic
    assert (out[["global_i_0", "global_j_0"]] >= 0).all().all()
    assert out["global_i_0"].max() < 2 * CFG["phenotype_dimensions"][0]
    assert out["global_j_0"].max() < 2 * CFG["phenotype_dimensions"][1]
    assert "prefix_recomb" in names and "mapped_single_gene" in names


def test_step_format_merge_fails_on_empty_metadata_probe(tmp_path):
    runner = _load_runner()
    fast_merge, cells, cp_min, ph_meta, sbs_meta = _inputs()
    probe = ph_meta.drop(columns="channel").iloc[
        0:0
    ]  # the 2026-10-08 scaffold: schema only, 0 rows
    _nested_well(tmp_path, fast_merge, cells, cp_min, probe, probe)
    paths = runner.Paths(tmp_path, 1, "A1")
    runner._ensure_dirs(paths)

    with pytest.raises(ValueError, match="metadata is empty"):
        runner.step_format_merge(CFG, paths, force=False)
    assert not paths.merge_formatted.exists()


def test_skip_if_exists_respected(tmp_path):
    runner = _load_runner()
    _nested_well(tmp_path, *_inputs())
    paths = runner.Paths(tmp_path, 1, "A1")
    runner._ensure_dirs(paths)
    runner.step_format_merge(CFG, paths, force=False)
    before = paths.merge_formatted.stat().st_mtime_ns
    runner.step_format_merge(CFG, paths, force=False)
    assert paths.merge_formatted.stat().st_mtime_ns == before


class _Snakemake:
    """Just enough of the `snakemake` script global for workflow/scripts/merge/format_merge.py."""

    class _Input(list):
        pass

    class _Params:
        pass

    def __init__(self, paths, out):
        self.input = self._Input(
            [str(paths.fast_merge), str(paths.sbs_cells), str(paths.ph_cp_min)]
        )
        self.input.phenotype_metadata = str(paths.ph_metadata)
        self.input.sbs_metadata = str(paths.sbs_metadata)
        self.params = self._Params()
        self.params.approach = CFG["approach"]
        self.params.phenotype_dimensions = CFG["phenotype_dimensions"]
        self.params.sbs_dimensions = CFG["sbs_dimensions"]
        self.output = [str(out)]


def test_direct_and_snakemake_format_merge_agree(tmp_path):
    runner = _load_runner()
    _nested_well(tmp_path, *_inputs(seed=3))
    paths = runner.Paths(tmp_path, 1, "A1")
    runner._ensure_dirs(paths)
    runner.step_format_merge(CFG, paths, force=False)

    smk_out = tmp_path / "smk_merge_formatted.parquet"
    runpy.run_path(
        str(SMK_SCRIPT), init_globals={"snakemake": _Snakemake(paths, smk_out)}
    )

    direct = pd.read_parquet(paths.merge_formatted)
    smk = pd.read_parquet(smk_out)
    assert list(direct.columns) == list(smk.columns)
    pd.testing.assert_frame_equal(direct, smk)
    assert all(c in direct.columns for c in COORD_COLS)
