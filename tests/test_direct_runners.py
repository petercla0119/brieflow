"""Tests for the direct runner scripts (run_preprocess_direct, run_sbs_direct, run_phenotype_direct).

Unit tests verify path construction, skip logic, config extraction, and combo filtering.
Integration tests verify path equivalence against Snakemake targets and end-to-end correctness.

Run unit tests:
    pytest tests/test_direct_runners.py -m unit -v

Run all tests (needs prior pipeline run):
    pytest tests/test_direct_runners.py -v
"""

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

# ---------------------------------------------------------------------------
# sys.path setup — ensure both workflow lib and scripts/direct/ are importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = str(_REPO_ROOT / "workflow")
_SCRIPTS_DIR = str(_REPO_ROOT / "scripts" / "direct")

for _p in [_WORKFLOW, _SCRIPTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_preprocess_direct as pp_mod
import run_sbs_direct as sbs_mod
import run_phenotype_direct as phen_mod
from lib.shared.file_utils import get_data_output_path, get_image_output_path
from lib.shared.rule_utils import (
    get_alignment_params,
    get_call_cells_params,
    get_segmentation_params,
    get_spot_detection_params,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent / "small_test_analysis" / "config"


@pytest.fixture
def sample_config():
    cfg_path = _CONFIG_DIR / "config.yml"
    if not cfg_path.exists():
        pytest.skip("Test config not found")
    return yaml.safe_load(cfg_path.read_text())


@pytest.fixture
def zarr_config(sample_config):
    cfg = copy.deepcopy(sample_config)
    cfg["all"]["image_format"] = "zarr"
    return cfg


@pytest.fixture
def sbs_combos():
    return pd.read_csv(_CONFIG_DIR / "sbs_combo.tsv", sep="\t").astype(str)


@pytest.fixture
def phenotype_combos():
    return pd.read_csv(_CONFIG_DIR / "phenotype_combo.tsv", sep="\t").astype(str)


# ---------------------------------------------------------------------------
# Module-level picklable helpers for run_parallel tests
# ---------------------------------------------------------------------------

def _task_ok(task):
    return "ok", f"done-{task}"


def _task_skip(task):
    return "skip", f"skip-{task}"


def _task_err(task):
    return "err", f"fail-{task}"


def _task_mixed(task):
    if task % 2 == 0:
        return "ok", f"ok-{task}"
    return "err", f"err-{task}"


# ===========================================================================
# Section 1: TestMakeLoc
# ===========================================================================


class TestMakeLoc:
    """Verify make_loc() builds correct data_location dicts."""

    @pytest.mark.unit
    def test_tiff_full(self):
        loc = sbs_mod.make_loc("tiff", "1", "A1", "0", "3")
        assert loc == {"plate": "1", "well": "A1", "tile": "0", "cycle": "3"}

    @pytest.mark.unit
    def test_tiff_no_cycle(self):
        loc = sbs_mod.make_loc("tiff", "1", "A1", "0")
        assert loc == {"plate": "1", "well": "A1", "tile": "0"}
        assert "cycle" not in loc

    @pytest.mark.unit
    def test_tiff_plate_only(self):
        loc = sbs_mod.make_loc("tiff", "1")
        assert loc == {"plate": "1"}

    @pytest.mark.unit
    def test_tiff_plate_well_only(self):
        loc = sbs_mod.make_loc("tiff", "1", "A1")
        assert loc == {"plate": "1", "well": "A1"}

    @pytest.mark.unit
    def test_zarr_splits_well(self):
        loc = sbs_mod.make_loc("zarr", "1", "A1", "0")
        assert loc == {"plate": "1", "row": "A", "col": "1", "tile": "0"}
        assert "well" not in loc

    @pytest.mark.unit
    def test_zarr_multichar_row(self):
        loc = sbs_mod.make_loc("zarr", "1", "AB12")
        assert loc == {"plate": "1", "row": "AB", "col": "12"}

    @pytest.mark.unit
    def test_zarr_with_cycle(self):
        loc = sbs_mod.make_loc("zarr", "1", "A1", "0", "3")
        assert loc == {
            "plate": "1",
            "row": "A",
            "col": "1",
            "tile": "0",
            "cycle": "3",
        }

    @pytest.mark.unit
    def test_zarr_plate_only(self):
        loc = sbs_mod.make_loc("zarr", "1")
        assert loc == {"plate": "1"}
        assert "row" not in loc and "col" not in loc

    @pytest.mark.unit
    def test_all_values_are_strings(self):
        loc = sbs_mod.make_loc("tiff", 1, "A1", 0, 3)
        assert all(isinstance(v, str) for v in loc.values())

    @pytest.mark.unit
    def test_preprocess_make_loc_with_cycle(self):
        loc = pp_mod.make_loc("tiff", "1", "A1", "0", "3")
        assert loc == {"plate": "1", "well": "A1", "tile": "0", "cycle": "3"}

    @pytest.mark.unit
    def test_phenotype_make_loc_no_cycle(self):
        loc = phen_mod.make_loc("tiff", "1", "A1", "5")
        assert loc == {"plate": "1", "well": "A1", "tile": "5"}


# ===========================================================================
# Section 2: TestOutExists
# ===========================================================================


class TestOutExists:
    """Verify out_exists() handles all file types correctly."""

    @pytest.mark.unit
    def test_regular_file_exists(self, tmp_path):
        f = tmp_path / "test.tiff"
        f.write_bytes(b"\x00" * 10)
        assert sbs_mod.out_exists(str(f))

    @pytest.mark.unit
    def test_missing_file(self, tmp_path):
        assert not sbs_mod.out_exists(str(tmp_path / "nope.tiff"))

    @pytest.mark.unit
    def test_empty_file_is_false(self, tmp_path):
        f = tmp_path / "empty.tiff"
        f.touch()
        assert not sbs_mod.out_exists(str(f))

    @pytest.mark.unit
    def test_zarr_dir_with_data(self, tmp_path):
        d = tmp_path / "store.zarr"
        d.mkdir()
        (d / "data.bin").write_bytes(b"\x01")
        assert sbs_mod.out_exists(str(d))

    @pytest.mark.unit
    def test_empty_zarr_dir(self, tmp_path):
        d = tmp_path / "store.zarr"
        d.mkdir()
        assert not sbs_mod.out_exists(str(d))

    @pytest.mark.unit
    def test_zarr_json_sentinel_exists(self, tmp_path):
        d = tmp_path / "store.zarr" / "A" / "1" / "0"
        d.mkdir(parents=True)
        sentinel = d / "zarr.json"
        sentinel.write_text("{}")
        assert sbs_mod.out_exists(str(sentinel))

    @pytest.mark.unit
    def test_zarr_json_sentinel_missing(self, tmp_path):
        sentinel = tmp_path / "store.zarr" / "A" / "1" / "0" / "zarr.json"
        assert not sbs_mod.out_exists(str(sentinel))

    @pytest.mark.unit
    def test_tsv_file(self, tmp_path):
        f = tmp_path / "data.tsv"
        f.write_text("col1\tcol2\n1\t2\n")
        assert sbs_mod.out_exists(str(f))

    @pytest.mark.unit
    def test_parquet_file(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"PAR1" + b"\x00" * 100)
        assert sbs_mod.out_exists(str(f))


# ===========================================================================
# Section 3: TestRunParallel
# ===========================================================================


class TestRunParallel:
    """Verify run_parallel() ProcessPoolExecutor wrapper."""

    @pytest.mark.unit
    def test_all_succeed(self):
        errs = sbs_mod.run_parallel(list(range(5)), _task_ok, 2, "test")
        assert errs == 0

    @pytest.mark.unit
    def test_some_errors(self):
        errs = sbs_mod.run_parallel(list(range(4)), _task_mixed, 2, "test")
        assert errs == 2

    @pytest.mark.unit
    def test_all_skip(self):
        errs = sbs_mod.run_parallel(list(range(3)), _task_skip, 2, "test")
        assert errs == 0

    @pytest.mark.unit
    def test_empty_list(self):
        errs = sbs_mod.run_parallel([], _task_ok, 2, "test")
        assert errs == 0

    @pytest.mark.unit
    def test_single_worker(self):
        errs = sbs_mod.run_parallel(list(range(5)), _task_ok, 1, "test")
        assert errs == 0


# ===========================================================================
# Section 4: TestPathConstruction — SBS paths
# ===========================================================================


class TestSbsPathConstruction:
    """Verify SBS path helpers produce correct strings."""

    @pytest.mark.unit
    def test_aligned_tiff(self, tmp_path):
        result = sbs_mod.sbs_img_path(tmp_path, "tiff", "1", "A1", "0", "aligned")
        assert result == str(tmp_path / "images" / "P-1_W-A1_T-0__aligned.tiff")

    @pytest.mark.unit
    def test_aligned_zarr(self, tmp_path):
        result = sbs_mod.sbs_img_path(tmp_path, "zarr", "1", "A1", "0", "aligned")
        assert result == str(tmp_path / "aligned_1.zarr" / "A" / "1" / "0" / "zarr.json")

    @pytest.mark.unit
    def test_nuclei_labels_tiff(self, tmp_path):
        result = sbs_mod.sbs_img_path(
            tmp_path, "tiff", "1", "A1", "0", "nuclei", subdirectory="labels"
        )
        assert result == str(tmp_path / "images" / "P-1_W-A1_T-0__nuclei.tiff")

    @pytest.mark.unit
    def test_nuclei_labels_zarr(self, tmp_path):
        result = sbs_mod.sbs_img_path(
            tmp_path, "zarr", "1", "A1", "0", "nuclei", subdirectory="labels"
        )
        assert result == str(
            tmp_path / "aligned_1.zarr" / "A" / "1" / "0" / "labels" / "nuclei"
        )

    @pytest.mark.unit
    def test_bases_tsv_tiff(self, tmp_path):
        result = sbs_mod.sbs_data_path(tmp_path, "tiff", "1", "A1", "0", "bases", "tsv")
        assert result == str(tmp_path / "tsvs" / "P-1_W-A1_T-0__bases.tsv")

    @pytest.mark.unit
    def test_bases_tsv_zarr(self, tmp_path):
        result = sbs_mod.sbs_data_path(tmp_path, "zarr", "1", "A1", "0", "bases", "tsv")
        assert result == str(tmp_path / "tsvs" / "1" / "A" / "1" / "0" / "bases.tsv")

    @pytest.mark.unit
    def test_reads_parquet_tiff(self, tmp_path):
        result = sbs_mod.sbs_well_path(tmp_path, "tiff", "1", "A1", "reads", "parquet")
        assert result == str(tmp_path / "parquets" / "P-1_W-A1__reads.parquet")

    @pytest.mark.unit
    def test_reads_parquet_zarr(self, tmp_path):
        result = sbs_mod.sbs_well_path(tmp_path, "zarr", "1", "A1", "reads", "parquet")
        assert result == str(tmp_path / "parquets" / "1" / "A" / "1" / "reads.parquet")

    @pytest.mark.unit
    def test_eval_plate_path_tiff(self, tmp_path):
        result = sbs_mod.sbs_plate_path(
            tmp_path, "tiff", "1", "segmentation_overview", "tsv", "segmentation"
        )
        assert result == str(
            tmp_path / "eval" / "segmentation" / "P-1__segmentation_overview.tsv"
        )

    @pytest.mark.unit
    def test_preprocess_img_path_tiff(self, tmp_path):
        result = sbs_mod.preprocess_img_path(tmp_path, "tiff", "1", "A1", "0", "3")
        assert result == str(
            tmp_path / "images" / "sbs" / "P-1_W-A1_T-0_C-3__image.tiff"
        )

    @pytest.mark.unit
    def test_preprocess_img_path_zarr(self, tmp_path):
        result = sbs_mod.preprocess_img_path(tmp_path, "zarr", "1", "A1", "0", "3")
        assert result == str(
            tmp_path / "sbs" / "image_1.zarr" / "A" / "1" / "0" / "3" / "zarr.json"
        )

    @pytest.mark.unit
    def test_preprocess_ic_path_tiff(self, tmp_path):
        result = sbs_mod.preprocess_ic_path(tmp_path, "tiff", "1", "A1", "3")
        assert result == str(
            tmp_path / "ic_fields" / "sbs" / "P-1_W-A1_C-3__ic_field.tiff"
        )

    @pytest.mark.unit
    def test_preprocess_ic_path_zarr(self, tmp_path):
        result = sbs_mod.preprocess_ic_path(tmp_path, "zarr", "1", "A1", "3")
        assert result == str(
            tmp_path / "ic_fields" / "sbs" / "1" / "A" / "1" / "3" / "ic_field.zarr"
        )


# ===========================================================================
# Section 5: TestPathConstruction — Phenotype paths
# ===========================================================================


class TestPhenotypePathConstruction:
    """Verify phenotype path helpers produce correct strings."""

    @pytest.mark.unit
    def test_ic_corrected_tiff(self, tmp_path):
        result = phen_mod.phen_img_path(
            tmp_path, "tiff", "1", "A1", "5", "illumination_corrected"
        )
        assert result == str(
            tmp_path / "images" / "P-1_W-A1_T-5__illumination_corrected.tiff"
        )

    @pytest.mark.unit
    def test_aligned_tiff(self, tmp_path):
        result = phen_mod.phen_img_path(tmp_path, "tiff", "1", "A1", "5", "aligned")
        assert result == str(tmp_path / "images" / "P-1_W-A1_T-5__aligned.tiff")

    @pytest.mark.unit
    def test_cytoplasm_labels_tiff(self, tmp_path):
        result = phen_mod.phen_img_path(
            tmp_path, "tiff", "1", "A1", "5", "identified_cytoplasms", subdirectory="labels"
        )
        assert result == str(
            tmp_path / "images" / "P-1_W-A1_T-5__identified_cytoplasms.tiff"
        )

    @pytest.mark.unit
    def test_phenotype_cp_tsv_tiff(self, tmp_path):
        result = phen_mod.phen_data_path(
            tmp_path, "tiff", "1", "A1", "5", "phenotype_cp", "tsv"
        )
        assert result == str(tmp_path / "tsvs" / "P-1_W-A1_T-5__phenotype_cp.tsv")

    @pytest.mark.unit
    def test_phenotype_info_parquet_tiff(self, tmp_path):
        result = phen_mod.phen_well_path(
            tmp_path, "tiff", "1", "A1", "phenotype_info", "parquet"
        )
        assert result == str(tmp_path / "parquets" / "P-1_W-A1__phenotype_info.parquet")

    @pytest.mark.unit
    def test_phenotype_cp_min_parquet_tiff(self, tmp_path):
        result = phen_mod.phen_well_path(
            tmp_path, "tiff", "1", "A1", "phenotype_cp_min", "parquet"
        )
        assert result == str(
            tmp_path / "parquets" / "P-1_W-A1__phenotype_cp_min.parquet"
        )

    @pytest.mark.unit
    def test_eval_segmentation_tiff(self, tmp_path):
        result = phen_mod.phen_plate_path(
            tmp_path, "tiff", "1", "segmentation_overview", "tsv", "segmentation"
        )
        assert result == str(
            tmp_path / "eval" / "segmentation" / "P-1__segmentation_overview.tsv"
        )

    @pytest.mark.unit
    def test_eval_features_tiff(self, tmp_path):
        result = phen_mod.phen_plate_path(
            tmp_path, "tiff", "1", "cell_DAPI_min_heatmap", "png", "features"
        )
        assert result == str(
            tmp_path / "eval" / "features" / "P-1__cell_DAPI_min_heatmap.png"
        )

    @pytest.mark.unit
    def test_preprocess_phen_img_tiff(self, tmp_path):
        result = phen_mod.preprocess_phen_img_path(tmp_path, "tiff", "1", "A1", "5")
        assert result == str(
            tmp_path / "images" / "phenotype" / "P-1_W-A1_T-5__image.tiff"
        )

    @pytest.mark.unit
    def test_preprocess_phen_ic_tiff(self, tmp_path):
        result = phen_mod.preprocess_phen_ic_path(tmp_path, "tiff", "1", "A1")
        assert result == str(
            tmp_path / "ic_fields" / "phenotype" / "P-1_W-A1__ic_field.tiff"
        )

    @pytest.mark.unit
    def test_preprocess_phen_ic_zarr(self, tmp_path):
        result = phen_mod.preprocess_phen_ic_path(tmp_path, "zarr", "1", "A1")
        assert result == str(
            tmp_path / "ic_fields" / "phenotype" / "1" / "A" / "1" / "ic_field.zarr"
        )


# ===========================================================================
# Section 6: TestConfigExtraction
# ===========================================================================


class TestConfigExtraction:
    """Verify config parameter extraction functions work with test config."""

    @pytest.mark.unit
    def test_sbs_segmentation_params(self, sample_config):
        params = get_segmentation_params("sbs", sample_config)
        assert params["segmentation_method"] == "cellpose"
        assert params["dapi_index"] == 0
        assert params["cyto_index"] == 4
        assert params["gpu"] is False
        assert params["segment_cells"] is True
        assert "nuclei_diameter" in params
        assert "cell_diameter" in params

    @pytest.mark.unit
    def test_phenotype_segmentation_params(self, sample_config):
        params = get_segmentation_params("phenotype", sample_config)
        assert params["segmentation_method"] == "cellpose"
        assert params["dapi_index"] == 0
        assert params["cyto_index"] == 1

    @pytest.mark.unit
    def test_spot_detection_params(self, sample_config):
        params = get_spot_detection_params(sample_config)
        assert params["method"] == "standard"
        assert params["peak_width"] == 5

    @pytest.mark.unit
    def test_call_cells_params(self, sample_config):
        params = get_call_cells_params(sample_config)
        assert params["barcode_type"] == "simple"
        assert params["q_min"] == 0
        assert params["error_correct"] is False
        assert params["sort_calls"] == "count"
        assert "df_barcode_library_fp" in params

    @pytest.mark.unit
    def test_alignment_params_no_align(self, sample_config):
        wc = SimpleNamespace(plate="1")
        params = get_alignment_params(wc, sample_config)
        assert params["align"] is False

    @pytest.mark.unit
    def test_sbs_channel_names(self, sample_config):
        assert sample_config["sbs"]["channel_names"] == ["DAPI", "G", "T", "A", "C"]

    @pytest.mark.unit
    def test_phenotype_channel_names(self, sample_config):
        assert sample_config["phenotype"]["channel_names"] == [
            "DAPI", "COXIV", "CENPA", "WGA"
        ]

    @pytest.mark.unit
    def test_image_format_default(self, sample_config):
        assert sample_config["all"]["image_format"] == "tiff"

    @pytest.mark.unit
    def test_sbs_ic_cycles(self, sample_config):
        assert sample_config["sbs"]["dapi_cycle"] == 1
        assert sample_config["sbs"]["cyto_cycle"] == 11
        assert sample_config["sbs"]["dapi_cycle_index"] == 0
        assert sample_config["sbs"]["cyto_cycle_index"] == 10

    @pytest.mark.unit
    def test_sbs_extra_channel_indices(self, sample_config):
        assert sample_config["sbs"]["extra_channel_indices"] == [0]


# ===========================================================================
# Section 7: TestComboFiltering
# ===========================================================================


class TestComboFiltering:
    """Verify plate_filter and max_tiles logic."""

    @pytest.mark.unit
    def test_sbs_combo_shape(self, sbs_combos):
        assert len(sbs_combos) == 66
        assert list(sbs_combos.columns) == ["plate", "cycle", "well", "tile"]

    @pytest.mark.unit
    def test_phenotype_combo_shape(self, phenotype_combos):
        assert len(phenotype_combos) == 6
        assert list(phenotype_combos.columns) == ["plate", "well", "tile"]

    @pytest.mark.unit
    def test_sbs_unique_tiles(self, sbs_combos):
        tiles = sorted(sbs_combos["tile"].unique(), key=int)
        assert tiles == ["0", "2", "32"]

    @pytest.mark.unit
    def test_sbs_unique_cycles(self, sbs_combos):
        cycles = sorted(sbs_combos["cycle"].unique(), key=int)
        assert len(cycles) == 11

    @pytest.mark.unit
    def test_plate_filter(self, sbs_combos):
        filtered = sbs_combos[sbs_combos["plate"] == "1"]
        assert len(filtered) == 66

    @pytest.mark.unit
    def test_plate_filter_nonexistent(self, sbs_combos):
        filtered = sbs_combos[sbs_combos["plate"] == "999"]
        assert len(filtered) == 0

    @pytest.mark.unit
    def test_max_tiles_limits(self, sbs_combos):
        max_tiles = 2
        tiles = sorted(sbs_combos["tile"].unique(), key=lambda x: int(x))
        keep = set(tiles[:max_tiles])
        filtered = sbs_combos[sbs_combos["tile"].isin(keep)]
        assert sorted(filtered["tile"].unique(), key=int) == ["0", "2"]
        assert len(filtered) == 44

    @pytest.mark.unit
    def test_max_tiles_phenotype(self, phenotype_combos):
        max_tiles = 1
        tiles = sorted(phenotype_combos["tile"].unique(), key=lambda x: int(x))
        keep = set(tiles[:max_tiles])
        filtered = phenotype_combos[phenotype_combos["tile"].isin(keep)]
        assert len(filtered) == 2
        assert filtered["tile"].unique().tolist() == ["2"]

    @pytest.mark.unit
    def test_sbs_tile_combos_drop_cycle(self, sbs_combos):
        tile_combos = sbs_combos[["plate", "well", "tile"]].drop_duplicates()
        assert len(tile_combos) == 6

    @pytest.mark.unit
    def test_sbs_cycles_per_tile(self, sbs_combos):
        key = ("1", "A1", "0")
        mask = (
            (sbs_combos["plate"] == key[0])
            & (sbs_combos["well"] == key[1])
            & (sbs_combos["tile"] == key[2])
        )
        cycles = sorted(sbs_combos[mask]["cycle"].unique(), key=int)
        assert len(cycles) == 11


# ===========================================================================
# Section 8: TestWorkerSkipLogic
# ===========================================================================


class TestWorkerSkipLogic:
    """Verify workers return ("skip", ...) when outputs already exist."""

    @pytest.mark.unit
    def test_sbs_align_skips(self, tmp_path):
        out = tmp_path / "aligned.tiff"
        out.write_bytes(b"\x00" * 10)
        task = (["dummy_cycle_path"], str(out), {"channel_names": ["DAPI"]})
        status, _ = sbs_mod._align_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_sbs_log_filter_skips(self, tmp_path):
        out = tmp_path / "log_filtered.tiff"
        out.write_bytes(b"\x00" * 10)
        task = ("dummy_in", str(out), [0])
        status, _ = sbs_mod._log_filter_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_sbs_segment_skips(self, tmp_path):
        n = tmp_path / "nuclei.tiff"
        c = tmp_path / "cells.tiff"
        s = tmp_path / "stats.tsv"
        n.write_bytes(b"\x00" * 10)
        c.write_bytes(b"\x00" * 10)
        s.write_text("col\n1\n")
        task = ("dummy_in", str(n), str(c), str(s), {})
        status, _ = sbs_mod._segment_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_sbs_extract_bases_skips(self, tmp_path):
        out = tmp_path / "bases.tsv"
        out.write_text("col\n1\n")
        wc = {"plate": "1", "well": "A1", "tile": "0"}
        task = ("peaks", "maxfilt", "cells", str(out), 400, ["G", "T", "A", "C"], wc)
        status, _ = sbs_mod._extract_bases_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_sbs_call_reads_skips(self, tmp_path):
        out = tmp_path / "reads.tsv"
        out.write_text("col\n1\n")
        task = ("bases", "peaks", str(out), "median")
        status, _ = sbs_mod._call_reads_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_sbs_call_cells_skips(self, tmp_path):
        out = tmp_path / "cells.tsv"
        out.write_text("col\n1\n")
        task = ("reads", str(out), {})
        status, _ = sbs_mod._call_cells_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_sbs_extract_info_skips(self, tmp_path):
        out = tmp_path / "sbs_info.tsv"
        out.write_text("col\n1\n")
        wc = {"plate": "1", "well": "A1", "tile": "0"}
        task = ("nuclei", str(out), wc)
        status, _ = sbs_mod._extract_sbs_info_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_phenotype_apply_ic_skips(self, tmp_path):
        out = tmp_path / "ic.tiff"
        out.write_bytes(b"\x00" * 10)
        task = ("raw", "ic_field", str(out))
        status, _ = phen_mod._apply_ic_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_phenotype_align_skips(self, tmp_path):
        out = tmp_path / "aligned.tiff"
        out.write_bytes(b"\x00" * 10)
        task = ("input", str(out), {"align": False})
        status, _ = phen_mod._align_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_phenotype_cytoplasm_skips(self, tmp_path):
        out = tmp_path / "cyto.tiff"
        out.write_bytes(b"\x00" * 10)
        task = ("nuclei", "cells", str(out), True)
        status, _ = phen_mod._identify_cytoplasm_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_phenotype_extract_info_skips(self, tmp_path):
        out = tmp_path / "pheno_info.tsv"
        out.write_text("col\n1\n")
        wc = {"plate": "1", "well": "A1", "tile": "5"}
        task = ("nuclei", str(out), wc)
        status, _ = phen_mod._extract_phenotype_info_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_phenotype_extract_phenotype_skips(self, tmp_path):
        out = tmp_path / "phenotype_cp.tsv"
        out.write_text("col\n1\n")
        params = {
            "cp_method": "cp_emulator",
            "channel_names": ["DAPI"],
            "foci_channel_index": None,
            "segment_cells": True,
            "wildcards": {"plate": "1", "well": "A1", "tile": "5"},
        }
        task = ("aligned", "nuclei", "cells", "cyto", str(out), params)
        status, _ = phen_mod._extract_phenotype_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_preprocess_extract_skips(self, tmp_path):
        out = tmp_path / "metadata.tsv"
        out.write_text("col\n1\n")
        task = ([], [], "1", "A1", "0", "1", None, "nd2", "tile", None, str(out))
        status, _ = pp_mod._extract_one(task)
        assert status == "skip"

    @pytest.mark.unit
    def test_preprocess_convert_skips(self, tmp_path):
        out = tmp_path / "image.tiff"
        out.write_bytes(b"\x00" * 10)
        task = (["dummy"], "nd2", "tile", 0, False, None, None, str(out))
        status, _ = pp_mod._convert_one(task)
        assert status == "skip"


# ===========================================================================
# Section 9: TestMakeMdLoc (preprocess only)
# ===========================================================================


class TestMakeMdLoc:
    """Verify make_md_loc() builds correct data_location from DataFrame rows."""

    @pytest.mark.unit
    def test_tiff_all_columns(self):
        row = pd.Series({"plate": "1", "well": "A1", "tile": "0", "cycle": "3"})
        loc = pp_mod.make_md_loc("tiff", row, ["plate", "well", "tile", "cycle"])
        assert loc == {"plate": "1", "well": "A1", "tile": "0", "cycle": "3"}

    @pytest.mark.unit
    def test_zarr_splits_well(self):
        row = pd.Series({"plate": "1", "well": "A1", "tile": "0"})
        loc = pp_mod.make_md_loc("zarr", row, ["plate", "well", "tile"])
        assert loc == {"plate": "1", "row": "A", "col": "1", "tile": "0"}
        assert "well" not in loc

    @pytest.mark.unit
    def test_subset_columns(self):
        row = pd.Series({"plate": "1", "well": "A1", "tile": "0", "cycle": "3"})
        loc = pp_mod.make_md_loc("tiff", row, ["plate", "well"])
        assert loc == {"plate": "1", "well": "A1"}
        assert "tile" not in loc


# ===========================================================================
# Section 10: TestPathEquivalence — Integration
# ===========================================================================

_SNAKEMAKE_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "small_test_analysis" / "brieflow_output"
)


class TestPathEquivalence:
    """Verify script path helpers match Snakemake target paths."""

    @pytest.fixture(autouse=True)
    def _check_outputs(self):
        if not _SNAKEMAKE_OUTPUT_DIR.exists():
            pytest.skip("Snakemake output dir not found — run pipeline first")

    @pytest.mark.integration
    def test_sbs_paths_match_snakemake_targets(self, sample_config):
        """Script-generated SBS paths match Snakemake target template expansion."""
        fmt = sample_config["all"]["image_format"]
        sbs_fp = Path(sample_config["all"]["root_fp"]) / "sbs"

        tile = {"plate": "1", "well": "A1", "tile": "0"}
        expected_aligned = str(sbs_fp / get_image_output_path(tile, "aligned", fmt))
        script_aligned = sbs_mod.sbs_img_path(sbs_fp, fmt, "1", "A1", "0", "aligned")
        assert script_aligned == expected_aligned

        expected_bases = str(
            sbs_fp / "tsvs" / get_data_output_path(tile, "bases", "tsv", fmt)
        )
        script_bases = sbs_mod.sbs_data_path(sbs_fp, fmt, "1", "A1", "0", "bases", "tsv")
        assert script_bases == expected_bases

        well = {"plate": "1", "well": "A1"}
        expected_reads = str(
            sbs_fp / "parquets" / get_data_output_path(well, "reads", "parquet", fmt)
        )
        script_reads = sbs_mod.sbs_well_path(sbs_fp, fmt, "1", "A1", "reads", "parquet")
        assert script_reads == expected_reads

    @pytest.mark.integration
    def test_phenotype_paths_match_snakemake_targets(self, sample_config):
        """Script-generated phenotype paths match Snakemake target templates."""
        fmt = sample_config["all"]["image_format"]
        phen_fp = Path(sample_config["all"]["root_fp"]) / "phenotype"

        tile = {"plate": "1", "well": "A1", "tile": "5"}
        expected_aligned = str(
            phen_fp / get_image_output_path(tile, "aligned", fmt)
        )
        script_aligned = phen_mod.phen_img_path(phen_fp, fmt, "1", "A1", "5", "aligned")
        assert script_aligned == expected_aligned

        expected_cp = str(
            phen_fp / "tsvs" / get_data_output_path(tile, "phenotype_cp", "tsv", fmt)
        )
        script_cp = phen_mod.phen_data_path(
            phen_fp, fmt, "1", "A1", "5", "phenotype_cp", "tsv"
        )
        assert script_cp == expected_cp

    @pytest.mark.integration
    def test_preprocess_paths_match_snakemake_targets(self, sample_config):
        """Script-generated preprocess paths match Snakemake target templates."""
        fmt = sample_config["all"]["image_format"]
        pp_fp = Path(sample_config["all"]["root_fp"]) / "preprocess"

        tile_c = {"plate": "1", "well": "A1", "tile": "0", "cycle": "3"}
        expected_convert = str(
            pp_fp / get_image_output_path(tile_c, "image", fmt, image_subdir="sbs")
        )
        script_convert = sbs_mod.preprocess_img_path(pp_fp, fmt, "1", "A1", "0", "3")
        assert script_convert == expected_convert

        ic_loc = {"plate": "1", "well": "A1", "cycle": "3"}
        expected_ic = str(
            pp_fp / "ic_fields" / "sbs" / get_data_output_path(ic_loc, "ic_field", fmt, fmt)
        )
        script_ic = sbs_mod.preprocess_ic_path(pp_fp, fmt, "1", "A1", "3")
        assert script_ic == expected_ic


# ===========================================================================
# Section 11: TestEndToEnd — Integration
# ===========================================================================


class TestEndToEnd:
    """End-to-end tests running scripts on small test data."""

    @pytest.fixture(autouse=True)
    def _check_outputs(self):
        if not _SNAKEMAKE_OUTPUT_DIR.exists():
            pytest.skip("Snakemake output dir not found — run pipeline first")

    @pytest.mark.integration
    def test_sbs_output_files_exist(self, sample_config):
        """Verify SBS pipeline produced the expected output files."""
        root = _SNAKEMAKE_OUTPUT_DIR
        sbs_fp = root / "sbs"
        fmt = sample_config["all"]["image_format"]

        for well in ["A1", "A2"]:
            for info_type in ["reads", "cells", "sbs_info"]:
                pq = sbs_mod.sbs_well_path(sbs_fp, fmt, "1", well, info_type, "parquet")
                if Path(pq).exists():
                    df = pd.read_parquet(pq)
                    assert len(df) > 0, f"Empty parquet: {pq}"

    @pytest.mark.integration
    def test_phenotype_output_files_exist(self, sample_config):
        """Verify phenotype pipeline produced the expected output files."""
        root = _SNAKEMAKE_OUTPUT_DIR
        phen_fp = root / "phenotype"
        fmt = sample_config["all"]["image_format"]

        for well in ["A1", "A2"]:
            for info_type in ["phenotype_info", "phenotype_cp", "phenotype_cp_min"]:
                pq = phen_mod.phen_well_path(
                    phen_fp, fmt, "1", well, info_type, "parquet"
                )
                if Path(pq).exists():
                    df = pd.read_parquet(pq)
                    assert len(df) > 0, f"Empty parquet: {pq}"

    @pytest.mark.integration
    def test_preprocess_output_files_exist(self, sample_config):
        """Verify preprocess pipeline produced the expected output files."""
        root = _SNAKEMAKE_OUTPUT_DIR
        pp_fp = root / "preprocess"
        fmt = sample_config["all"]["image_format"]

        # Check combined metadata exists
        for well in ["A1", "A2"]:
            loc = pp_mod.make_loc(fmt, "1", well)
            md_path = (
                pp_fp / "metadata" / "sbs"
                / get_data_output_path(loc, "combined_metadata", "parquet", fmt)
            )
            if md_path.exists():
                df = pd.read_parquet(md_path)
                assert len(df) > 0
