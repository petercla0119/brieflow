"""Unit tests for HCS plate metadata functions in workflow/lib/shared/hcs.py."""

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_WORKFLOW = str(_REPO_ROOT / "workflow")
if _WORKFLOW not in sys.path:
    sys.path.insert(0, _WORKFLOW)

from workflow.lib.shared.hcs import discover_plate_structure, write_hcs_metadata


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fov(plate_path: Path, row: str, col: str, tile: str):
    """Create a minimal FOV zarr.json so discover_plate_structure finds it."""
    fov_dir = plate_path / row / col / tile
    fov_dir.mkdir(parents=True, exist_ok=True)
    (fov_dir / "zarr.json").write_text(json.dumps({
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
    }))


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverPlateStructure:
    """discover_plate_structure locates FOVs by scanning zarr.json files."""

    def test_finds_fovs(self, tmp_path):
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")
        _make_fov(plate, "A", "1", "1")
        _make_fov(plate, "B", "2", "0")

        result = discover_plate_structure(plate)
        assert set(result) == {("A", "1", "0"), ("A", "1", "1"), ("B", "2", "0")}

    def test_empty_plate(self, tmp_path):
        plate = tmp_path / "empty.zarr"
        plate.mkdir()
        assert discover_plate_structure(plate) == []


class TestWriteHcsMetadata:
    """write_hcs_metadata creates plate/row/well zarr.json files."""

    def test_creates_plate_row_well_json(self, tmp_path):
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")

        write_hcs_metadata(plate)

        assert (plate / "zarr.json").exists()
        assert (plate / "A" / "zarr.json").exists()
        assert (plate / "A" / "1" / "zarr.json").exists()

    def test_plate_metadata_content(self, tmp_path):
        plate = tmp_path / "aligned_1.zarr"
        _make_fov(plate, "A", "1", "0")
        _make_fov(plate, "A", "1", "1")

        write_hcs_metadata(plate)

        meta = _read_json(plate / "zarr.json")
        assert meta["zarr_format"] == 3
        assert meta["node_type"] == "group"

        ome = meta["attributes"]["ome"]
        assert ome["version"] == "0.5"

        p = ome["plate"]
        assert p["field_count"] == 2
        assert p["rows"] == [{"name": "A"}]
        assert p["columns"] == [{"name": "1"}]
        assert len(p["wells"]) == 1
        assert p["wells"][0]["path"] == "A/1"

    def test_well_metadata_content(self, tmp_path):
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")
        _make_fov(plate, "A", "1", "2")

        write_hcs_metadata(plate)

        well_meta = _read_json(plate / "A" / "1" / "zarr.json")
        images = well_meta["attributes"]["ome"]["well"]["images"]
        paths = [img["path"] for img in images]
        assert "0" in paths
        assert "2" in paths

    def test_row_metadata_is_minimal_group(self, tmp_path):
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")

        write_hcs_metadata(plate)

        row_meta = _read_json(plate / "A" / "zarr.json")
        assert row_meta["zarr_format"] == 3
        assert row_meta["node_type"] == "group"
        assert row_meta["attributes"] == {}

    def test_multiple_wells(self, tmp_path):
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")
        _make_fov(plate, "A", "2", "0")
        _make_fov(plate, "B", "1", "0")

        write_hcs_metadata(plate)

        meta = _read_json(plate / "zarr.json")
        wells = meta["attributes"]["ome"]["plate"]["wells"]
        well_paths = {w["path"] for w in wells}
        assert well_paths == {"A/1", "A/2", "B/1"}

    def test_empty_plate_no_crash(self, tmp_path):
        plate = tmp_path / "empty.zarr"
        plate.mkdir()

        # Should not raise, should not write plate metadata
        write_hcs_metadata(plate)
        assert not (plate / "zarr.json").exists()

    def test_with_channels_metadata(self, tmp_path):
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")

        channels_metadata = [
            {"name": "DAPI", "description": "nuclear stain"},
            {"name": "GFP", "description": "reporter"},
        ]
        write_hcs_metadata(plate, channels_metadata=channels_metadata)

        meta = _read_json(plate / "zarr.json")
        cm = meta["attributes"]["channels_metadata"]
        assert len(cm) == 2
        assert cm[0]["name"] == "DAPI"
        assert cm[1]["name"] == "GFP"

    def test_nonexistent_plate_raises(self, tmp_path):
        """Failure case: plate directory does not exist."""
        missing = tmp_path / "does_not_exist.zarr"
        with pytest.raises(FileNotFoundError):
            write_hcs_metadata(missing)

    def test_single_well_single_tile_field_count(self, tmp_path):
        """Edge case: single well with one tile produces field_count=1."""
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")

        write_hcs_metadata(plate)

        meta = _read_json(plate / "zarr.json")
        assert meta["attributes"]["ome"]["plate"]["field_count"] == 1

    def test_field_count_is_max_across_wells(self, tmp_path):
        """Edge case: wells with different FOV counts; field_count = max."""
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")
        _make_fov(plate, "A", "1", "1")
        _make_fov(plate, "A", "1", "2")
        _make_fov(plate, "B", "1", "0")  # only 1 FOV

        write_hcs_metadata(plate)

        meta = _read_json(plate / "zarr.json")
        assert meta["attributes"]["ome"]["plate"]["field_count"] == 3

    def test_no_channels_metadata_omits_key(self, tmp_path):
        """When channels_metadata is None, the key should not appear."""
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")

        write_hcs_metadata(plate, channels_metadata=None)

        meta = _read_json(plate / "zarr.json")
        assert "channels_metadata" not in meta["attributes"]

    def test_full_hcs_hierarchy_valid(self, tmp_path):
        """End-to-end: write_hcs_metadata produces a hierarchy that passes HCS plate validation."""
        plate = tmp_path / "aligned_1.zarr"
        _make_fov(plate, "A", "1", "0")
        _make_fov(plate, "A", "1", "1")
        _make_fov(plate, "A", "2", "0")
        _make_fov(plate, "B", "1", "0")

        write_hcs_metadata(plate)

        meta = _read_json(plate / "zarr.json")
        ome = meta["attributes"]["ome"]
        assert ome["version"] == "0.5"
        p = ome["plate"]
        assert set(r["name"] for r in p["rows"]) == {"A", "B"}
        assert set(c["name"] for c in p["columns"]) == {"1", "2"}
        well_paths = {w["path"] for w in p["wells"]}
        assert well_paths == {"A/1", "A/2", "B/1"}
        assert p["field_count"] == 2

        for w in p["wells"]:
            row_json = plate / w["path"].split("/")[0] / "zarr.json"
            assert row_json.exists()
            rm = _read_json(row_json)
            assert rm["zarr_format"] == 3 and rm["node_type"] == "group"

            well_json = plate / w["path"] / "zarr.json"
            assert well_json.exists()
            wm = _read_json(well_json)
            assert "well" in wm["attributes"]["ome"]
            assert len(wm["attributes"]["ome"]["well"]["images"]) > 0

    def test_labels_metadata_written(self, tmp_path):
        """Edge case: field with a label store gets labels group metadata."""
        plate = tmp_path / "plate.zarr"
        _make_fov(plate, "A", "1", "0")

        # Create a fake label store inside the field
        label_dir = plate / "A" / "1" / "0" / "labels" / "nuclei"
        label_dir.mkdir(parents=True)
        (label_dir / "zarr.json").write_text(json.dumps({
            "zarr_format": 3,
            "node_type": "group",
            "attributes": {"ome": {"image-label": {"version": "0.5"}}},
        }))

        write_hcs_metadata(plate)

        labels_group = plate / "A" / "1" / "0" / "labels" / "zarr.json"
        assert labels_group.exists()
        labels_meta = _read_json(labels_group)
        assert "nuclei" in labels_meta["attributes"]["ome"]["labels"]


# ---------------------------------------------------------------------------
# Integration: verify HCS metadata across all pipeline module outputs
# ---------------------------------------------------------------------------

_TEST_ANALYSIS = Path(__file__).resolve().parent / "small_test_analysis"

_SBS_STORE_TYPES = [
    "aligned",
    "illumination_corrected",
    "log_filtered",
    "standard_deviation",
    "peaks",
    "max_filtered",
]

_PHENOTYPE_STORE_TYPES = [
    "aligned",
    "illumination_corrected",
]

_PREPROCESS_STORE_TYPES = [
    "image",
]


def _find_output_dir() -> Path:
    canonical = _TEST_ANALYSIS / "brieflow_output"
    if canonical.exists():
        return canonical
    candidates = sorted(
        [p for p in _TEST_ANALYSIS.iterdir()
         if p.is_dir() and p.name.startswith("brieflow_output")],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    for p in candidates:
        if (p / "sbs").exists() or (p / "preprocess").exists():
            return p
    pytest.skip("Brieflow output directory not found.")


def _assert_valid_hcs_plate(store_dir: Path):
    """Assert a zarr store has valid HCS plate/row/well metadata hierarchy."""
    plate_json = store_dir / "zarr.json"
    assert plate_json.exists(), f"Missing plate zarr.json: {store_dir}"
    meta = json.loads(plate_json.read_text())

    assert meta.get("zarr_format") == 3, f"Bad zarr_format in {plate_json}"
    assert meta.get("node_type") == "group", f"Bad node_type in {plate_json}"

    ome = meta.get("attributes", {}).get("ome", {})
    assert "plate" in ome, f"Missing plate metadata in {plate_json}"
    plate = ome["plate"]
    assert "rows" in plate and len(plate["rows"]) > 0, f"No rows in {plate_json}"
    assert "columns" in plate and len(plate["columns"]) > 0, f"No columns in {plate_json}"
    assert "wells" in plate and len(plate["wells"]) > 0, f"No wells in {plate_json}"
    assert "field_count" in plate and plate["field_count"] > 0, f"Bad field_count in {plate_json}"

    for well_entry in plate["wells"]:
        well_path = store_dir / well_entry["path"]
        row_path = well_path.parent

        row_json = row_path / "zarr.json"
        assert row_json.exists(), f"Missing row zarr.json: {row_json}"
        row_meta = json.loads(row_json.read_text())
        assert row_meta.get("zarr_format") == 3
        assert row_meta.get("node_type") == "group"

        well_json = well_path / "zarr.json"
        assert well_json.exists(), f"Missing well zarr.json: {well_json}"
        well_meta = json.loads(well_json.read_text())
        assert well_meta.get("zarr_format") == 3
        well_ome = well_meta.get("attributes", {}).get("ome", {})
        assert "well" in well_ome, f"Missing well metadata in {well_json}"
        images = well_ome["well"].get("images", [])
        assert len(images) > 0, f"No images listed in well: {well_json}"


@pytest.mark.integration
class TestSbsStoresHaveHcsMetadata:
    """Verify SBS zarr stores have full HCS hierarchy after pipeline run."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = _find_output_dir()
        self.sbs_dir = self.root / "sbs"
        if not self.sbs_dir.exists():
            pytest.skip("SBS output directory not found.")

    def test_sbs_stores_have_plate_metadata(self):
        found_any = False
        for store_type in _SBS_STORE_TYPES:
            for store_dir in sorted(self.sbs_dir.glob(f"{store_type}_*.zarr")):
                found_any = True
                _assert_valid_hcs_plate(store_dir)
        if not found_any:
            pytest.skip("No SBS zarr stores found.")


@pytest.mark.integration
class TestPhenotypeStoresHaveHcsMetadata:
    """Verify phenotype zarr stores have full HCS hierarchy after pipeline run."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = _find_output_dir()
        self.pheno_dir = self.root / "phenotype"
        if not self.pheno_dir.exists():
            pytest.skip("Phenotype output directory not found.")

    def test_phenotype_stores_have_plate_metadata(self):
        found_any = False
        for store_type in _PHENOTYPE_STORE_TYPES:
            for store_dir in sorted(self.pheno_dir.glob(f"{store_type}_*.zarr")):
                found_any = True
                _assert_valid_hcs_plate(store_dir)
        if not found_any:
            pytest.skip("No phenotype zarr stores found.")


@pytest.mark.integration
class TestPreprocessStoresHaveHcsMetadata:
    """Verify preprocess zarr stores have full HCS hierarchy after pipeline run."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = _find_output_dir()
        self.preprocess_dir = self.root / "preprocess"
        if not self.preprocess_dir.exists():
            pytest.skip("Preprocess output directory not found.")

    def test_preprocess_sbs_stores_have_plate_metadata(self):
        sbs_dir = self.preprocess_dir / "sbs"
        if not sbs_dir.exists():
            pytest.skip("Preprocess SBS output not found.")
        found_any = False
        for store_type in _PREPROCESS_STORE_TYPES:
            for store_dir in sorted(sbs_dir.glob(f"{store_type}_*.zarr")):
                found_any = True
                _assert_valid_hcs_plate(store_dir)
        if not found_any:
            pytest.skip("No preprocess SBS zarr stores found.")

    def test_preprocess_phenotype_stores_have_plate_metadata(self):
        pheno_dir = self.preprocess_dir / "phenotype"
        if not pheno_dir.exists():
            pytest.skip("Preprocess phenotype output not found.")
        found_any = False
        for store_type in _PREPROCESS_STORE_TYPES:
            for store_dir in sorted(pheno_dir.glob(f"{store_type}_*.zarr")):
                found_any = True
                _assert_valid_hcs_plate(store_dir)
        if not found_any:
            pytest.skip("No preprocess phenotype zarr stores found.")
