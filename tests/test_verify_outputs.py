"""Unit tests for tests/small_test_analysis/verify_outputs.py.

The verifier is what makes the CI legs mean something, so the failure cases
matter more than the pass case: a checker that cannot go red is decoration.
Runs on synthetic output trees -- no Zenodo download needed.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

VERIFY_FP = (
    Path(__file__).resolve().parent / "small_test_analysis" / "verify_outputs.py"
)
_spec = importlib.util.spec_from_file_location("verify_outputs", VERIFY_FP)
vo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vo)


# Flat TIFF naming: P-1_W-A1__name.parquet
FLAT_OUTPUTS = [
    "preprocess/metadata/sbs/P-1_W-A1__combined_metadata.parquet",
    "preprocess/metadata/phenotype/P-1_W-A1__combined_metadata.parquet",
    "sbs/parquets/P-1_W-A1__sbs_info.parquet",
    "sbs/parquets/P-1_W-A1__cells.parquet",
    "phenotype/parquets/P-1_W-A1__phenotype_info.parquet",
    "phenotype/parquets/P-1_W-A1__phenotype_cp.parquet",
    "merge/parquets/P-1_W-A1__merge_final.parquet",
]

# Nested zarr naming: {plate}/{row}/{col}/name.parquet
NESTED_OUTPUTS = [p.replace("P-1_W-A1__", "1/A/1/") for p in FLAT_OUTPUTS]


def write_parquet(path, rows=1):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"label": list(range(rows))}).to_parquet(path)


def write_tiff(root):
    tiff = root / "preprocess/images/sbs/P-1_W-A1_T-0__image.tiff"
    tiff.parent.mkdir(parents=True, exist_ok=True)
    tiff.write_bytes(b"II*\x00fake")


def write_zarr_store(root, hollow=False):
    """A minimal .zarr store. Hollow = metadata only, as an OOM leaves behind."""
    tile = root / "preprocess/sbs/image_1.zarr/A/1/0"
    tile.mkdir(parents=True, exist_ok=True)
    (tile / "zarr.json").write_text("{}")
    if not hollow:
        chunk = tile / "c" / "0" / "0"
        chunk.mkdir(parents=True, exist_ok=True)
        (chunk / "0").write_bytes(b"\x01\x02")


def build_tiff_root(tmp_path, outputs=FLAT_OUTPUTS):
    root = tmp_path / "brieflow_output"
    for rel in outputs:
        write_parquet(root / rel)
    write_tiff(root)
    return root


def test_complete_flat_run_passes(tmp_path):
    root = build_tiff_root(tmp_path)
    assert vo.check(root, vo.COMMON + vo.LEGS["tiff"][1]) == []


def test_nested_zarr_naming_also_matches(tmp_path):
    # One pattern set has to cover both layouts; nested naming must not read as
    # "no output".
    root = build_tiff_root(tmp_path, outputs=NESTED_OUTPUTS)
    assert vo.check(root, vo.COMMON + vo.LEGS["tiff"][1]) == []


def test_missing_output_fails(tmp_path):
    root = build_tiff_root(tmp_path)
    (root / "merge/parquets/P-1_W-A1__merge_final.parquet").unlink()
    failures = vo.check(root, vo.COMMON + vo.LEGS["tiff"][1])
    assert len(failures) == 1
    assert "merge_final" in failures[0]


def test_zero_row_parquet_fails(tmp_path):
    # The known failure mode: driver prints COMPLETE, exits 0, writes nothing
    # useful. A valid parquet with no rows must not count as an output.
    root = build_tiff_root(tmp_path)
    write_parquet(root / "phenotype/parquets/P-1_W-A1__phenotype_cp.parquet", rows=0)
    failures = vo.check(root, vo.COMMON + vo.LEGS["tiff"][1])
    assert len(failures) == 1
    assert "0 rows" in failures[0]


def test_zero_byte_file_fails(tmp_path):
    root = build_tiff_root(tmp_path)
    (root / "preprocess/images/sbs/P-1_W-A1_T-0__image.tiff").write_bytes(b"")
    failures = vo.check(root, vo.COMMON + vo.LEGS["tiff"][1])
    assert len(failures) == 1
    assert "zero bytes" in failures[0]


def test_zarr_leg_passes_with_chunks(tmp_path):
    root = tmp_path / "brieflow_output_zarr"
    for rel in NESTED_OUTPUTS:
        write_parquet(root / rel)
    write_zarr_store(root)
    assert vo.check(root, vo.COMMON + vo.LEGS["zarr"][1]) == []


def test_hollow_zarr_store_fails(tmp_path):
    root = tmp_path / "brieflow_output_zarr"
    for rel in NESTED_OUTPUTS:
        write_parquet(root / rel)
    write_zarr_store(root, hollow=True)
    failures = vo.check(root, vo.COMMON + vo.LEGS["zarr"][1])
    assert len(failures) == 1
    assert "hollow" in failures[0]


def test_zarr_leg_fails_on_a_tiff_only_run(tmp_path):
    # The whole point of the zarr leg: a run that wrote TIFFs must not pass it.
    root = build_tiff_root(tmp_path, outputs=NESTED_OUTPUTS)
    failures = vo.check(root, vo.COMMON + vo.LEGS["zarr"][1])
    assert any(".zarr" in f for f in failures)


def test_main_exits_nonzero_when_outputs_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "sys.argv", ["verify_outputs.py", "tiff", "--root", str(tmp_path / "nope")]
    )
    with pytest.raises(SystemExit) as exc:
        vo.main()
    assert exc.value.code != 0


def test_main_exits_zero_on_a_complete_run(tmp_path, monkeypatch):
    root = build_tiff_root(tmp_path)
    monkeypatch.setattr("sys.argv", ["verify_outputs.py", "tiff", "--root", str(root)])
    vo.main()  # no SystemExit == success
