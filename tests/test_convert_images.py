"""Validation for the bidirectional TIFF<->Zarr converter (workflow/scripts/convert_images.py).

Runs against the small-test trees at $BRIEFLOW_SMALL_TEST (default: the broad-cpu
path). Skips if the data is absent.

Policy (see CONVERTER_PLAN.md sec 6):
  * Path mapping: every planned dst must exist in the native other-format tree.
  * Lossless per-file: read(dst) == read(src), exactly, for every category. This is
    direction-agnostic and unaffected by IC-sampling nondeterminism, because it
    compares a converted file to *its own source*, not across the two native runs.
  * Raw pre-IC images (sbs/phenotype preprocess): additionally bit-exact vs the
    NATIVE other-format tree -- these are deterministic across runs.
  * IC fields / illumination_corrected / aligned / labels are NOT asserted bit-equal
    across the two native trees (separate runs, unseeded 5% IC tile sampling ->
    2-5% per-channel drift); only shape/dtype are checked against native.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

WF = Path(__file__).resolve().parents[1] / "workflow"
sys.path.insert(0, str(WF / "scripts"))
sys.path.insert(0, str(WF))
import convert_images as ci  # noqa: E402
from lib.shared.image_io import read_image  # noqa: E402

DATA = Path(os.environ.get(
    "BRIEFLOW_SMALL_TEST",
    "/mnt/work/broad-analysis/brieflow-small-test-zarr3/tests/small_test_analysis",
))
TIFF = DATA / "brieflow_output"
ZARR = DATA / "brieflow_output_zarr"

pytestmark = pytest.mark.skipif(
    not (TIFF.exists() and ZARR.exists()),
    reason=f"small-test trees not found under {DATA}",
)

RAW_CATEGORIES = {"sbs_pp", "pheno_pp"}


def _first_per_category(direction, src_root):
    """One (category, src, dst_rel, is_label) per category -- keeps tests fast."""
    seen = {}
    for cat in ci.ALL_CATEGORIES:
        for src, rel, is_label in ci.plan(direction, src_root, [cat]):
            seen[cat] = (cat, src, rel, is_label)
            break
    return list(seen.values())


# --- Path mapping: every planned dst exists in the native other-format tree ---

@pytest.mark.parametrize("direction,src,ref", [
    ("tiff2zarr", TIFF, ZARR),
    ("zarr2tiff", ZARR, TIFF),
])
def test_all_mappings_resolve_to_native(direction, src, ref):
    mappings = ci.plan(direction, src, ci.ALL_CATEGORIES)
    assert mappings, "no files discovered"
    missing = [rel for _s, rel, _l in mappings if not (ref / rel).exists()]
    assert not missing, f"{len(missing)} dst paths absent in native tree, e.g. {missing[:3]}"


def test_both_directions_symmetric():
    n_t2z = len(ci.plan("tiff2zarr", TIFF, ci.ALL_CATEGORIES))
    n_z2t = len(ci.plan("zarr2tiff", ZARR, ci.ALL_CATEGORIES))
    assert n_t2z == n_z2t, f"tiff2zarr={n_t2z} vs zarr2tiff={n_z2t}"


# --- Lossless per-file conversion (the core correctness guarantee) ------------

@pytest.mark.parametrize("direction,src_root", [
    ("tiff2zarr", TIFF),
    ("zarr2tiff", ZARR),
])
def test_lossless_roundtrip_per_category(direction, src_root, tmp_path):
    for cat, src, rel, is_label in _first_per_category(direction, src_root):
        ci.convert_one(src, tmp_path / rel, is_label)
        got = read_image(str(tmp_path / rel))
        want = read_image(str(src))
        assert got.shape == want.shape, f"{cat}: shape {got.shape} != {want.shape}"
        assert got.dtype == want.dtype, f"{cat}: dtype {got.dtype} != {want.dtype}"
        assert np.array_equal(got, want), f"{cat}: values differ after {direction}"


def test_label_store_not_clobbered(tmp_path):
    """Writing nested labels into an aligned store must not destroy the aligned array."""
    dst_root = tmp_path
    for cat in ["aligned", "nuclei", "cells", "cytoplasm"]:
        for src, rel, is_label in ci.plan("tiff2zarr", TIFF, [cat]):
            ci.convert_one(src, dst_root / rel, is_label)
            break
    aligned_rel = ci.plan("tiff2zarr", TIFF, ["aligned"])[0][1]  # ends in zarr.json
    store_group = (dst_root / aligned_rel).parent
    aligned = read_image(str(store_group))
    assert aligned.ndim == 3 and aligned.shape[0] >= 1, "aligned array lost after label writes"
    for lbl in ["nuclei", "cells", "identified_cytoplasms"]:
        mask = read_image(str(store_group / "labels" / lbl))
        assert mask.dtype == np.uint32 and mask.ndim == 2


# --- Raw images: bit-exact vs the NATIVE other-format tree --------------------

@pytest.mark.parametrize("direction,src_root,ref_root", [
    ("tiff2zarr", TIFF, ZARR),
    ("zarr2tiff", ZARR, TIFF),
])
def test_raw_images_match_native(direction, src_root, ref_root, tmp_path):
    checked = 0
    for cat, src, rel, is_label in _first_per_category(direction, src_root):
        if cat not in RAW_CATEGORIES:
            continue
        ci.convert_one(src, tmp_path / rel, is_label)
        got = read_image(str(tmp_path / rel))
        native = read_image(str(ref_root / rel))
        assert np.array_equal(got, native), f"{cat}: converted != native ({direction})"
        checked += 1
    assert checked, "no raw categories checked"
