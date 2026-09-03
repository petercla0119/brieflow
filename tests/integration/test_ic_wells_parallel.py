"""Integration: serial vs parallel IC must be byte-identical, plus resume + error isolation.

The correctness gate for well-group parallelism: running IC groups concurrently (each with
fewer threads) must produce byte-identical fields to the serial loop. calculate_ic_field is
documented bit-identical across n_jobs, so serial(n_jobs=total) == parallel(n_jobs=per_group).
Deterministic via sample_fraction=1.0 + ic_random_seed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "direct"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "workflow"))
import run_preprocess_direct as rpd  # noqa: E402
from lib.shared.file_utils import get_image_output_path  # noqa: E402
from lib.shared.image_io import read_image, save_image  # noqa: E402

PP = {"sample_fraction": 1.0, "ic_random_seed": 0}
WELLS = ("A1", "A2", "A3")


def _make_dataset(root, tiles=4, shape=(2, 32, 32)):
    """Write identical (seeded) input images under root and return the combos df."""
    rng = np.random.default_rng(0)
    rows = []
    for w in WELLS:
        for t in range(tiles):
            loc = rpd.make_loc("tiff", "1", w, str(t))
            out = root / get_image_output_path(
                loc, "image", "tiff", image_subdir="phenotype"
            )
            out.parent.mkdir(parents=True, exist_ok=True)
            save_image(rng.integers(0, 65535, shape, dtype=np.uint16), str(out))
            rows.append({"plate": "1", "well": w, "tile": str(t)})
    return pd.DataFrame(rows)


def _ic_files(root):
    return sorted((root / "ic_fields").rglob("*.tif*"))


def test_serial_and_parallel_byte_identical(tmp_path):
    ser, par = tmp_path / "serial", tmp_path / "parallel"
    combos_s = _make_dataset(ser)
    combos_p = _make_dataset(par)  # same rng seed -> identical inputs

    e1 = rpd.run_ic_step(
        "phenotype", combos_s, ser, "tiff", {**PP, "ic_group_concurrency": 1}, 4, False
    )
    e2 = rpd.run_ic_step(
        "phenotype", combos_p, par, "tiff", {**PP, "ic_group_concurrency": 3}, 4, False
    )
    assert e1 == 0 and e2 == 0

    sf, pf = _ic_files(ser), _ic_files(par)
    assert len(sf) == len(WELLS) and len(pf) == len(WELLS)
    for a, b in zip(sf, pf):
        assert a.relative_to(ser) == b.relative_to(par)
        assert np.array_equal(read_image(str(a)), read_image(str(b))), (
            f"IC differs: {a.name}"
        )


def test_resume_skips_existing(tmp_path):
    root = tmp_path / "r"
    combos = _make_dataset(root)
    assert (
        rpd.run_ic_step(
            "phenotype",
            combos,
            root,
            "tiff",
            {**PP, "ic_group_concurrency": 1},
            4,
            False,
        )
        == 0
    )
    before = {p: p.read_bytes() for p in _ic_files(root)}
    # second run (parallel) must skip everything and change nothing
    assert (
        rpd.run_ic_step(
            "phenotype",
            combos,
            root,
            "tiff",
            {**PP, "ic_group_concurrency": 3},
            4,
            False,
        )
        == 0
    )
    after = {p: p.read_bytes() for p in _ic_files(root)}
    assert before == after


def test_error_isolation_parallel(tmp_path):
    root = tmp_path / "e"
    combos = _make_dataset(root)
    # Break one well's inputs -> its group errors; the other two must still complete.
    for f in (root / "images" / "phenotype").rglob("*A2*"):
        f.unlink()
    errs = rpd.run_ic_step(
        "phenotype", combos, root, "tiff", {**PP, "ic_group_concurrency": 3}, 4, False
    )
    assert errs == 1
    written = {p.name for p in _ic_files(root)}
    assert not any("A2" in n for n in written)
    assert len(written) == 2
