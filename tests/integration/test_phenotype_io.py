import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_ANALYSIS = Path(__file__).resolve().parents[1] / "small_test_analysis"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.shared.parquet_io import read_parquet


def _resolve_output_dir() -> Path:
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


@pytest.mark.integration
def test_extract_phenotype_output_is_parquet():
    out_dir = _resolve_output_dir()
    phenotype_dir = out_dir / "phenotype"

    per_tile_parquets = list(phenotype_dir.glob("parquets/**/*__phenotype_cp.parquet"))
    stale_tsvs = list(phenotype_dir.glob("tsvs/**/*__phenotype_cp.tsv"))

    assert len(per_tile_parquets) >= 1, (
        "No per-tile phenotype_cp.parquet found under phenotype/parquets/; "
        "TSV→parquet migration may not have run."
    )
    assert len(stale_tsvs) == 0, (
        f"Stale per-tile phenotype_cp.tsv files still present: {stale_tsvs}"
    )

    sample = read_parquet(per_tile_parquets[0])
    assert "label" in sample.columns
    assert any(col.endswith("_min") for col in sample.columns), (
        "Expected at least one *_min intensity column in per-tile parquet"
    )
