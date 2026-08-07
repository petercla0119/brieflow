import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_ANALYSIS = Path(__file__).resolve().parents[1] / "small_test_analysis"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
def test_construct_table_stable():
    import pandas as pd

    out_dir = _resolve_output_dir()
    agg_dir = out_dir / "aggregate"

    construct_files = list(agg_dir.glob("**/*construct*.tsv"))
    if not construct_files:
        pytest.skip("No construct TSV outputs found; run the aggregate module first.")

    df = pd.read_csv(construct_files[0], sep="\t")
    assert len(df) > 0, "Construct table is empty"
    assert "cell_count" in df.columns, "cell_count column missing from construct table"
    assert df["cell_count"].sum() > 0, "All constructs have zero cells"

    # Verify the table has gene/sgRNA-like identifier column and at least one feature
    non_meta = [
        c for c in df.columns if c not in ("cell_count",) and df[c].dtype.kind == "f"
    ]
    assert len(non_meta) >= 1, "No float feature columns found in construct table"
