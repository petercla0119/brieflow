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
def test_merge_output_stable():
    out_dir = _resolve_output_dir()
    merge_dir = out_dir / "merge"

    fast_merge_files = list(merge_dir.glob("parquets/**/*__fast_merge.parquet"))
    if not fast_merge_files:
        pytest.skip("No fast_merge parquet outputs found; run the merge module first.")

    df = read_parquet(fast_merge_files[0])
    # Core columns that the vectorized groupby refactor must preserve
    required = {"cell_0", "cell_1", "distance", "tile", "site"}
    missing = required - set(df.columns)
    assert not missing, f"fast_merge output missing columns: {missing}"
