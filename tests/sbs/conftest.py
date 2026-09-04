import sys
from pathlib import Path

WORKFLOW = Path(__file__).resolve().parents[2] / "workflow"
if str(WORKFLOW) not in sys.path:
    sys.path.insert(0, str(WORKFLOW))

import pandas as pd
import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
BARCODE_LIB_FP = Path("/mnt/work/broad-analysis/broad-tdp-gws/analysis/config/barcode_library.tsv")

# Only use tiles where both golden files exist
_all_tiles = ["P-4_W-A1_T-0", "P-4_W-A1_T-50", "P-4_W-A1_T-100"]
INTEGRATION_TILES = [
    t for t in _all_tiles
    if (GOLDEN_DIR / f"{t}__reads.tsv").exists()
    and (GOLDEN_DIR / f"{t}__cells.tsv").exists()
]

# Matches production config.yml sbs section
MULTI_CC_KWARGS = dict(
    q_min=0.0,
    map_start=1,
    map_end=12,
    prefix_map="prefix_map",
    recomb_start=13,
    recomb_end=15,
    prefix_recomb="prefix_recomb",
    recomb_filter_col="Q_recomb",
    recomb_q_thresh=0.1,
    error_correct=True,
    sort_calls="peak",
    max_distance=1,
    n_barcodes=2,
    barcode_info_cols=None,
)


@pytest.fixture(scope="session")
def barcode_lib():
    return pd.read_csv(BARCODE_LIB_FP, sep="\t")
