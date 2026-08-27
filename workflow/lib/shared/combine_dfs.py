"""Combine per-tile TSVs into one dtype-normalized pandas DataFrame.

Shared by the Snakemake combine script (workflow/scripts/shared/combine_dfs.py)
and the direct runner (scripts/direct/run_sbs_direct.py --step combine) so the
concat + dtype-normalization logic lives in exactly one place.
"""

import pandas as pd

from lib.shared.file_utils import validate_dtypes

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def _read_concat(paths):
    """Fast multithreaded read + vertical concat of per-tile TSVs.

    Uses a single polars scan_csv over all paths (multithreaded, one concat).
    infer_schema_length=None scans full files so dtype inference matches
    pandas whole-file inference. Falls back to per-file pandas reads on any
    error (empty file, or schema mismatch across tiles) -- mirroring the
    original skip-empty / pd.concat-upcast behavior exactly.
    """
    if _HAS_POLARS:
        try:
            df = pl.read_csv(
                paths, separator="\t", infer_schema_length=None
            ).to_pandas()
            return df if not df.empty else None
        except Exception:
            pass  # fall through to per-file pandas (schema mismatch / empty file)

    dfs = []
    for f in paths:
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except Exception:
            pass  # skip empty / unreadable, as the original code did
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def combine_tile_dfs(paths):
    """Read per-tile TSVs, concat, and normalize dtypes.

    Returns a pandas DataFrame with the SAME dtype normalization the previous
    pandas-only implementation produced (validate_dtypes, then a 95%-threshold
    numeric coercion of any leftover object columns). Returns None if no input
    file yielded rows, so callers can skip writing.
    """
    combined = _read_concat(paths)
    if combined is None:
        return None

    combined = validate_dtypes(combined)
    # ponytail: validate_dtypes turns object cols into "string" dtype, so this
    # loop is a near no-op on real data (nothing left as object). Kept for
    # byte-identical parity with the previous two-pass implementation.
    for col in combined.select_dtypes(include="object").columns:
        converted = pd.to_numeric(combined[col], errors="coerce")
        if converted.notna().sum() >= combined[col].notna().sum() * 0.95:
            combined[col] = converted

    return combined
