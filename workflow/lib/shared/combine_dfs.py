"""Combine per-tile intermediates (parquet-or-tsv, prefer parquet) into one
dtype-normalized pandas DataFrame.

Shared by the Snakemake combine script (workflow/scripts/shared/combine_dfs.py)
and the direct runner (scripts/direct/run_sbs_direct.py --step combine) so the
concat + dtype-normalization logic lives in exactly one place.

NOTE on float parity: polars parses CSV floats with correct round-to-nearest
(matching pandas float_precision="round_trip" and Python float()). pandas'
DEFAULT read_csv parser is imprecise (xstrtod), so the fast path differs from
the old pandas-default output by up to ~5e-13 on high-precision float columns
(e.g. sbs_info i/j, cells Q_*). Values are numerically equivalent, NOT byte
identical. See _READ_STATS to confirm which backend ran.

DECISION (Phase 1, Option A, 2026-08-28): we ACCEPT this ~1e-12 difference.
polars is the more-correct parser; downstream analysis is unaffected. Validated
both ways on real plate-4 well A1: round_trip==polars EXACT, and default-vs-
polars max |diff| 1.8e-12 (reads) / 9.1e-13 (cells) / 2.3e-13 (sbs_info), all
<< rtol=1e-9. Identity tests compare float columns with np.allclose(rtol=1e-9)
and everything else byte-exactly (see tests/sbs/test_combine_dfs_opt.py).

Phase 2 (2026-08-29): inputs are now parquet-or-tsv per tile (prefer parquet
via resolve_table_path). Parquet round-trips floats exactly, so parquet-mode
combine is EXACT to the in-memory frame and stays within Option A vs the
TSV-mode golden. Production TSV-only wells still combine unchanged.
"""

import os

import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import resolve_table_path, read_parquet

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

# Backend counter so tests can assert the polars fast path actually ran.
_READ_STATS = {"polars": 0, "pandas_fallback": 0}


def _header_cols(f):
    with open(f) as fh:
        return set(fh.readline().rstrip("\n").split("\t"))


def _cols(f, fmt):
    """Column set for a resolved tile file, by format (for the union-missing cast)."""
    if fmt == "parquet":
        return set(pl.scan_parquet(f).collect_schema().names())
    return _header_cols(f)


def _read_concat(paths):
    """Fast multithreaded read + vertical (union) concat of per-tile tables.

    Each path is resolved to its parquet-else-tsv sibling; scan_parquet/scan_csv
    is chosen per file. Uses one polars diagonal_relaxed concat over per-file
    lazy scans, which
    reproduces the pandas concat column UNION (missing column -> null). 0-byte
    files are pre-filtered (pandas skips them via EmptyDataError; scan_csv can
    choke on them); header-only 0-row files are kept so their columns still
    join the union, matching pandas. Any polars error (schema/parse) falls
    back to per-file pandas reads + concat. Returns None only when no file
    yields a frame (mirrors the old `if not dfs` guard).
    """
    resolved = []
    for p_in in paths:
        f, fmt = resolve_table_path(p_in)
        if f is not None:
            resolved.append((f, fmt))

    if _HAS_POLARS and resolved:
        try:
            lfs = [
                pl.scan_parquet(f)
                if fmt == "parquet"
                else pl.scan_csv(f, separator="\t", infer_schema_length=None)
                for f, fmt in resolved
            ]
            df = pl.concat(lfs, how="diagonal_relaxed").collect()
            # pandas concat reindexes frames missing a column, upcasting it to
            # float(nullable); validate_dtypes then lands integer-valued ones on
            # Int64. Replicate: cast integer union-missing columns to Float64 so
            # the fast path yields the same dtype as the pandas path.
            # _cols is one footer/header read per file; hoist it so the
            # union scan stays O(files), not O(files x columns).
            file_cols = [_cols(f, fmt) for f, fmt in resolved]
            union_missing = {
                c for s in file_cols for c in df.columns if c not in s
            }
            casts = [
                pl.col(c).cast(pl.Float64)
                for c in union_missing
                if df.schema[c].is_integer()
            ]
            if casts:
                df = df.with_columns(casts)
            _READ_STATS["polars"] += 1
            return df.to_pandas()
        except Exception:
            pass  # schema/parse error -> pandas fallback below

    dfs = []
    for f, fmt in resolved:
        if fmt == "parquet":
            dfs.append(read_parquet(f))  # resolver already filtered 0-byte
            continue
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except pd.errors.EmptyDataError:
            pass  # skip empty tiles ONLY, matching the original get_file semantics.
            # Any other parse error (truncated/corrupt tile) must propagate and
            # fail the rule loud -- silently dropping a tile is data loss.
    if not dfs:
        return None
    _READ_STATS["pandas_fallback"] += 1
    return pd.concat(dfs, ignore_index=True)


def combine_tile_dfs(paths):
    """Read per-tile TSVs, concat, and normalize dtypes.

    Returns a pandas DataFrame with the same dtype normalization the previous
    pandas-only implementation produced (validate_dtypes, then a 95%-threshold
    numeric coercion of leftover object columns -- kept because it genuinely
    fires on real data, e.g. cells no_recomb_*/gene_id_*). Returns None if no
    input file yielded a frame, so callers can skip writing.
    """
    combined = _read_concat(paths)
    if combined is None:
        return None

    combined = validate_dtypes(combined)
    for col in combined.select_dtypes(include="object").columns:
        converted = pd.to_numeric(combined[col], errors="coerce")
        if converted.notna().sum() >= combined[col].notna().sum() * 0.95:
            combined[col] = converted

    if os.environ.get("COMBINE_DFS_DEBUG"):
        import sys

        print(f"[combine_dfs] backends={_READ_STATS}", file=sys.stderr)

    return combined
