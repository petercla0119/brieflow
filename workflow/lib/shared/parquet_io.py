"""Fast parquet I/O using polars with pandas compatibility.

Drop-in replacements for pd.read_parquet / df.to_parquet that use polars
under the hood for faster reads and writes. All functions accept and return
pandas DataFrames so downstream code is unchanged.

Usage:
    from lib.shared.parquet_io import read_parquet, write_parquet, read_parquets

    # Single file read (returns pandas DataFrame)
    df = read_parquet("data.parquet")
    df = read_parquet("data.parquet", columns=["gene", "feature_1"])

    # Multiple file concat (replaces pd.concat([pd.read_parquet(p) for p in paths]))
    df = read_parquets(paths)
    df = read_parquets(paths, columns=["gene", "feature_1"])

    # Write (accepts pandas DataFrame)
    write_parquet(df, "output.parquet")
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def read_parquet(
    path: Union[str, Path],
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read a parquet file, returning a pandas DataFrame.

    Uses polars lazy scan for speed when available, falls back to pandas.
    """
    if _HAS_POLARS:
        try:
            lf = pl.scan_parquet(path)
            if columns is not None:
                lf = lf.select(columns)
            return lf.collect().to_pandas()
        except (pl.exceptions.SchemaError, pl.exceptions.ComputeError):
            # Mixed types in columns --- fall back to pandas which is more lenient
            return pd.read_parquet(path, columns=columns)
    else:
        return pd.read_parquet(path, columns=columns)


def read_parquets(
    paths: Sequence[Union[str, Path]],
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read and concatenate multiple parquet files into one pandas DataFrame.

    Replaces the common pattern:
        pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    """
    if not paths:
        return pd.DataFrame()

    if _HAS_POLARS:
        try:
            lf = pl.scan_parquet(paths, missing_columns="insert")
            if columns is not None:
                lf = lf.select(columns)
            return lf.collect().to_pandas()
        except (pl.exceptions.SchemaError, pl.exceptions.ComputeError):
            # Schema/type mismatch across files --- fall back to per-file reads
            dfs = [read_parquet(p, columns=columns) for p in paths]
            return pd.concat(dfs, ignore_index=True)
    else:
        dfs = [pd.read_parquet(p, columns=columns) for p in paths]
        return pd.concat(dfs, ignore_index=True)


def resolve_table_path(path):
    """Resolve a per-tile intermediate path, preferring parquet over its tsv sibling.

    `path` may be given with either a .parquet or .tsv suffix. Checks the
    .parquet sibling first, then .tsv. A file counts only if it exists AND is
    non-empty (size > 0), matching the 0-byte skip semantics of the combine
    fallback. Returns (str_path, "parquet"|"tsv") or (None, None) if neither
    sibling has content.
    """
    base = Path(path)
    for suffix, fmt in ((".parquet", "parquet"), (".tsv", "tsv")):
        cand = base.with_suffix(suffix)
        if cand.exists() and cand.stat().st_size > 0:
            return str(cand), fmt
    return None, None


def read_table(path):
    """Read a per-tile intermediate as a pandas DataFrame.

    Prefers parquet over its tsv sibling (via resolve_table_path). Raises
    FileNotFoundError if neither sibling has content. parquet -> read_parquet();
    tsv -> pd.read_csv (sep tab).
    """
    resolved, fmt = resolve_table_path(path)
    if resolved is None:
        raise FileNotFoundError(f"No non-empty parquet or tsv sibling found for {path}")
    if fmt == "parquet":
        return read_parquet(resolved)
    return pd.read_csv(resolved, sep="\t")


def write_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """Write a pandas DataFrame to parquet.

    Uses polars for faster writes when available, falls back to pandas.
    """
    if _HAS_POLARS:
        # polars stringifies object columns whose cells are numpy arrays
        # (e.g. sbs_info "bounds", a per-cell bbox that round-trips out of a
        # parquet List column as an ndarray). Convert those to python lists so
        # polars infers a proper List type instead of str(ndarray), keeping the
        # combined table byte-consistent with the per-tile parquet files.
        obj_cols = df.select_dtypes(include="object").columns
        if len(obj_cols):
            df = df.copy()
            for c in obj_cols:
                nonnull = df[c].dropna()
                if len(nonnull) and isinstance(nonnull.iloc[0], np.ndarray):
                    df[c] = df[c].map(
                        lambda x: x.tolist() if isinstance(x, np.ndarray) else x
                    )
        pl.from_pandas(df).write_parquet(str(path))
    else:
        df.to_parquet(path, index=False)


def _float32_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema(
        [
            pa.field(f.name, pa.float32(), f.nullable)
            if pa.types.is_float64(f.type)
            else f
            for f in schema
        ],
        metadata=schema.metadata,
    )


def common_columns(paths: Sequence[Union[str, Path]]) -> list[str]:
    """Column names present in every file, in the first file order (schema reads only)."""
    names = pq.read_schema(paths[0]).names
    keep = set(names)
    for path in paths[1:]:
        keep &= set(pq.read_schema(path).names)
    return [n for n in names if n in keep]


def columns_without_nulls(paths: Sequence[Union[str, Path]]) -> list[str]:
    """Columns with zero nulls across all files, from parquet statistics only (no data read).

    Equivalent to ``pd.concat(frames).dropna(axis=1).columns`` for files whose writer
    recorded null counts; a column with missing statistics is kept. Per-well NaN drops can leave
    files with different columns, so only the shared columns are considered.
    """
    has_null = set()
    for path in paths:
        md = pq.read_metadata(path)
        for rg in range(md.num_row_groups):
            for i in range(md.num_columns):
                col = md.row_group(rg).column(i)
                stats = col.statistics
                if stats is not None and stats.null_count:
                    has_null.add(col.path_in_schema)
    return [name for name in common_columns(paths) if name not in has_null]


def read_parquet_frames(
    paths: Sequence[Union[str, Path]],
    columns: Optional[list[str]] = None,
    float32: bool = True,
):
    """Yield ``(path, DataFrame)`` for each non-empty file, one file in memory at a time.

    float64 columns are cast to float32 in Arrow before the pandas conversion, so the
    frame never exists at double width. Use instead of ``Dataset.scanner().take`` for
    batching: ``take`` decodes the whole dataset whatever the index count.
    """
    for path in paths:
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows == 0:
            continue
        table = pf.read(columns=columns)
        if float32:
            table = table.cast(_float32_schema(table.schema))
        yield path, table.to_pandas()


def sample_parquet_rows(
    paths: Sequence[Union[str, Path]],
    n_rows: int,
    seed: Optional[int] = 0,
    columns: Optional[list[str]] = None,
    float32: bool = True,
) -> pd.DataFrame:
    """Uniform random sample of ``n_rows`` across files, reading one file at a time."""
    paths = [p for p in paths if pq.read_metadata(p).num_rows > 0]
    sizes = np.array([pq.read_metadata(p).num_rows for p in paths])
    total = int(sizes.sum())
    n_rows = min(n_rows, total)
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(total, size=n_rows, replace=False))
    offsets = np.concatenate([[0], np.cumsum(sizes)])
    frames = []
    for path, start, end in zip(paths, offsets[:-1], offsets[1:]):
        local = picked[(picked >= start) & (picked < end)] - start
        if len(local) == 0:
            continue
        table = pq.read_table(path, columns=columns).take(pa.array(local))
        if float32:
            table = table.cast(_float32_schema(table.schema))
        frames.append(table.to_pandas())
    return pd.concat(frames, ignore_index=True)
