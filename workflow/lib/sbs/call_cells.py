##############################################
# call_cells.py – self‑contained utility for  #
# calling single‑cell barcodes and mapping    #
# them to a pool / design table.              #
##############################################

from __future__ import annotations

import warnings
from typing import List, Optional

import numpy as np
import pandas as pd
import Levenshtein  # type: ignore

# --------------------------------------------------------------------------
# Constant strings – change these to match your column names once and forget
# --------------------------------------------------------------------------
PREFIX = "prefix"  # internal helper (temporary column)
BARCODE = "barcode"  # raw barcode observed in a read

# positional annotations in the reads dataframe (coming from extract_bases)
WELL = "well"
TILE = "tile"
CELL = "cell"
READ = "read"
POSITION_I = "pos_i"
POSITION_J = "pos_j"
Q_MIN = "Q_min"  # quality column produced upstream

# columns we create while calling cells
BARCODE_0 = "cell_barcode_0"
BARCODE_1 = "cell_barcode_1"
BARCODE_COUNT_0 = "cell_barcode_count_0"
BARCODE_COUNT_1 = "cell_barcode_count_1"
BARCODE_COUNT = "barcode_count"

# optional UMI support (only used if df_UMI is passed)
UMI_0 = "UMI_0"
UMI_1 = "UMI_1"
UMI_COUNT_0 = "UMI_count_0"
UMI_COUNT_1 = "UMI_count_1"
UMI_COUNT = "UMI_count"

# --------------------------------------------------------------------------
# Main public function
# --------------------------------------------------------------------------

def call_cells(
    reads_data: pd.DataFrame,
    *,
    df_pool: Optional[pd.DataFrame] = None,
    barcode_col: str = "iBAR2_f7",  # name of the barcode column in df_pool
    gene_col: str = "gene",  # name of the gene column in df_pool
    info_cols: Optional[List[str]] = None,  # additional columns to merge (besides gene)
    q_min: int = 0,
    df_UMI: Optional[pd.DataFrame] = None,
    error_correct: bool = True,
    max_distance: int = 1,
    distance_metric: str = "hamming",
) -> pd.DataFrame:
    """Return a per‑cell dataframe with top barcodes + (optionally) gene info.

    Parameters
    ----------
    reads_data : DataFrame
        Output of *call_reads* / *extract_bases* with at least columns
        ["well", "tile", "cell", "barcode", "Q_min"].
    df_pool : DataFrame, optional
        Design / pool table that contains at least *barcode_col* and *gene_col*.
        If *None*, the function just returns counts of top barcodes per cell.
    barcode_col : str
        Column in *df_pool* that holds the reference barcode sequence used for
        mapping. Example: "iBAR2_f7".
    gene_col : str
        Column in *df_pool* that holds the gene symbol. Will be merged as
        "gene_0" / "gene_1".
    info_cols : list[str], optional
        Extra columns from *df_pool* to merge (e.g. sgRNA, gene_id …). The
        *gene_col* will always be included automatically.
    q_min : int
        Minimum Q_min to retain a read.
    df_UMI : DataFrame, optional
        If provided, will add UMI information per cell (same schema as reads).
    error_correct : bool
        Whether to apply barcode error correction (Hamming / Levenshtein).
    max_distance : int
        Max edit distance for correction (passed to *error_correct_reads*).
    distance_metric : {"hamming", "levenshtein"}
        Distance metric for error correction.

    Returns
    -------
    DataFrame
        One row per called cell with columns:
        * well, tile, cell
        * cell_barcode_[0/1] and counts
        * barcode_count (total reads per cell)
        * gene_[0/1] plus any *info_cols* provided.
    """

    if reads_data.empty:
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. basic QC / filtering
    # ------------------------------------------------------------------
    reads_filt = reads_data.loc[reads_data[Q_MIN] >= q_min].copy()

    # ------------------------------------------------------------------
    # 2. optional mapping against pool / design table
    # ------------------------------------------------------------------
    if df_pool is None:
        df_cells = _call_cells_no_ref(reads_filt)
    else:
        # ensure the reference table has the expected columns
        df_pool = df_pool.copy()
        if barcode_col not in df_pool.columns:
            raise KeyError(f"df_pool is missing column '{barcode_col}'")
        if gene_col not in df_pool.columns:
            raise KeyError(f"df_pool is missing column '{gene_col}'")

        # build PREFIX column matching the experimental barcode length
        prefix_len = len(reads_filt.iloc[0][BARCODE])
        df_pool[PREFIX] = df_pool[barcode_col].str.slice(0, prefix_len)

        # ensure we have the info columns list, always include gene_col
        if info_cols is None:
            info_cols = []
        info_cols = list(dict.fromkeys([gene_col] + info_cols))  # preserve order, dedupe

        df_cells = _call_cells_mapping(
            reads_filt,
            df_pool,
            prefix_col=PREFIX,
            barcode_info_cols=info_cols,
            error_correct=error_correct,
            max_distance=max_distance,
            distance_metric=distance_metric,
        )

    # ------------------------------------------------------------------
    # 3. optional UMI aggregation
    # ------------------------------------------------------------------
    if df_UMI is not None:
        df_cells = _add_UMIs(df_cells, df_UMI)

    return df_cells

# --------------------------------------------------------------------------
# Internal helpers – no‑ref branch
# --------------------------------------------------------------------------

def _call_cells_no_ref(df_reads: pd.DataFrame) -> pd.DataFrame:
    cols = [WELL, TILE, CELL]

    # count barcodes per cell, keep top 2
    s = (
        df_reads.drop_duplicates([WELL, TILE, READ])
        .groupby(cols)[BARCODE]
        .value_counts()
        .rename("count")
        .sort_values(ascending=False)
        .reset_index()
        .groupby(cols)
    )

    df_cells = (
        df_reads.join(
            s.nth(0)[cols + [BARCODE]].rename(columns={BARCODE: BARCODE_0}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(0)[cols + ["count"]].rename(columns={"count": BARCODE_COUNT_0}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[cols + [BARCODE]].rename(columns={BARCODE: BARCODE_1}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[cols + ["count"]].rename(columns={"count": BARCODE_COUNT_1}).set_index(cols),
            on=cols,
        )
        .join(s["count"].sum().rename(BARCODE_COUNT), on=cols)
        .assign(
            **{
                BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0).astype(int),
                BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0).astype(int),
            }
        )
        .drop_duplicates(cols)
        .drop([READ, BARCODE, POSITION_I, POSITION_J], axis=1, errors="ignore")
        .query("cell > 0")
    )
    return df_cells

# --------------------------------------------------------------------------
# Internal helpers – mapping branch
# --------------------------------------------------------------------------

def _call_cells_mapping(
    df_reads: pd.DataFrame,
    df_pool: pd.DataFrame,
    *,
    prefix_col: str,
    barcode_info_cols: List[str],
    error_correct: bool,
    max_distance: int,
    distance_metric: str,
) -> pd.DataFrame:
    # error‑correct reads barcode column against reference prefixes
    if error_correct:
        df_reads = df_reads.copy()
        df_reads.loc[:, BARCODE] = error_correct_reads(
            df_reads[BARCODE], df_pool[prefix_col], max_distance=max_distance, distance_metric=distance_metric
        )

    # flag mapped reads
    df_mapped = (
        pd.merge(
            df_reads,
            df_pool[[prefix_col]],
            how="left",
            left_on=BARCODE,
            right_on=prefix_col,
        )
        .assign(mapped=lambda x: x[prefix_col].notna())
        .drop(prefix_col, axis=1)
    )

    cols = [WELL, TILE, CELL]
    s = (
        df_mapped.drop_duplicates([WELL, TILE, READ])
        .groupby(cols + ["mapped"])[BARCODE]
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["mapped", "count"], ascending=False)
        .groupby(cols)
    )

    df_cells = (
        df_reads.join(
            s.nth(0)[cols + [BARCODE]].rename(columns={BARCODE: BARCODE_0}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(0)[cols + ["count"]].rename(columns={"count": BARCODE_COUNT_0}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[cols + [BARCODE]].rename(columns={BARCODE: BARCODE_1}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[cols + ["count"]].rename(columns={"count": BARCODE_COUNT_1}).set_index(cols),
            on=cols,
        )
        .join(s["count"].sum().rename(BARCODE_COUNT), on=cols)
        .assign(
            **{
                BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0).astype(int),
                BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0).astype(int),
            }
        )
        .drop_duplicates(cols)
        .drop([READ, BARCODE, POSITION_I, POSITION_J], axis=1, errors="ignore")
        .query("cell > 0")
    )

    # merge guide / gene info for each of the two barcodes
    for idx, bc_col in enumerate([BARCODE_0, BARCODE_1]):
        suffix = f"_{idx}"
        right_cols = [prefix_col] + barcode_info_cols
        missing = [c for c in barcode_info_cols if c not in df_pool.columns]
        if missing:
            warnings.warn(f"Adding missing columns {missing} to df_pool (filled with NaN)")
            for c in missing:
                df_pool[c] = np.nan

        df_cells = pd.merge(
            df_cells,
            df_pool[right_cols],
            how="left",
            left_on=bc_col,
            right_on=prefix_col,
            suffixes=("", "_drop"),
        ).drop(prefix_col, axis=1)
        df_cells = df_cells.rename({c: c + suffix for c in barcode_info_cols}, axis=1)

    return df_cells

# --------------------------------------------------------------------------
# UMI helper
# --------------------------------------------------------------------------

def _add_UMIs(df_cells: pd.DataFrame, df_UMI: pd.DataFrame) -> pd.DataFrame:
    cols = [WELL, TILE, CELL]
    s = (
        df_UMI.drop_duplicates([WELL, TILE, READ])
        .groupby(cols)[BARCODE]
        .value_counts()
        .rename("count")
        .sort_values(ascending=False)
        .reset_index()
        .groupby(cols)
    )

    df_cells_UMI = (
        df_UMI.join(
            s.nth(0)[cols + [BARCODE]].rename(columns={BARCODE: UMI_0}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(0)[cols + ["count"]].rename(columns={"count": UMI_COUNT_0}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[cols + [BARCODE]].rename(columns={BARCODE: UMI_1}).set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[cols + ["count"]].rename(columns={"count": UMI_COUNT_1}).set_index(cols),
            on=cols,
        )
        .join(s["count"].sum().rename(UMI_COUNT), on=cols)
        .assign(
            **{
                UMI_COUNT_0: lambda x: x[UMI_COUNT_0].fillna(0).astype(int),
                UMI_COUNT_1: lambda x: x[UMI_COUNT_1].fillna(0).astype(int),
            }
        )
        .drop_duplicates(cols)
        .drop([READ, BARCODE], axis=1, errors="ignore")
        .drop([POSITION_I, POSITION_J], axis=1, errors="ignore")
    )

    extra_cols = df_cells_UMI.columns.difference(df_cells.columns)
    return df_cells.merge(df_cells_UMI[list(extra_cols) + cols], on=cols, how="left")


def error_correct_reads(reads, reference, max_distance=1, distance_metric="hamming"):
    """Error correct reads against a reference set of barcodes.

    Compares each read to the reference set and corrects it to the closest unique reference
    if within the specified distance threshold.

    Args:
        reads (pd.Series): Series with reads for error correction
        reference (pd.Series): Series with reference sequences
        max_distance (int, optional): Maximum distance for correction. Correction is performed
            only if (1) one reference sequence is closest (no ties) and (2) that unique reference
            sequence is within this distance. Default is 2.
        distance_metric (str, optional): Distance metric to compare barcodes.
            Options are 'hamming' (default) and 'levenshtein'.

    Returns:
        pd.Series: Corrected reads
    """
    # Calculate distance from each read to each reference barcode
    dist_to_ref = barcode_distance_matrix(
        reads.to_list(),
        reference.to_list(),
        distance_metric=distance_metric,
    )

    # Find minimum distance to reference for each read
    min_dist_to_ref = dist_to_ref.min(axis=1)

    # Determine which reads have a unique closest match
    unique_dist = np.array(
        [
            np.sum(dist_to_ref[x] == min_dist_to_ref[x]) == 1
            for x in range(dist_to_ref.shape[0])
        ]
    )

    # Filter for reads that have a unique closest match within max_distance
    corrected_subset = (unique_dist) & (min_dist_to_ref <= max_distance)

    # Get the corrected barcodes for eligible reads
    corrected_barcodes = reference.loc[
        dist_to_ref[corrected_subset].argmin(axis=1)
    ].values

    # Create copy of reads and update only the ones that can be corrected
    corrected_reads = reads.copy()
    corrected_reads.loc[corrected_subset] = corrected_barcodes

    return corrected_reads


def barcode_distance_matrix(barcodes_1, barcodes_2=False, distance_metric="hamming"):
    """Calculate distances between two sets of barcodes.

    Creates a matrix of distances between all pairs of barcodes from two sets.
    If only one set is provided, computes self-distances.

    Args:
        barcodes_1 (list): First list of barcode sequences
        barcodes_2 (list or bool, optional): Second list of barcode sequences.
            If False, uses barcodes_1 for both sets. Default is False.
        distance_metric (str, optional): Type of distance to calculate.
            Options are 'hamming' or 'levenshtein'. Default is 'hamming'.
    Returns:
        numpy.ndarray: Matrix of distances between barcode pairs
    """
    import warnings

    # Define the distance function based on chosen metric
    if distance_metric == "hamming":
        distance = lambda i, j: Levenshtein.hamming(i, j)
    elif distance_metric == "levenshtein":
        distance = lambda i, j: Levenshtein.distance(i, j)
    else:
        warnings.warn(
            'distance_metric must be "hamming" or "levenshtein" - defaulting to "hamming"'
        )
        distance = lambda i, j: Levenshtein.hamming(i, j)

    # If second set not provided, use the first set
    if isinstance(barcodes_2, bool):
        barcodes_2 = barcodes_1

    # Create distance matrix for all barcode pairs
    bc_distance_matrix = np.zeros((len(barcodes_1), len(barcodes_2)))
    for a, i in enumerate(barcodes_1):
        for b, j in enumerate(barcodes_2):
            bc_distance_matrix[a, b] = distance(i, j)

    return bc_distance_matrix
