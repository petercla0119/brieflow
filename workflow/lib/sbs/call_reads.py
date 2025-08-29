"""Read-Calling Utilities — NovaSeq X three-colour edition
=========================================================

Converts raw spot intensities from Illumina NovaSeq X chemistry
(Cy3 / Far-red / GFP channels) into base calls, quality scores and barcodes.

Channel → base logic
--------------------
    * Cy3 + GFP                 → C
    * GFP only                  → A
    * Cy3 only                  → T
    * (dark) no channel on      → G
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Path to the joblib bundle you saved after training
_THIS_DIR   = Path(__file__).resolve().parent   # folder containing this script
_MODEL_PATH = _THIS_DIR / "segmented_bc.joblib"   # always resolves correctly

# project-level constants
# Assuming 'lib.sbs.constants' exists in the project structure
# For standalone execution, these would need to be defined directly.
# Example: WELL, TILE, ... = "well", "tile", ...
from lib.sbs.constants import (
    WELL,
    TILE,
    CELL,
    READ,
    CHANNEL,
    CYCLE,
    BARCODE,
    INTENSITY,
)

# ------------------------------------------------------------------
# 3-COLOUR CHEMISTRY CONSTANTS
# ------------------------------------------------------------------
CHANNEL_ORDER = ["Cy3", "Far-red", "GFP"]  # fixed order
N_CHANNELS = len(CHANNEL_ORDER)            # == 3
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# SAVE TENSOR FOR TRAINING
# ------------------------------------------------------------------
def save_corrected_tensor(
    df_bases: pd.DataFrame,
    Y: np.ndarray,
    out_csv: str | Path,
    channel_names: list[str] = CHANNEL_ORDER,
):
    """
    Flatten corrected tensor Y back to long format and save.
    """
    meta_cols = ["read", "cycle", "cell", "i", "j", "tile", "well"]

    print("→ saving with channel_names =", channel_names)
    print("→ Y has shape", Y.shape)

    # *** keep one row per (read, cycle) only ***
    meta = df_bases[meta_cols].drop_duplicates().reset_index(drop=True)

    assert len(meta) == Y.shape[0], (
        f"meta rows ({len(meta)}) ≠ Y rows ({Y.shape[0]}). "
        "Did you forget drop_duplicates?"
    )

    long = []
    for ch_idx, ch_name in enumerate(channel_names):
        chunk = meta.copy()
        chunk["channel"]   = ch_name
        chunk["intensity"] = Y[:, ch_idx]
        long.append(chunk)

    pd.concat(long, ignore_index=True).to_csv(out_csv, index=False)
    print(f"✓ wrote corrected intensities → {out_csv}")


# ------------------------------------------------------------------
#  TOP-LEVEL DRIVER
# ------------------------------------------------------------------
def call_reads(
    bases_data: pd.DataFrame,
    peaks_data: np.ndarray | None = None,
    correction_only_in_cells: bool = True,
    normalize_bases_first: bool = False,
    method: str = "median",
) -> pd.DataFrame:
    """Return a df_reads table with base calls and Q-scores."""
    print("using: ", _MODEL_PATH)
    if bases_data.empty:
        cycles = bases_data["cycle"].nunique()
        base_cols = ["read", "cell", "i", "j", "tile", "well", "barcode"]
        q_cols = [f"Q_{i}" for i in range(cycles)]
        return pd.DataFrame(columns=base_cols + q_cols + ["Q_min", "peak"])

    if correction_only_in_cells and bases_data.query("cell > 0").empty:
        return pd.DataFrame()

    if method == "median":
        cleaned = bases_data.pipe(clean_up_bases)
        if normalize_bases_first:
            cleaned = cleaned.pipe(normalize_by_spot_sum)
        df_reads = cleaned.pipe(
            do_median_call,
            correction_only_in_cells=correction_only_in_cells,
        )
    elif method == "percentile":
        df_reads = bases_data.pipe(clean_up_bases).pipe(
            do_percentile_call,
            correction_only_in_cells=correction_only_in_cells,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    if peaks_data is not None:
        i, j = df_reads[["i", "j"]].values.T
        df_reads["peak"] = peaks_data[i, j]

    return df_reads


# ------------------------------------------------------------------
#  PRE-PROCESSING HELPERS
# ------------------------------------------------------------------
def clean_up_bases(df_bases: pd.DataFrame) -> pd.DataFrame:
    """Ensure deterministic ordering for downstream reshapes."""
    return df_bases.sort_values([WELL, TILE, CELL, READ, CYCLE, CHANNEL])

def normalize_by_spot_sum(df: pd.DataFrame) -> pd.DataFrame:
    """
    Divide every intensity by ( spot_sum / median_spot_sum ).

    After this transform the *median* total intensity per spot
    (Cy3 + Far-red + GFP) is exactly 1.0.
    """
    meta_cols = ["read", "cycle", "cell", "i", "j", "tile", "well"]

    # 1. total brightness per spot  (broadcasts back to every channel row)
    spot_sum = df.groupby(meta_cols)["intensity"].transform("sum")

    # 2. global median of those totals
    median_sum = spot_sum.median()
    if median_sum == 0:
        median_sum = 1.0                         # safety against all-zero input

    # 3. scaling factor for each row
    scale = spot_sum / median_sum               # ≈1 when spot is “typical”

    out = df.copy()
    
    out["intensity"] = df["intensity"] / scale  # element-wise divide
    print("initial:\n", df["intensity"].head())
    print("sum transformed:\n", out["intensity"].head())

    return out


def normalize_bases(df: pd.DataFrame) -> pd.DataFrame:
    """Median-normalise each fluorescence channel."""
    out = df.copy()
    # FIX: Replaced slow row-wise df.apply with vectorized df.transform,
    # which is significantly more performant on large datasets.
    channel_medians = df.groupby(CHANNEL)[INTENSITY].transform("median")
    out[INTENSITY] = df[INTENSITY] / channel_medians
    return out


def dataframe_to_values(df: pd.DataFrame, value: str = "intensity") -> np.ndarray:
    """Convert sorted DataFrame → tensor  (reads × cycles × channels)."""
    # sanity-check: every cycle should have the same row count
    rows_per_cycle = df.groupby(CYCLE)[READ].nunique()
    assert rows_per_cycle.nunique() == 1, (
        "Inconsistent reads per cycle: "
        f"{rows_per_cycle.to_dict()}"
    )

    n_cycles   = df[CYCLE].nunique()
    n_channels = df[CHANNEL].nunique()

    return np.asarray(df[value]).reshape(-1, n_cycles, n_channels)


# ------------------------------------------------------------------
#  COLOUR-MATRIX COMPENSATION
# ------------------------------------------------------------------
def transform_medians(X: np.ndarray, correction_quartile: float = 0):
    """Median-based colour unmixing."""

    def _medians(arr: np.ndarray):
        out = []
        for i in range(arr.shape[1]):
            max_spots = arr[arr.argmax(axis=1) == i]
            try:
                # Find threshold based on the brightest channel
                thr = np.quantile(max_spots[:, i], q=correction_quartile)
                # Filter spots above threshold before taking median
                bright_spots = max_spots[max_spots[:, i] >= thr]
                out.append(np.median(bright_spots, axis=0))
            except (IndexError, ValueError):
                # Fallback for empty or problematic slices
                out.append(np.median(max_spots, axis=0) if len(max_spots) > 0 else np.zeros(arr.shape[1]))
        return np.asarray(out)

    M = _medians(X).T
    # Avoid division by zero if a column sum is zero
    with np.errstate(divide='ignore', invalid='ignore'):
        M_norm = M / M.sum(axis=0, keepdims=True)
    M_norm = np.nan_to_num(M_norm, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Use pseudo-inverse for stability if matrix is singular
    try:
        W = np.linalg.inv(M_norm)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(M_norm)
    
    Y = (W @ X.T).T.astype(int)
    return Y, W


def transform_percentiles(X: np.ndarray):
    """95th-percentile unmixing."""

    def _perc(arr: np.ndarray):
        out = []
        for i in range(arr.shape[1]):
            with np.errstate(divide='ignore', invalid='ignore'):
                rowsums = arr.sum(axis=1, keepdims=True)
                rel = np.nan_to_num(arr / rowsums)
            
            # Use nanpercentile to handle potential NaNs gracefully
            perc_val = np.nanpercentile(rel[:, i], 95)
            high = arr[rel[:, i] >= perc_val]
            
            out.append(np.median(high, axis=0) if len(high) > 0 else np.zeros(arr.shape[1]))
        return np.asarray(out)

    M = _perc(X).T
    with np.errstate(divide='ignore', invalid='ignore'):
        M_norm = M / M.sum(axis=0, keepdims=True)
    M_norm = np.nan_to_num(M_norm, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        W = np.linalg.inv(M_norm)
    except np.linalg.LinAlgError:
        W = np.linalg.pinv(M_norm)

    Y = (W @ X.T).T.astype(int)
    return Y, W


# ------------------------------------------------------------------
#  READ-CALLING WRAPPERS
# ------------------------------------------------------------------
def do_median_call(
    df_bases: pd.DataFrame,
    correction_quartile: float = 0.0,
    correction_only_in_cells: bool = True,
    correction_by_cycle: bool = False,
) -> pd.DataFrame:
    """Colour-correct and call bases via median correction."""
    n_cycles = df_bases[CYCLE].nunique()
    
    # FIX: The original implementation of `correction_by_cycle` was logically flawed
    # and would scramble the data. This revised logic correctly processes the data
    # by reshaping to a 3D tensor, applying corrections cycle-by-cycle, and then
    # reshaping back. This ensures data integrity.
    if correction_by_cycle:
        X_all = dataframe_to_values(df_bases) # (reads, cycles, channels)
        Y_all_3d = np.zeros_like(X_all, dtype=int)

        for cyc_idx in range(n_cycles):
            X_cycle = X_all[:, cyc_idx, :]
            if correction_only_in_cells:
                # Get read indices that correspond to cells for this cycle
                cell_mask = df_bases[df_bases[CYCLE] == cyc_idx + 1][CELL] > 0
                X_cell_cycle = X_cycle[cell_mask.values, :]
                if X_cell_cycle.shape[0] == 0: # No cells in this cycle
                    _, W = np.eye(N_CHANNELS), np.eye(N_CHANNELS) 
                else:
                    _, W = transform_medians(X_cell_cycle, correction_quartile)
                Y_cycle = (W @ X_cycle.T).T.astype(int)
            else:
                Y_cycle, _ = transform_medians(X_cycle, correction_quartile)
            Y_all_3d[:, cyc_idx, :] = Y_cycle
        
        # Reshape back to 2D for downstream functions
        Y = Y_all_3d.reshape(-1, N_CHANNELS)
    else:
        # Original logic for whole-dataset correction
        if correction_only_in_cells:
            print('doing it in cells only')
            df_cell = df_bases.query("cell > 0")
            if df_cell.empty:
                Y = dataframe_to_values(df_bases).reshape(-1, N_CHANNELS)
            else:
                X_cell = dataframe_to_values(df_cell).reshape(-1, N_CHANNELS)
                _, W = transform_medians(X_cell, correction_quartile)
                X_all = dataframe_to_values(df_bases).reshape(-1, N_CHANNELS)
                Y = (W @ X_all.T).T.astype(int)
        else:
            X = dataframe_to_values(df_bases).reshape(-1, N_CHANNELS)
            Y, _ = transform_medians(X, correction_quartile)

    save_corrected_tensor(df_bases, Y, out_csv="corrected_intensities.csv")

    return call_barcodes(df_bases, Y)


def do_percentile_call(
    df_bases: pd.DataFrame,
    correction_only_in_cells: bool = False,
) -> pd.DataFrame:
    """Colour-correct and call bases via 95-percentile correction."""
    if correction_only_in_cells:
        df_cell = df_bases.query("cell > 0")
        if df_cell.empty: # Handle case with no cells
            Y = dataframe_to_values(df_bases).reshape(-1, N_CHANNELS)
        else:
            X_cell = dataframe_to_values(df_cell).reshape(-1, N_CHANNELS)
            _, W = transform_percentiles(X_cell)
            X_all = dataframe_to_values(df_bases).reshape(-1, N_CHANNELS)
            Y = (W @ X_all.T).T.astype(int)
    else:
        X = dataframe_to_values(df_bases).reshape(-1, N_CHANNELS)
        Y, _ = transform_percentiles(X)

    return call_barcodes(df_bases, Y)


# ------------------------------------------------------------------
#  3-COLOUR BASE-CALLER
# ------------------------------------------------------------------
_MODEL_CACHE = {} 

def _load_bundle():
    """Load the joblib bundle once and stash it in _MODEL_CACHE."""
    if _MODEL_CACHE:
        return _MODEL_CACHE

    bundle = joblib.load(_MODEL_PATH)
    print('using: ', _MODEL_PATH)
    # Prefer XGB if present, else RF
    if "xgb" in bundle:
        stuff = bundle["xgb"]
        model, thresholds, le = stuff["model"], stuff["thresholds"], stuff["label_encoder"]
        feat_names = stuff.get("feature_names")
        _MODEL_CACHE.update(model=model, thresholds=thresholds, encoder=le, use_le=True, feature_names=feat_names)
    elif "rf" in bundle:
        stuff = bundle["rf"]
        model, thresholds = stuff["model"], stuff["thresholds"]
        feat_names = stuff.get("feature_names")
        _MODEL_CACHE.update(model=model, thresholds=thresholds, encoder=None, use_le=False, feature_names=feat_names)
    else:
        raise RuntimeError("Bundle must contain 'rf' or 'xgb'!")
    
    # --------------- FINAL safety fallback ---------------------
    if _MODEL_CACHE["feature_names"] is None:
        if hasattr(model, "feature_names_in_"):            # scikit-learn
            _MODEL_CACHE["feature_names"] = list(model.feature_names_in_)
        else:                                              # XGB Booster
            _MODEL_CACHE["feature_names"] = model.get_booster().feature_names

    return _MODEL_CACHE


def _engineer_features(flat: np.ndarray,
                       gfp_idx: int, cy3_idx: int, far_idx: int) -> pd.DataFrame:
    """Feature block identical to what was used during training."""
    df = pd.DataFrame(
        {
            "GFP":      flat[:, gfp_idx],
            "Cy3":      flat[:, cy3_idx],
            "Far-red":  flat[:, far_idx],
        }
    )

    # log1p
    for c in ["GFP", "Cy3", "Far-red"]:
        df[f"log_{c}"] = np.log1p(df[c])

    # fractions
    total = df[["GFP", "Cy3", "Far-red"]].sum(axis=1).replace(0, 1)
    for c in ["GFP", "Cy3", "Far-red"]:
        df[f"frac_{c}"] = df[c] / total

    # log-ratios
    chans = ["GFP", "Cy3", "Far-red"]
    for i in range(3):
        for j in range(i + 1, 3):
            a, b = chans[i], chans[j]
            df[f"logratio_{a}_over_{b}"] = np.log1p(df[a]) - np.log1p(df[b])
            df[f"logratio_{b}_over_{a}"] = -df[f"logratio_{a}_over_{b}"]
    return df


def _apply_thresholds(proba_df: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    """Convert per-class probabilities → A/C/G/T/N calls."""
    # top_class = proba_df.idxmax(axis=1)
    # top_prob  = proba_df.max(axis=1)
    # calls = np.where(
    #     top_prob >= top_class.map(thresholds),  # boolean mask
    #     top_class,                              # keep best class
    #     "N"                                     # else no-call
    # )
    # return calls
    return proba_df.idxmax(axis=1).to_numpy()


def call_bases_3colour(
    values: np.ndarray,
    gfp_idx: int = 0,
    cy3_idx: int = 1,
    far_idx: int = 2,
) -> list[str]:
    """
    Translate Cy3 / Far-red / GFP intensities into A/C/G/T (or N) bases
    using the trained ML model stored in `basecaller.joblib`.
    """
    # 1) model & meta
    meta = _load_bundle()
    model, thresholds = meta["model"], meta["thresholds"]
    use_le, le = meta["use_le"], meta["encoder"]

    n_reads, n_cycles, _ = values.shape
    flat = values.reshape(-1, 3)                       # (reads*cycles, 3)

    # 2) feature-engineer
    X = _engineer_features(flat, gfp_idx, cy3_idx, far_idx)
    expected = meta["feature_names"]
    X = X.reindex(columns=expected) # Use reindex for safety

    # 3) predict probabilities
    if use_le:
        proba = model.predict_proba(X)
        classes = le.inverse_transform(np.arange(len(le.classes_)))
        proba_df = pd.DataFrame(proba, columns=classes)
    else:
        proba_df = pd.DataFrame(model.predict_proba(X), columns=model.classes_)

    # 4) thresholds → calls
    calls_flat = _apply_thresholds(proba_df, thresholds)

    # 5) reshape back to reads × cycles → join
    calls = calls_flat.reshape(n_reads, n_cycles).astype(str)
    return ["".join(seq) for seq in calls]


# ------------------------------------------------------------------
#  BARCODE ASSEMBLY + Q-SCORES
# ------------------------------------------------------------------
def call_barcodes(
    df_bases: pd.DataFrame,
    Y: np.ndarray,
) -> pd.DataFrame:
    """Assemble barcodes, attach quality metrics and metadata."""
    # Define the unique identifier for a read spot
    read_id_cols = [WELL, TILE, CELL, READ]
    
    # Create the base DataFrame for reads by dropping duplicates
    df_reads = df_bases.drop_duplicates(subset=read_id_cols).copy()
    
    n_cycles = df_bases[CYCLE].nunique()
    
    # Reshape corrected intensities for base calling and Q-score calculation
    Y_3d = Y.reshape(-1, n_cycles, N_CHANNELS)

    df_reads[BARCODE] = call_bases_3colour(
        Y_3d,
        cy3_idx=CHANNEL_ORDER.index("Cy3"),
        far_idx=CHANNEL_ORDER.index("Far-red"),
        gfp_idx=CHANNEL_ORDER.index("GFP"),
    )

    Q = quality(Y_3d)
    for i in range(Q.shape[1]):
        df_reads[f"Q_{i}"] = Q[:, i]
    df_reads["Q_min"] = df_reads.filter(regex=r"^Q_\d+$").min(axis=1)

    # Drop columns that vary by cycle or channel, which are now redundant
    return df_reads.drop(columns=[CYCLE, CHANNEL, INTENSITY], errors='ignore')


def quality(X: np.ndarray) -> np.ndarray:
    """
    Compute quality scores based on channel intensity separation.
    A higher score indicates a larger separation between the top two channels.
    """
    # Sort channel intensities for each spot at each cycle
    Xs = np.sort(X, axis=-1).astype(float)
    
    # Add a small epsilon to avoid division by zero or log(0) with zero intensities
    epsilon = 1e-9
    s1 = Xs[..., -1] + epsilon # Brightest channel
    s2 = Xs[..., -2] + epsilon # Second-brightest channel

    # FIX: The original formula `(2 * q).clip(0, 1)` discards all resolution for
    # confidence scores above 0.5. This corrected version returns the raw confidence
    # score `q` directly, which ranges from 0 (no separation) to 1 (high separation),
    # providing a much more informative metric.
    q = 1 - np.log2(2 + s2) / np.log2(2 + s1)
    
    # Return the confidence score, clipped to ensure it's within [0, 1]
    return q.clip(0, 1)


# ------------------------------------------------------------------
#  OPTIONAL PLOTTING (unchanged except for 3-colour compatibility)
# ------------------------------------------------------------------
def plot_normalization_comparison(
    df_bases: pd.DataFrame,
    channel_pairs: list[tuple[str, str]] | None = None,
    channel_order: list[str] = CHANNEL_ORDER,
    figsize: tuple[int, int] = (14, 18),
):
    """
    Diagnostic scatter-plots for raw vs colour-corrected intensities
    in 3-colour NovaSeq X data.

    Parameters
    ----------
    df_bases : tidy long-format DataFrame with `channel`, `intensity`, …
    channel_pairs : list of (ch1, ch2) tuples to plot.  Defaults to all
                    unique unordered pairs from `channel_order`.
    channel_order : order used for reshaping & axis labelling.
    """

    # ---------- set-up ----------
    if channel_pairs is None:
        # all unordered unique pairs
        channel_pairs = [
            (channel_order[i], channel_order[j])
            for i in range(len(channel_order))
            for j in range(i + 1, len(channel_order))
        ]

    n_rows = 1 + 2      # raw + (median-, percentile-corrected)
    n_cols = len(channel_pairs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                             squeeze=False)

    ch_to_idx = {c: i for i, c in enumerate(channel_order)}
    colour_map = dict(zip(channel_order, ["green", "red", "blue"]))  # tweak as preferred

    # ---------- pre-compute corrected tensors ----------
    df_sorted = clean_up_bases(df_bases)
    X_raw   = dataframe_to_values(df_sorted)                    # (reads, cycles, n_ch)
    X_norm  = dataframe_to_values(normalize_bases(df_sorted))   # median-per-channel normalised

    Y_med,  _ = transform_medians(X_norm.reshape(-1, X_norm.shape[-1]))
    Y_perc, _ = transform_percentiles(X_raw.reshape(-1, X_raw.shape[-1]))

    # reshape back to 2-D (rows align with df_sorted)
    Y_med  = Y_med.reshape(-1, *X_raw.shape[1:]).reshape(-1, len(channel_order))
    Y_perc = Y_perc.reshape(-1, *X_raw.shape[1:]).reshape(-1, len(channel_order))

    # DataFrame of raw intensities for convenience
    raw_tbl = df_sorted.pivot_table(
        index=["read", "cycle"],
        columns="channel",
        values="intensity",
        aggfunc="first",
    ).reset_index()

    # attach corrected values
    for idx, ch in enumerate(channel_order):
        raw_tbl[f"median_{ch}"]      = Y_med[:, idx]
        raw_tbl[f"percentile_{ch}"]  = Y_perc[:, idx]

    # identify brightest channel *within the pair* (for colouring)
    def brightest(row, chA, chB, prefix=""):
        return chA if row[f"{prefix}{chA}"] >= row[f"{prefix}{chB}"] else chB

    # ---------- plotting ----------
    for col_idx, (chA, chB) in enumerate(channel_pairs):
        # --- raw ---
        ax = axes[0, col_idx]
        raw_tbl["bright_raw"] = raw_tbl.apply(
            brightest, axis=1, chA=chA, chB=chB, prefix=""
        )
        for ch in (chA, chB):
            m = raw_tbl["bright_raw"] == ch
            ax.scatter(raw_tbl.loc[m, chB], raw_tbl.loc[m, chA],
                       s=10, alpha=0.5, color=colour_map[ch], label=ch)
        ax.set_title(f"{chA} vs {chB} — raw")
        ax.set_xlabel(chB); ax.set_ylabel(chA)

        # --- corrected: median / percentile ---
        for row_idx, meth in enumerate(("median_", "percentile_"), start=1):
            ax = axes[row_idx, col_idx]
            raw_tbl[f"bright_{meth}"] = raw_tbl.apply(
                brightest, axis=1, chA=chA, chB=chB, prefix=meth
            )
            for ch in (chA, chB):
                m = raw_tbl[f"bright_{meth}"] == ch
                ax.scatter(raw_tbl.loc[m, f"{meth}{chB}"],
                           raw_tbl.loc[m, f"{meth}{chA}"],
                           s=10, alpha=0.5, color=colour_map[ch])
            maxv = max(*ax.get_xlim(), *ax.get_ylim(), 1)
            ax.plot([0, maxv], [0, maxv], "k--", alpha=0.6)
            label = "median" if meth == "median_" else "percentile"
            ax.set_title(f"{chA} vs {chB} — {label}")
            ax.set_xlabel(chB); ax.set_ylabel(chA)

    # one legend for all plots
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=colour_map[ch], markersize=10,
                      label=ch) for ch in channel_order]
    handles.append(Line2D([0], [0], ls="--", color="k", label="y = x"))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, 0.02), ncol=len(handles))
    plt.tight_layout(); plt.subplots_adjust(bottom=0.12)
