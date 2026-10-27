"""Bootstrap statistical testing for perturbation effects.

Construct nulls come from the beta-quantile shortcut (no resampling, nothing persisted);
the gene null is the elementwise median across the gene's construct nulls. P-values are
two-tailed empirical; FDR correction is applied over all features at once.
"""

import numpy as np
import pandas as pd
import random
from typing import Tuple, List, Dict, Any, Optional, Union
from scipy.stats import false_discovery_control


def prepare_control_pools(
    controls_arr: np.ndarray,
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Split controls into per-control-id pools, each sorted per feature.

    Args:
        controls_arr (np.ndarray): (n_rows, 1 + n_features); column 0 is the control id.

    Returns:
        Tuple[List[np.ndarray], np.ndarray]: per-pool (n_pool, n_features) arrays sorted
            along axis 0, and the pool sizes (sampling weights).
    """
    control_ids = controls_arr[:, 0]
    features = controls_arr[:, 1:].astype(np.float64)
    pools = [
        np.sort(features[control_ids == cid], axis=0) for cid in np.unique(control_ids)
    ]
    sizes = np.array([len(pool) for pool in pools], dtype=np.int64)
    return pools, sizes


def beta_quantile_null(
    pools: List[np.ndarray],
    sizes: np.ndarray,
    sample_size: int,
    num_sims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Null medians without resampling, via the order-statistic identity.

    The median of ``sample_size`` i.i.d. draws from a pool's empirical distribution F is
    F^-1(U) with U ~ Beta((n+1)/2, (n+1)/2) (exact for odd n). One Beta draw plus one
    indexed lookup into the pre-sorted pool replaces an n-cell resample and a median.
    Pools are chosen with probability proportional to size, matching the resampler's
    "pick a random control row, use its pool" semantics.

    Args:
        pools (List[np.ndarray]): Per-pool features sorted along axis 0 (prepare_control_pools).
        sizes (np.ndarray): Pool sizes.
        sample_size (int): Cells per simulated construct.
        num_sims (int): Number of simulated medians.
        rng (np.random.Generator): Random generator.

    Returns:
        np.ndarray: (num_sims, n_features) null medians.
    """
    n_features = pools[0].shape[1]
    pool_idx = rng.choice(len(pools), size=num_sims, p=sizes / sizes.sum())
    a = (sample_size + 1) / 2
    u = rng.beta(a, a, size=(num_sims, n_features))
    null_medians = np.empty((num_sims, n_features), dtype=np.float64)
    for p_i in np.unique(pool_idx):
        rows = np.flatnonzero(pool_idx == p_i)
        pool = pools[p_i]
        k = np.minimum((u[rows] * len(pool)).astype(np.int64), len(pool) - 1)
        null_medians[rows] = np.take_along_axis(pool, k, axis=0)
    return null_medians


def create_pseudogene_groups(
    construct_features_df: pd.DataFrame,
    pseudogene_patterns: Dict[str, Dict[str, Union[str, int]]],
    perturbation_col: str,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """Group constructs into pseudo-genes based on patterns.

    Args:
        construct_features_df (pd.DataFrame): DataFrame containing construct features.
        pseudogene_patterns (Dict[str, Dict[str, Union[str, int]]]): Dictionary mapping
            category names to pattern configurations. Each configuration should contain
            'pattern' (str) and 'constructs_per_pseudogene' (int) keys.
        perturbation_col (str): Name of the column containing perturbation identifiers.
        seed (int, optional): Random seed for reproducible grouping. Defaults to 42.

    Returns:
        Tuple[List[Dict[str, Any]], pd.DataFrame]: A tuple containing:
            - List of pseudo-gene group dictionaries with 'pseudogene_id', 'category',
              and 'constructs' keys
            - DataFrame of remaining individual constructs not grouped into pseudo-genes
    """
    if not pseudogene_patterns:
        return [], construct_features_df

    print(f"Pseudogene patterns provided: {pseudogene_patterns}")
    print(f"Using random seed: {seed}")

    # Set random seed for reproducible grouping
    random.seed(seed)

    pseudogene_groups = []
    remaining_constructs = construct_features_df.copy()

    for category_name, config in pseudogene_patterns.items():
        pattern = config["pattern"]
        constructs_per_pseudogene = config["constructs_per_pseudogene"]

        print(f"Processing category '{category_name}' with pattern '{pattern}'")

        # Find constructs matching this pattern
        mask = remaining_constructs[perturbation_col].str.match(pattern, na=False)
        matching_constructs = remaining_constructs[mask]

        if len(matching_constructs) == 0:
            print(
                f"  No constructs found matching pattern '{pattern}' for category '{category_name}'"
            )
            continue

        print(
            f"  Found {len(matching_constructs)} constructs matching pattern '{pattern}'"
        )
        print(f"  Examples: {matching_constructs[perturbation_col].head(3).tolist()}")

        # Remove matched constructs from remaining pool
        remaining_constructs = remaining_constructs[~mask]

        # Sort constructs by ID for deterministic ordering before shuffle
        construct_list = matching_constructs.sort_values(perturbation_col).to_dict(
            "records"
        )

        # Shuffle with seeded random state
        random.shuffle(construct_list)

        # Create pseudo-gene groups
        pseudogene_counter = 1
        for i in range(0, len(construct_list), constructs_per_pseudogene):
            group = construct_list[i : i + constructs_per_pseudogene]
            pseudogene_id = f"{category_name}_pseudogene_{pseudogene_counter:02d}"

            pseudogene_groups.append(
                {
                    "pseudogene_id": pseudogene_id,
                    "category": category_name,
                    "constructs": group,
                }
            )
            pseudogene_counter += 1

        print(
            f"  Created {pseudogene_counter - 1} pseudo-genes for category '{category_name}'"
        )

    print(f"Total pseudo-genes created: {len(pseudogene_groups)}")
    print(f"Remaining individual constructs: {len(remaining_constructs)}")

    return pseudogene_groups, remaining_constructs


def calculate_pvals(
    null_medians_arr: np.ndarray, observed_medians: np.ndarray
) -> np.ndarray:
    """Calculate two-tailed p-values from bootstrap null distribution.

    Args:
        null_medians_arr (np.ndarray): Array of null distribution medians from
            bootstrap simulations. Shape: (n_simulations, n_features).
        observed_medians (np.ndarray): Array of observed median values for each feature.
            Shape: (n_features,).

    Returns:
        np.ndarray: Array of two-tailed p-values for each feature.
    """
    # Calculate one-tailed p-values
    pvals_one_tail = (null_medians_arr > observed_medians).mean(axis=0)

    # Convert to two-tailed by taking minimum and doubling
    pvals_two_tail = np.array([1 - x if x > 0.5 else x for x in pvals_one_tail])
    pvals_two_tail = pvals_two_tail * 2

    return pvals_two_tail


# Constructs with fewer cells are not tested: the Beta identity is loose there and the
# observed median is itself noise. Nontargeting pools are unaffected (they are the null).
MIN_CELLS = 30


def bootstrap_gene(
    pools: List[np.ndarray],
    sizes: np.ndarray,
    constructs: List[Tuple[str, int, np.ndarray]],
    gene_observed: np.ndarray,
    num_sims: int,
    rng: np.random.Generator,
    min_cells: int = MIN_CELLS,
) -> Tuple[List[Tuple[str, int, Optional[np.ndarray]]], Optional[np.ndarray], int]:
    """Construct and gene p-values for one gene; nulls live only in memory.

    Args:
        pools (List[np.ndarray]): Sorted per-control pools (prepare_control_pools).
        sizes (np.ndarray): Pool sizes.
        constructs (List[Tuple[str, int, np.ndarray]]): (id, cell count, observed medians)
            per construct of the gene.
        gene_observed (np.ndarray): Observed gene-level medians.
        num_sims (int): Simulated medians per construct.
        rng (np.random.Generator): Random generator.
        min_cells (int): Constructs below this cell count get no p-values and do not
            enter the gene null.

    Returns:
        Tuple: per-construct (id, cell count, p-values or None), gene p-values or None
            (no construct passed min_cells), number of constructs in the gene null.
    """
    rows, nulls = [], []
    for construct_id, sample_size, observed in constructs:
        if sample_size < min_cells:
            rows.append((construct_id, sample_size, None))
            continue
        null = beta_quantile_null(pools, sizes, sample_size, num_sims, rng)
        rows.append((construct_id, sample_size, calculate_pvals(null, observed)))
        nulls.append(null)
    if not nulls:
        return rows, None, 0
    # Gene null = median of the construct nulls draw by draw: a sampled median of medians.
    gene_p = calculate_pvals(np.median(np.stack(nulls), axis=0), gene_observed)
    return rows, gene_p, len(nulls)


def apply_multiple_hypothesis_correction(
    df: pd.DataFrame, feature_cols: List[str], min_p_value: Optional[float] = None
) -> pd.DataFrame:
    """Apply multiple hypothesis testing correction to p-values.

    Args:
        df (pd.DataFrame): DataFrame containing p-values to be corrected.
        feature_cols (List[str]): List of column names containing p-values to correct.
        min_p_value (Optional[float], optional): Minimum detectable p-value for FDR
            correction. If None, will be auto-detected from 'num_sims' column or
            default to 1e-5. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with additional columns for each feature:
            - {feature}_pval: Original p-values (including 0s)
            - {feature}_neg_log10_pval: -log10(p-values), with ceiling for p=0 cases
            - {feature}_fdr: FDR-corrected p-values using Benjamini-Hochberg method
            - {feature}_neg_log10_fdr: -log10(FDR-corrected p-values)
    """
    print("Applying multiple hypothesis testing correction...")

    # Auto-detect minimum p-value for FDR correction (but keep original p=0 values)
    if min_p_value is None:
        # Look for num_sims column to determine minimum detectable p-value
        if "num_sims" in df.columns:
            min_p_value = 1.0 / df["num_sims"].max()
            max_log10 = np.log10(df["num_sims"].max())
            print(
                f"For FDR correction, treating p=0 as {min_p_value} (-log10 = {max_log10})"
            )
        else:
            min_p_value = 1e-5  # Conservative default (corresponds to -log10 = 5)
            max_log10 = 5.0
            print(
                f"Using default: treating p=0 as {min_p_value} for FDR, -log10 = {max_log10}"
            )

    df_corrected = df.copy()

    # First pass: collect all valid p-values for global FDR correction
    all_pvals = []
    pval_positions = []  # Track which feature and row each p-value came from

    for feature in feature_cols:
        pvals = df[feature].values
        valid_mask = pd.notna(pvals) & (pvals >= 0) & (pvals <= 1)

        for i, (pval, valid) in enumerate(zip(pvals, valid_mask)):
            if valid:
                # Handle p-values of exactly 0 (perfect separation in bootstrap)
                adjusted_pval = max(pval, min_p_value) if pval == 0 else pval
                all_pvals.append(adjusted_pval)
                pval_positions.append((feature, i))

    print(f"Found {len(all_pvals)} valid p-values across all features")

    # Apply global FDR correction to all p-values
    if len(all_pvals) > 1:
        try:
            fdr_corrected_all = false_discovery_control(all_pvals, method="bh")
            print(
                f"Global FDR correction applied. Significant at 0.05: {(fdr_corrected_all < 0.05).sum()}"
            )
        except Exception as e:
            print(f"Global FDR correction failed: {e}")
            fdr_corrected_all = all_pvals  # Fallback to uncorrected
    else:
        print("Insufficient p-values for FDR correction")
        fdr_corrected_all = all_pvals

    # Initialize output columns
    for feature in feature_cols:
        df_corrected[f"{feature}_neg_log10_pval"] = np.nan
        df_corrected[f"{feature}_fdr"] = np.nan
        df_corrected[f"{feature}_neg_log10_fdr"] = np.nan

    # Second pass: populate corrected values
    fdr_idx = 0
    for feature in feature_cols:
        print(f"Processing feature: {feature}")
        pvals = df[feature].values
        valid_mask = pd.notna(pvals) & (pvals >= 0) & (pvals <= 1)

        if valid_mask.sum() == 0:
            print(f"  No valid p-values found for {feature}")
            continue

        print(f"  Found {valid_mask.sum()} valid p-values")

        # Process each p-value
        neg_log10_pvals = np.full(len(pvals), np.nan)
        fdr_pvals = np.full(len(pvals), np.nan)

        for i, (pval, valid) in enumerate(zip(pvals, valid_mask)):
            if valid:
                # Calculate -log10(p): use ceiling for p=0, otherwise normal calculation
                if pval == 0:
                    if "num_sims" in df.columns:
                        neg_log10_pvals[i] = np.log10(df.iloc[i]["num_sims"])
                    else:
                        neg_log10_pvals[i] = 5.0
                else:
                    neg_log10_pvals[i] = -np.log10(pval)

                # Get corresponding FDR-corrected p-value
                fdr_pvals[i] = fdr_corrected_all[fdr_idx]
                fdr_idx += 1

        # Compute -log10(FDR)
        neg_log10_fdr = np.full(len(fdr_pvals), np.nan)
        for i, fdr_val in enumerate(fdr_pvals):
            if not np.isnan(fdr_val) and fdr_val > 0:
                neg_log10_fdr[i] = -np.log10(fdr_val)
            elif not np.isnan(fdr_val) and fdr_val == 0:
                neg_log10_fdr[i] = max_log10

        # Store results (keep original p-values unchanged)
        df_corrected[f"{feature}_neg_log10_pval"] = neg_log10_pvals
        df_corrected[f"{feature}_fdr"] = fdr_pvals
        df_corrected[f"{feature}_neg_log10_fdr"] = neg_log10_fdr

        # Report zeros found
        zero_count = (df[feature].values == 0).sum()
        if zero_count > 0:
            if "num_sims" in df.columns:
                max_log10 = np.log10(df["num_sims"].max())
                print(
                    f"  Found {zero_count} p-values of exactly 0 (kept as 0, -log10 = {max_log10:.1f})"
                )
            else:
                print(
                    f"  Found {zero_count} p-values of exactly 0 (kept as 0, -log10 = 5.0)"
                )

    # Reorder columns: metadata + feature groups
    print("Reordering columns by feature...")

    # Identify metadata columns
    all_cols = df_corrected.columns.tolist()
    derived_cols = []
    for feature in feature_cols:
        derived_cols.extend(
            [
                f"{feature}_neg_log10_pval",
                f"{feature}_fdr",
                f"{feature}_neg_log10_fdr",
            ]
        )

    metadata_cols = [
        col for col in all_cols if col not in feature_cols and col not in derived_cols
    ]

    # Rename original columns to _pval
    rename_dict = {feature: f"{feature}_pval" for feature in feature_cols}
    df_corrected = df_corrected.rename(columns=rename_dict)

    # Build ordered column list
    ordered_cols = metadata_cols.copy()
    for feature in feature_cols:
        ordered_cols.extend(
            [
                f"{feature}_pval",
                f"{feature}_neg_log10_pval",
                f"{feature}_fdr",
                f"{feature}_neg_log10_fdr",
            ]
        )

    return df_corrected[ordered_cols]
