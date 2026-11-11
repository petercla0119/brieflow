"""Bootstrap statistical testing for perturbation effects.

Construct nulls come from the beta-quantile shortcut (no resampling, nothing persisted);
the gene null is the elementwise median across the gene's construct nulls, each construct
drawn and permuted independently so no draw pairs the same control pool across sgRNAs.
P-values are two-tailed empirical; Benjamini-Hochberg runs within each feature.
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


# Constructs with fewer cells get no p-value and stay out of the gene null. The null is
# valid at any n (the Beta identity is exact for odd n), so this is a coverage choice, not
# a correctness one: at 30 a TDP-GWS plate lost 35 % of its sgRNAs and 13 % of its genes;
# at 10 it loses 4 % of sgRNAs (decision 2026-11-10). Controls are unaffected (they are
# the null).
MIN_CELLS = 10

# Features are processed in chunks so a gene's live null arrays stay within a memory
# budget instead of num_sims x all features per construct (233 MB at 100k sims x 291
# features; a 658-construct pseudo-gene then needs 150 GB and took a 64 GB box down).
# Chunk width = budget / (2 arrays x num_sims x constructs x 8 B), capped at FEATURE_CHUNK.
FEATURE_CHUNK = 16
NULL_BUDGET_BYTES = 1 << 30


def _chunk_width(num_sims: int, n_constructs: int) -> int:
    """Features per chunk so the construct nulls plus their stack fit NULL_BUDGET_BYTES."""
    per_feature = 2 * num_sims * n_constructs * 8
    return max(1, min(FEATURE_CHUNK, NULL_BUDGET_BYTES // per_feature))


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
    tested = [(cid, n, obs) for cid, n, obs in constructs if n >= min_cells]
    skipped = [(cid, n, None) for cid, n, obs in constructs if n < min_cells]
    if not tested:
        return skipped, None, 0
    n_features = pools[0].shape[1]
    construct_p = np.empty((len(tested), n_features))
    gene_p = np.empty(n_features)
    width = _chunk_width(num_sims, len(tested))
    for start in range(0, n_features, width):
        sl = slice(start, min(start + width, n_features))
        chunk_pools = [pool[:, sl] for pool in pools]
        nulls = []
        for i, (_, sample_size, observed) in enumerate(tested):
            null = beta_quantile_null(chunk_pools, sizes, sample_size, num_sims, rng)
            construct_p[i, sl] = calculate_pvals(null, observed[sl])
            nulls.append(null)
        gene_p[sl] = calculate_pvals(gene_null(nulls, rng), gene_observed[sl])
    rows = [(cid, n, construct_p[i]) for i, (cid, n, _) in enumerate(tested)] + skipped
    order = {cid: k for k, (cid, _, _) in enumerate(constructs)}
    rows.sort(key=lambda r: order[r[0]])
    return rows, gene_p, len(tested)


def gene_null(nulls: List[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Gene-level null: the elementwise median across independently ordered construct nulls.

    Each construct's draws are permuted before the median, so draw ``i`` of one construct
    never pairs with draw ``i`` of another. Taking the median over aligned draws re-uses the
    same control pool (and the same Beta quantile) across the gene's sgRNAs and inflates the
    variance of the gene null, which lowers every gene p-value.

    Args:
        nulls (List[np.ndarray]): (num_sims, n_features) null medians per construct.
        rng (np.random.Generator): Random generator.

    Returns:
        np.ndarray: (num_sims, n_features) gene null medians.
    """
    num_sims = nulls[0].shape[0]
    return np.median(np.stack([n[rng.permutation(num_sims)] for n in nulls]), axis=0)


def apply_multiple_hypothesis_correction(
    df: pd.DataFrame, feature_cols: List[str], min_p_value: Optional[float] = None
) -> pd.DataFrame:
    """Benjamini-Hochberg within each feature, plus -log10 columns.

    One family per feature column: the p-values of one feature across all rows (genes or
    constructs) are corrected together, and features do not compete with each other. A
    single pass over every gene x feature test does not work at genome scale: on one
    TDP-GWS plate (19,714 genes x 291 features, 100k sims) 484 tests sat at the 1/num_sims
    floor and the smallest adjusted p anywhere was 0.1185, so nothing passed.

    Args:
        df (pd.DataFrame): DataFrame containing p-values to be corrected.
        feature_cols (List[str]): List of column names containing p-values to correct.
        min_p_value (Optional[float], optional): Value used in place of p = 0 for the
            correction (the resolution floor). If None, 1 / max(num_sims) when a
            'num_sims' column exists, else 1e-5. Original p-values are kept as written.

    Returns:
        pd.DataFrame: DataFrame with, per feature:
            - {feature}_pval: Original p-values (including 0s)
            - {feature}_neg_log10_pval: -log10(p-values), with ceiling for p=0 cases
            - {feature}_fdr: FDR-corrected p-values using Benjamini-Hochberg method
            - {feature}_neg_log10_fdr: -log10(FDR-corrected p-values)
    """
    if min_p_value is None:
        if "num_sims" in df.columns:
            min_p_value = 1.0 / df["num_sims"].max()
        else:
            min_p_value = 1e-5
    max_log10 = -np.log10(min_p_value)
    print(
        f"BH per feature over {len(feature_cols)} features; p = 0 enters as {min_p_value}"
    )

    out = {c: df[c] for c in df.columns if c not in feature_cols}
    n_sig = 0
    for feature in feature_cols:
        pvals = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(pvals) & (pvals >= 0) & (pvals <= 1)
        fdr = np.full(len(pvals), np.nan)
        if valid.sum() > 1:
            fdr[valid] = false_discovery_control(
                np.maximum(pvals[valid], min_p_value), method="bh"
            )
        elif valid.sum() == 1:
            fdr[valid] = max(pvals[valid][0], min_p_value)
        with np.errstate(divide="ignore"):
            neg_log10_p = np.where(
                valid, -np.log10(np.maximum(pvals, min_p_value)), np.nan
            )
            neg_log10_fdr = np.where(
                np.isfinite(fdr), -np.log10(np.maximum(fdr, 10.0**-max_log10)), np.nan
            )
        n_sig += int(np.nansum(fdr < 0.05))
        out[f"{feature}_pval"] = pvals
        out[f"{feature}_neg_log10_pval"] = neg_log10_p
        out[f"{feature}_fdr"] = fdr
        out[f"{feature}_neg_log10_fdr"] = neg_log10_fdr
    print(f"Significant at FDR 0.05 across features: {n_sig}")
    return pd.DataFrame(out, index=df.index)
