"""This module provides functions for data preparation, PCA analysis, and normalization.

It includes utilities for aligning data, performing PCA transformations, and applying
typical variation normalization (TVN) on embeddings based on control perturbation units.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy import linalg


def prepare_alignment_data(
    metadata, features, batch_cols, pert_col, control_key, pert_id_col
):
    """Prepare batch values and split metadata and feature DataFrames.

    Args:
        metadata (pd.DataFrame): Input DataFrame containing metadata.
        features (pd.DataFrame): DataFrame containing feature data.
        batch_cols (list): List of column names used to generate batch values.
        pert_col (str): Column name representing perturbation labels.
        control_key (str): Key for identifying control samples in the metadata.
        pert_id_col (str): Column name for perturbation IDs.

    Returns:
        tuple: metadata (pd.DataFrame), features (numpy.ndarray)
    """
    metadata = metadata.copy()

    # Create batch values
    batch_values = metadata[batch_cols[0]].astype(str)
    for col in batch_cols[1:]:
        batch_values = batch_values + "_" + metadata[col].astype(str)

    # Add batch values to metadata
    metadata["batch_values"] = batch_values

    # Add unique number suffix to perturbation names based on pert_id_col
    if control_key is not None and pert_col is not None and pert_id_col is not None:
        control_mask = metadata[pert_col] == control_key
        metadata.loc[control_mask, pert_col] = (
            control_key + "_" + metadata.loc[control_mask, pert_id_col].astype(str)
        )

    # Extract feature data
    features = features.to_numpy()

    return metadata, features


def embed_file(
    path,
    columns,
    metadata_cols,
    pca,
    batch_cols,
    pert_col,
    control_key,
    pert_id_col,
    skip_perturbation_score=True,
):
    """Center-scale one well file by batch and project it with a fitted PCA.

    One file is one well, and ``batch_cols`` always includes the well, so the per-batch
    center-scale never needs cells from another file; that is what lets files be
    embedded independently (and in parallel, see :func:`embed_files`).

    Returns:
        tuple: metadata (pd.DataFrame) and the float32 embedding (np.ndarray).
    """
    import gc

    import pyarrow.parquet as pq
    from pandas.api.types import is_numeric_dtype

    from lib.aggregate.cell_data_utils import split_cell_data
    from lib.shared.parquet_io import _float32_schema

    table = pq.ParquetFile(path).read(columns=columns)
    subset_df = table.cast(_float32_schema(table.schema)).to_pandas()
    del table

    subset_df["perturbation_score"] = np.nan
    subset_df["perturbation_auc"] = np.nan
    if not skip_perturbation_score:
        from lib.aggregate.perturbation_score import perturbation_score

        perturbation_score(subset_df, metadata_cols, pert_col, control_key)

    for col in subset_df.columns:
        if is_numeric_dtype(subset_df[col]):
            subset_df[col] = subset_df[col].astype("float32")

    metadata, features = split_cell_data(subset_df, metadata_cols)
    del subset_df
    gc.collect()

    metadata, features = prepare_alignment_data(
        metadata, features, batch_cols, pert_col, control_key, pert_id_col
    )
    features = centerscale_by_batch(features, metadata, "batch_values")
    embedding = pca.transform(features).astype(np.float32)
    del features
    gc.collect()
    return metadata, embedding


def embed_files(paths, n_jobs=1, **kwargs):
    """Embed every well file with :func:`embed_file`, ``n_jobs`` files at a time.

    ``n_jobs > 1`` runs the files in a spawn pool. Each worker holds one well's raw
    features (tens of GB on a 755k-cell well with 3.6k features), so size ``n_jobs`` to
    memory, not cores. BLAS threads inside the workers are capped to 2 so the pool does
    not oversubscribe the node; the parent keeps its own thread count for TVN.

    Returns:
        tuple: concatenated metadata (pd.DataFrame) and embeddings (np.ndarray), in the
        order of ``paths`` whatever ``n_jobs`` is.
    """
    paths = list(paths)
    if n_jobs <= 1:
        results = []
        for i, path in enumerate(paths):
            print(f"Processing file {i + 1}/{len(paths)}: {path}")
            results.append(embed_file(path, **kwargs))
    else:
        import multiprocessing as mp
        import os

        thread_vars = (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_MAX_THREADS",
        )
        import __main__

        saved = {k: os.environ.get(k) for k in thread_vars}
        os.environ.update({k: "2" for k in thread_vars})
        # Snakemake runs scripts as a plain file with top-level code and no module spec;
        # a spawn worker would re-execute that file as its main module (re-running the
        # whole align step, then failing on the missing `snakemake` object) and the pool
        # would respawn workers forever. With no main file to import, the workers only
        # unpickle `_embed_file_kw` by importing this module.
        main_file = getattr(__main__, "__file__", None)
        hide_main = getattr(__main__, "__spec__", None) is None and main_file is not None
        if hide_main:
            __main__.__file__ = None
        try:
            with mp.get_context("spawn").Pool(n_jobs) as pool:
                results = pool.starmap(
                    _embed_file_kw, [(path, kwargs) for path in paths], chunksize=1
                )
        finally:
            if hide_main:
                __main__.__file__ = main_file
            for k, v in saved.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
    metadata = pd.concat([m for m, _ in results], ignore_index=True)
    embeddings = np.concatenate([e for _, e in results])
    return metadata, embeddings


def _embed_file_kw(path, kwargs):
    return embed_file(path, **kwargs)


def pca_variance_plot(features, variance_threshold=0.95, random_state=42):
    """Perform PCA analysis and create an explained variance plot.

    Args:
        features (np.ndarray): Array containing feature data to be analyzed.
        variance_threshold (float): Cumulative variance threshold. Defaults to 0.95.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - n_components (int): Number of components needed to reach the variance threshold.
            - fig (matplotlib.figure.Figure): Figure object for the explained variance plot.
    """
    # Copy and scale data
    features = features.copy()
    features = centerscale_by_batch(features)

    # Initialize and fit PCA
    pca = PCA(random_state=random_state)
    pca_transformed = pca.fit_transform(features)

    # Create DataFrame with PCA results
    n_components_total = pca_transformed.shape[1]
    pca_df = pd.DataFrame(
        pca_transformed,
        columns=[f"pca_{n}" for n in range(n_components_total)],
    )

    # Find number of components needed for threshold
    cumsum = pca.explained_variance_ratio_.cumsum()
    n_components = np.argwhere(cumsum >= variance_threshold)[0][0] + 1

    # Create variance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumsum, "-")
    ax.axhline(
        variance_threshold,
        linestyle="--",
        color="red",
        label=f"{variance_threshold * 100}% Threshold",
    )
    ax.axvline(n_components, linestyle="--", color="blue", label=f"n={n_components}")
    ax.set_ylabel("Cumulative fraction of variance explained")
    ax.set_xlabel("Number of principal components included")
    ax.set_title("PCA Explained Variance Ratio")
    ax.grid(True)
    ax.legend()

    return n_components, fig


def embed_by_pca(
    features: np.ndarray,
    metadata: pd.DataFrame = None,
    variance_or_ncomp=128,
    batch_col: str | None = None,
) -> np.ndarray:
    """Embed the whole input data using principal component analysis (PCA).

    Note that we explicitly center & scale the data (by batch) before an embedding operation with `PCA`.
    Centering and scaling is done by batch if `batch_col` is not None, and on the whole data otherwise.
    Also note that `PCA` transformer also does mean-centering on the whole data prior to the PCA operation.

    Args:
        features (np.ndarray): Features to transform
        metadata (pd.DataFrame): Metadata. Defaults to None.
        variance_or_ncomp (float, optional): Variance or number of components to keep after PCA.
            Defaults to 128 (n_components). If between 0 and 1, select the number of components such that
            the amount of variance that needs to be explained is greater than the percentage specified.
            If 1, a single component is kept, and if None, all components are kept.
        batch_col (str, optional): Column name for batch information. Defaults to None.

    Returns:
        np.ndarray: Transformed data using PCA.
    """
    features = centerscale_by_batch(features, metadata, batch_col)
    if features.shape[0] > 50000:
        sample = features[:50000]
        pca = PCA(n_components=variance_or_ncomp).fit(sample)
        features = pca.transform(features)
    else:
        features = PCA(n_components=variance_or_ncomp).fit_transform(features)
    return features


def tvn_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: str | None = None,
    control_col: str | None = None,
) -> np.ndarray:
    """Apply TVN (Typical Variation Normalization) to the data based on the control perturbation units.

    Note that the data is first centered and scaled based on the control units.

    Args:
        embeddings (np.ndarray): The embeddings to be normalized.
        metadata (pd.DataFrame): The metadata containing information about the samples.
        pert_col (str): The column name in the metadata DataFrame that represents the perturbation labels.
        control_key (str): The control perturbation label.
        batch_col (str, optional): Column name in the metadata DataFrame representing the batch labels
            to be used for CORAL normalization. Defaults to None.
        control_col (str, optional): Column name to use for identifying control cells.
            When None, uses pert_col. Useful when aggregating by barcode but controls
            are identified by gene name in a different column. Defaults to None.

    Returns:
        np.ndarray: The normalized embeddings.
    """
    lookup_col = control_col if control_col is not None else pert_col
    ctrl_ind = metadata[lookup_col].str.startswith(control_key).to_list()
    n_controls = sum(ctrl_ind)
    if n_controls == 0:
        print("Warning: no control cells found, skipping TVN normalization")
        return embeddings
    embeddings = centerscale_on_controls(
        embeddings, metadata, pert_col, control_key, control_col=lookup_col
    )
    embeddings = PCA().fit(embeddings[ctrl_ind]).transform(embeddings)
    embeddings = centerscale_on_controls(
        embeddings, metadata, pert_col, control_key, batch_col, control_col=lookup_col
    )
    target_cov = np.cov(embeddings[ctrl_ind], rowvar=False, ddof=1) + 0.5 * np.eye(
        embeddings.shape[1]
    )
    if batch_col is not None:
        # Pre-compute target matrix power once (expensive operation)
        target_cov_sqrt = linalg.fractional_matrix_power(target_cov, 0.5)
        batches = metadata[batch_col].unique()
        for batch in batches:
            batch_ind = metadata[batch_col] == batch
            batch_control_ind = (
                batch_ind & (metadata[lookup_col].str.startswith(control_key)).to_list()
            )
            n_controls = np.sum(batch_control_ind)
            if n_controls < embeddings.shape[1]:
                print(
                    f"Warning: batch {batch} has {n_controls} control cells "
                    f"(< {embeddings.shape[1]} features), skipping CORAL for this batch"
                )
                continue
            source_cov = np.cov(
                embeddings[batch_control_ind], rowvar=False, ddof=1
            ) + 0.5 * np.eye(embeddings.shape[1])
            source_cov_inv_sqrt = linalg.fractional_matrix_power(source_cov, -0.5)
            if not np.all(np.isfinite(source_cov_inv_sqrt)):
                print(
                    f"Warning: batch {batch} has singular covariance matrix, "
                    f"skipping CORAL for this batch"
                )
                continue
            embeddings[batch_ind] = np.matmul(
                embeddings[batch_ind], source_cov_inv_sqrt
            )
            embeddings[batch_ind] = np.matmul(embeddings[batch_ind], target_cov_sqrt)
    return embeddings


def centerscale_by_batch(
    features: np.ndarray, metadata: pd.DataFrame = None, batch_col: str | None = None
) -> np.ndarray:
    """Center and scale the input features by each batch. Not using any controls at all.

    We are using this prior to embedding high-dimensional data with PCA.

    Args:
        features (np.ndarray): Input features to be centered and scaled.
        metadata (pd.DataFrame): Metadata information for the input features.
        batch_col (str): Name of the column in metadata that contains batch information.

    Returns:
        np.ndarray: Centered and scaled features.
    """
    if batch_col is None:
        features = StandardScaler(copy=False).fit_transform(features)
    else:
        if metadata is None:
            raise ValueError("metadata must be provided if batch_col is not None")
        batches = metadata[batch_col].unique()
        for batch in batches:
            ind = metadata[batch_col] == batch
            features[ind, :] = StandardScaler(copy=False).fit_transform(
                features[ind, :]
            )
    return features


def median_absolute_deviation(arr):
    """Calculate MAD with scaling factor for normal distribution equivalence.

    Args:
        arr (np.ndarray): Input array to calculate MAD.

    Returns:
        np.ndarray: Scaled MAD values.
    """
    median = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - median), axis=0)
    return mad


def centerscale_on_controls(
    embeddings: np.ndarray,
    metadata: pd.DataFrame,
    pert_col: str,
    control_key: str,
    batch_col: str | None = None,
    method: str = "standard",
    control_col: str | None = None,
) -> np.ndarray:
    """Center and scale the embeddings on the control perturbation units in the metadata.

    If batch information is provided, the embeddings are centered and scaled by batch.

    Args:
        embeddings (numpy.ndarray): The embeddings to be aligned.
        metadata (pandas.DataFrame): The metadata containing information about the embeddings.
        pert_col (str): The column in the metadata containing perturbation information.
        control_key (str): The key for non-targeting controls in the metadata.
        batch_col (str, optional): Column name in the metadata representing the batch labels.
            Defaults to None.
        method (str, optional): Scaling method to use. Options are "standard" (mean/std)
            or "mad" (median/MAD). Defaults to "standard".
        control_col (str, optional): Column to use for identifying controls.
            When None, uses pert_col. Defaults to None.

    Returns:
        numpy.ndarray: The aligned embeddings.
    """
    if method not in ["standard", "mad"]:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard' or 'mad'.")

    # helper for MAD scaling (robust z-score, consistent with std under normality)
    def _mad_scale(X: np.ndarray, ref: np.ndarray) -> np.ndarray:
        med = np.median(ref, axis=0)
        mad = np.median(np.abs(ref - med), axis=0)
        mad_safe = np.where(mad == 0, 1.0, mad)  # avoid divide-by-zero
        return 0.6745 * (X - med) / mad_safe

    # boolean mask for "control" rows uses startswith on stringified column, handles NaNs
    lookup_col = control_col if control_col is not None else pert_col
    ctrl_mask_all = metadata[lookup_col].astype(str).str.startswith(control_key)

    if batch_col is not None:
        for batch in metadata[batch_col].unique():
            batch_mask = metadata[batch_col] == batch
            batch_ctrl_mask = batch_mask & ctrl_mask_all

            # only scale if we have controls in this batch
            if batch_ctrl_mask.sum() == 0:
                continue

            if method == "standard":
                embeddings[batch_mask] = (
                    StandardScaler(copy=False)
                    .fit(embeddings[batch_ctrl_mask])
                    .transform(embeddings[batch_mask])
                )
            else:  # method == "mad"
                ref = embeddings[batch_ctrl_mask]
                embeddings[batch_mask] = _mad_scale(embeddings[batch_mask], ref)

        return embeddings

    # no batching: use all controls
    if method == "standard":
        return (
            StandardScaler(copy=False)
            .fit(embeddings[ctrl_mask_all])
            .transform(embeddings)
        )
    else:  # method == "mad"
        ref = embeddings[ctrl_mask_all]
        return _mad_scale(embeddings, ref)
