import gc
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
import numpy as np

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.align import (
    prepare_alignment_data,
    centerscale_by_batch,
    tvn_on_controls,
)
from lib.aggregate.perturbation_score import perturbation_score
from lib.shared.parquet_io import (
    columns_without_nulls,
    read_parquet_frames,
    sample_parquet_rows,
)

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.feature_selection"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.feature_selection"
)
np.random.seed(0)

# Validate required params
for _param_name in ["metadata_cols_fp", "perturbation_name_col", "control_key"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

## Step 1: Create PCA transformation
PCA_SUBSET = 100000

# Filter out empty parquet files to avoid schema conflicts
non_empty_paths = [
    p for p in snakemake.input.filtered_paths if pq.read_metadata(p).num_rows > 0
]

if len(non_empty_paths) == 0:
    print("WARNING: No cells in input, writing empty output")
    pq.write_table(pa.table({}), snakemake.output[0])
    exit(0)

# One file (one well) is read at a time: Dataset.scanner().take() decodes every
# row of every file regardless of how many indices it is given.
keep_cols = columns_without_nulls(non_empty_paths)
print(
    f"{len(non_empty_paths)} non-empty parquet files; {len(keep_cols)} columns without nulls"
)

# load sample df as pandas dataframe
sample_df = sample_parquet_rows(non_empty_paths, PCA_SUBSET, seed=0, columns=keep_cols)
use_classifier = snakemake.params.use_classifier
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, use_classifier)
metadata, features = split_cell_data(sample_df, metadata_cols)
metadata, features = prepare_alignment_data(
    metadata,
    features,
    snakemake.params.batch_cols,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    snakemake.params.perturbation_id_col,
)
pca = PCA(n_components=snakemake.params.variance_or_ncomp).fit(
    centerscale_by_batch(features, metadata, "batch_values")
)
del sample_df
gc.collect()

## Step 2: Per-file alignment

metadata_cols += ["perturbation_score", "perturbation_auc"]
writer = None
for i, (path, subset_df) in enumerate(
    read_parquet_frames(non_empty_paths, columns=keep_cols)
):
    print(
        f"Processing file {i + 1}/{len(non_empty_paths)} with {len(subset_df)} cells: {path}"
    )

    # CALCULATE PERTURBATION SCORE
    subset_df["perturbation_score"] = np.nan
    subset_df["perturbation_auc"] = np.nan
    if not snakemake.params.skip_perturbation_score:
        perturbation_score(
            subset_df,
            metadata_cols,
            snakemake.params.perturbation_name_col,
            snakemake.params.control_key,
        )

    for col in subset_df.columns:
        if is_numeric_dtype(subset_df[col]):
            subset_df[col] = subset_df[col].astype("float32")

    metadata, features = split_cell_data(subset_df, metadata_cols)
    del subset_df
    gc.collect()

    metadata, features = prepare_alignment_data(
        metadata,
        features,
        snakemake.params.batch_cols,
        snakemake.params.perturbation_name_col,
        snakemake.params.control_key,
        snakemake.params.perturbation_id_col,
    )

    features = centerscale_by_batch(features, metadata, "batch_values")

    features = pca.transform(features)

    features = tvn_on_controls(
        features,
        metadata,
        snakemake.params.perturbation_name_col,
        snakemake.params.control_key,
        "batch_values",
        control_col=snakemake.params.control_name_col,
    )

    feature_columns = [f"PC_{j}" for j in range(features.shape[1])]
    features = pd.DataFrame(features, index=metadata.index, columns=feature_columns)
    aligned_cell_data = pd.concat([metadata, features], axis=1)
    del features
    gc.collect()

    # Convert to Arrow table and write chunk
    aligned_cell_data = pa.Table.from_pandas(aligned_cell_data, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(snakemake.output[0], aligned_cell_data.schema)
    writer.write_table(aligned_cell_data)

# Close writer
if writer is not None:
    writer.close()
