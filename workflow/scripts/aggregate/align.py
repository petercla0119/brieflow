import gc
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.align import (
    prepare_alignment_data,
    centerscale_by_batch,
    embed_files,
    tvn_on_controls,
)
from lib.shared.parquet_io import columns_without_nulls, sample_parquet_rows

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

## Step 2: Per-file PCA embedding
# Raw features are the memory problem (thousands of columns); the PCA embeddings are
# small enough to hold for every cell, so TVN below runs once over all cells — the
# per-batch TVN refits PCA() on each batch's own controls, which gives every batch a
# different width and rotation.

metadata_cols += ["perturbation_score", "perturbation_auc"]
align_n_jobs = int(getattr(snakemake.params, "align_n_jobs", 1) or 1)
print(f"Embedding {len(non_empty_paths)} files with align_n_jobs={align_n_jobs}")
metadata, embeddings = embed_files(
    non_empty_paths,
    n_jobs=align_n_jobs,
    columns=keep_cols,
    metadata_cols=metadata_cols,
    pca=pca,
    batch_cols=snakemake.params.batch_cols,
    pert_col=snakemake.params.perturbation_name_col,
    control_key=snakemake.params.control_key,
    pert_id_col=snakemake.params.perturbation_id_col,
    skip_perturbation_score=snakemake.params.skip_perturbation_score,
)

## Step 3: TVN over all cells, then write

gc.collect()
print(f"Running TVN on {embeddings.shape[0]} cells x {embeddings.shape[1]} components")

embeddings = tvn_on_controls(
    embeddings,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    "batch_values",
    control_col=snakemake.params.control_name_col,
)

feature_columns = [f"PC_{j}" for j in range(embeddings.shape[1])]
embeddings = pd.DataFrame(embeddings, index=metadata.index, columns=feature_columns)
aligned_cell_data = pd.concat([metadata, embeddings], axis=1)
del embeddings, metadata
gc.collect()

pq.write_table(
    pa.Table.from_pandas(aligned_cell_data, preserve_index=False), snakemake.output[0]
)
