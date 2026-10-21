import random

import matplotlib.pyplot as plt
import pyarrow.parquet as pq

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    }
)

from lib.aggregate.eval_aggregate import (
    nas_summary,
    plot_feature_distributions,
)
from lib.shared.parquet_io import sample_parquet_rows

SUBSET_SIZE = 100000

# Get merge dataset — filter out empty parquets to avoid schema conflicts
non_empty_split = [
    p for p in snakemake.input.split_datasets_paths if pq.read_metadata(p).num_rows > 0
]

if len(non_empty_split) == 0:
    print("WARNING: No cells in input, writing empty eval outputs")
    import pandas as pd

    pd.DataFrame().to_csv(snakemake.output[0], sep="\t", index=False)
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    fig.savefig(snakemake.output[1], dpi=300, bbox_inches="tight")
    fig.savefig(snakemake.output[2], dpi=300, bbox_inches="tight")
    plt.close(fig)
    exit(0)

# One file at a time: Dataset.scanner().take() decodes every file in full.
merge_data = sample_parquet_rows(non_empty_split, SUBSET_SIZE, seed=None, float32=False)

# Evaluate missing values
nas_df, nas_fig = nas_summary(merge_data, vis_subsample=50000)
nas_df.to_csv(snakemake.output[0], sep="\t", index=False)
nas_fig.savefig(snakemake.output[1], dpi=300, bbox_inches="tight", transparent=True)

# Get aligned dataset
aligned_data = sample_parquet_rows(
    [snakemake.input[0]], SUBSET_SIZE, seed=None, float32=False
)

# determine original and aligned columns
random.seed(42)
# Prefer cell-level features, fall back to nuclear if none found
merge_feature_cols = [
    col for col in merge_data.columns if ("cell_" in col and col.endswith("_mean"))
]
if len(merge_feature_cols) == 0:
    merge_feature_cols = [
        col
        for col in merge_data.columns
        if ("nucleus_" in col and col.endswith("_mean"))
    ]
pc_cols = [col for col in aligned_data.columns if col.startswith("PC_")]
aligned_feature_cols = random.sample(
    pc_cols, k=min(len(merge_feature_cols), len(pc_cols))
)

# Evaluate feature distributions
feature_distributions_fig = plot_feature_distributions(
    merge_feature_cols,
    merge_data,
    aligned_feature_cols,
    aligned_data,
)
feature_distributions_fig.savefig(
    snakemake.output[2], dpi=300, bbox_inches="tight", transparent=True
)
