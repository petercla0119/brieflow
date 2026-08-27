from lib.shared.combine_dfs import combine_tile_dfs
from lib.shared.parquet_io import write_parquet

# Read per-tile TSVs, concat, and normalize dtypes (shared helper).
# ponytail: Phase 2 will drop TSV intermediates for parquet upstream; this
# read+concat helper is where that swap lands.
combined_df = combine_tile_dfs(snakemake.input)

# Save the data based on output_type
output_type = getattr(snakemake.params, "output_type", "parquet")
if output_type == "parquet":
    write_parquet(combined_df, snakemake.output[0])
elif output_type == "tsv":
    combined_df.to_csv(snakemake.output[0], sep="\t", index=False)
else:
    raise ValueError(f"Unsupported output type: {output_type}")
