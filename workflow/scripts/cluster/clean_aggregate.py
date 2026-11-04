from lib.shared.parquet_io import read_df

# load aggregated data (tsv or parquet, per aggregate.table_format)
aggregated_data = read_df(snakemake.input[0])

# filter by cell cutoff
min_cell_cutoff = snakemake.params.min_cell_cutoffs[snakemake.params.cell_class]
aggregated_data = aggregated_data[aggregated_data["cell_count"] > min_cell_cutoff]

# save filtered data
aggregated_data.to_csv(snakemake.output[0], sep="\t", index=False)
