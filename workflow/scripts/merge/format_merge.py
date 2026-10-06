import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet, write_parquet
from lib.merge.format_merge import (
    fov_distance,
    identify_single_gene_mappings,
    calculate_channel_mins,
    attach_global_pixel_coords,
    select_sbs_merge_cols,
)

# Load data for formatting merge data
merge_data = validate_dtypes(read_parquet(snakemake.input[0]))
sbs_cells = validate_dtypes(read_parquet(snakemake.input[1]))
phenotype_min_cp = validate_dtypes(read_parquet(snakemake.input[2]))

approach = snakemake.params.approach
phenotype_dimensions = tuple(snakemake.params.phenotype_dimensions or (2960, 2960))
sbs_dimensions = tuple(snakemake.params.sbs_dimensions or (1480, 1480))

# Add FOV distances for both imaging modalities
merge_formatted = merge_data.pipe(
    fov_distance, i="i_0", j="j_0", dimensions=phenotype_dimensions, suffix="_0"
)
merge_formatted = merge_formatted.pipe(
    fov_distance, i="i_1", j="j_1", dimensions=sbs_dimensions, suffix="_1"
)

# Identify single gene mappings in SBS
sbs_cells["mapped_single_gene"] = sbs_cells.apply(
    lambda x: identify_single_gene_mappings(x), axis=1
)
# Merge cell information from sbs — per-barcode columns + prefix_recomb
sbs_merge_cols = select_sbs_merge_cols(sbs_cells)

merge_formatted = merge_formatted.merge(
    sbs_cells[sbs_merge_cols].rename({"tile": "site", "cell": "cell_1"}, axis=1),
    how="left",
    on=["plate", "well", "site", "cell_1"],
)

# Calculate minimum channel values for cells
phenotype_min_cp = calculate_channel_mins(phenotype_min_cp)
merge_formatted = merge_formatted.merge(
    phenotype_min_cp[["tile", "label", "channels_min"]].rename(
        columns={"label": "cell_0"}
    ),
    how="left",
    on=["tile", "cell_0"],
)

# Attach global pixel coords from tile-local coords (stitch approach derives these in stitch_merge)
if approach in ("fast", "positions"):
    phenotype_metadata = pd.read_parquet(snakemake.input.phenotype_metadata)
    sbs_metadata = pd.read_parquet(snakemake.input.sbs_metadata)
    merge_formatted = attach_global_pixel_coords(
        merge_formatted, phenotype_metadata, phenotype_dimensions, suffix="0"
    )
    merge_formatted = attach_global_pixel_coords(
        merge_formatted, sbs_metadata, sbs_dimensions, suffix="1"
    )

# Save formatted merge data
write_parquet(merge_formatted, snakemake.output[0])
