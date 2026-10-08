"""Annotate reads with per-read recombination / error-correction state (opt-in).

Same parameters as call_cells (get_call_cells_params); writes one row per input
read. Enabled by `sbs.annotate_reads: true`.
"""

from lib.sbs.call_cells import annotate_reads, load_barcode_library
from lib.shared.parquet_io import read_table, write_parquet

params = snakemake.params.config

reads_data = read_table(snakemake.input[0])
df_barcode_library = load_barcode_library(params["df_barcode_library_fp"])

if params.get("barcode_type", "simple") == "multi":
    annotated = annotate_reads(
        reads_data=reads_data,
        df_barcode_library=df_barcode_library,
        q_min=params["q_min"],
        map_start=params["map_start"],
        map_end=params["map_end"],
        prefix_map=params["prefix_map"],
        recomb_start=params["recomb_start"],
        recomb_end=params["recomb_end"],
        prefix_recomb=params["prefix_recomb"],
        recomb_filter_col=params["recomb_filter_col"],
        recomb_q_thresh=params["recomb_q_thresh"],
        recomb_call_mode=params.get("recomb_call_mode", "low_q"),
        error_correct=params["error_correct"],
        max_distance=params["max_distance"],
    )
else:
    annotated = annotate_reads(
        reads_data=reads_data,
        df_barcode_library=df_barcode_library,
        q_min=params["q_min"],
        barcode_col=params["barcode_col"],
        prefix_col=params["prefix_col"],
        error_correct=params["error_correct"],
        max_distance=params["max_distance"],
    )

write_parquet(annotated, snakemake.output[0])
