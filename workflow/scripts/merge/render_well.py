"""Render the whole well (per modality) from the positions placement -- see #88."""

from lib.merge.positions_merge import placement_from_table
from lib.merge.positions_overlay import image_path_templates, tile_image_paths
from lib.merge.render_well import channel_readers, render_channels, write_well_zarr
from lib.shared.parquet_io import read_parquet

cfg = {
    "render_well_channels": snakemake.params.render_well_channels,
}
placement = placement_from_table(read_parquet(snakemake.input[0]))
templates = image_path_templates(
    snakemake.params.root_fp, snakemake.params.image_format
)
dapi = {
    "phenotype": snakemake.params.phenotype_dapi_index,
    "sbs": snakemake.params.sbs_dapi_index,
}
for modality, out_path in zip(("phenotype", "sbs"), snakemake.output):
    place = placement[modality]
    paths = tile_image_paths(
        templates["images"][modality],
        place["tiles"].index,
        snakemake.params.plate,
        snakemake.params.well,
    )
    channels = render_channels(cfg, modality, dapi[modality])
    write_well_zarr(
        place,
        channel_readers(paths, channels),
        out_path,
        max_levels=int(snakemake.params.max_levels or 5),
        compression=snakemake.params.compression,
        extra_attrs={
            "modality": modality,
            "plate": str(snakemake.params.plate),
            "well": str(snakemake.params.well),
        },
    )
