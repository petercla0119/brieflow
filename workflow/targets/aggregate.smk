from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


AGGREGATE_FP = ROOT_FP / "aggregate"

# Define standard (non-montage) aggreagte outputs
AGGREGATE_OUTPUTS = {
    "split_datasets": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "cell_class": "{cell_class}",
                "channel_combo": "{channel_combo}",
            },
            "merge_data",
            "parquet",
        ),
    ],
    "filter": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "cell_class": "{cell_class}",
                "channel_combo": "{channel_combo}",
            },
            "filtered",
            "parquet",
        ),
    ],
    "generate_feature_table": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "features_singlecell",
            "parquet",
        ),
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "features_constructs",
            "tsv",
        ),
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "features_genes",
            "tsv",
        ),
    ],
    "align": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "aligned",
            "parquet",
        ),
    ],
    "aggregate": [
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "aggregated",
            "tsv",
        ),
    ],
    "eval_aggregate": [
        AGGREGATE_FP
        / "eval"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "na_stats",
            "tsv",
        ),
        AGGREGATE_FP
        / "eval"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "na_stats",
            "png",
        ),
        AGGREGATE_FP
        / "eval"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "feature_distributions",
            "png",
        ),
    ],
    # One AnnData per channel_combo, combining all cell classes
    "format_singlecell_anndata": [
        AGGREGATE_FP
        / "anndata"
        / get_filename(
            {"channel_combo": "{channel_combo}"},
            "singlecell",
            "h5ad",
        ),
    ],
}

AGGREGATE_OUTPUT_MAPPINGS = {
    "split_datasets": None,
    "filter": None,
    "perturbation_score_filter": None,
    "align": None,
    "aggregate": None,
    "eval_aggregate": None,
    "generate_feature_table": None,
    "format_singlecell_anndata": None,
}

AGGREGATE_OUTPUTS_MAPPED = map_outputs(AGGREGATE_OUTPUTS, AGGREGATE_OUTPUT_MAPPINGS)

# TODO: Use all combos
# aggregate_wildcard_combos = aggregate_wildcard_combos[
#     (aggregate_wildcard_combos["plate"].isin([1]))
#     & (aggregate_wildcard_combos["well"].isin(["A1"]))
#     & (aggregate_wildcard_combos["cell_class"].isin(["Mitotic"]))
#     & (aggregate_wildcard_combos["channel_combo"].isin(["DAPI_WGA"]))
# ]

AGGREGATE_TARGETS_ALL = outputs_to_targets(
    AGGREGATE_OUTPUTS, aggregate_wildcard_combos, AGGREGATE_OUTPUT_MAPPINGS
)


# Define montage outputs
# These are special because we dynamically derive outputs
if IMG_FMT == "zarr":
    # Zarr mode: individual crops in examples.zarr store
    MONTAGE_OUTPUTS = {
        "montage_data_dir": AGGREGATE_FP / "montages" / "{cell_class}__montage_data",
        "montage_data": AGGREGATE_FP
        / "montages"
        / "{cell_class}__montage_data"
        / get_filename(
            {"gene": "{gene}", "sgrna": "{sgrna}"},
            "montage_data",
            "tsv",
        ),
        "examples_zarr": AGGREGATE_FP / "montages" / "{cell_class}__examples.zarr",
        "montage_crop_flag": AGGREGATE_FP
        / "montages"
        / "{cell_class}__montages"
        / "{gene}"
        / "{sgrna}"
        / ".crop_done",
        "montage_flag": AGGREGATE_FP
        / "montages"
        / "{cell_class}__montages_complete.flag",
    }
else:
    # TIFF mode: PNG per channel + TIFF overlay
    MONTAGE_OUTPUTS = {
        "montage_data_dir": AGGREGATE_FP / "montages" / "{cell_class}__montage_data",
        "montage_data": AGGREGATE_FP
        / "montages"
        / "{cell_class}__montage_data"
        / get_filename(
            {"gene": "{gene}", "sgrna": "{sgrna}"},
            "montage_data",
            "tsv",
        ),
        "montage": AGGREGATE_FP
        / "montages"
        / "{cell_class}__montages"
        / "{gene}"
        / "{sgrna}"
        / get_filename(
            {"channel": "{channel}"},
            "montage",
            "png",
        ),
        "montage_overlay": AGGREGATE_FP
        / "montages"
        / "{cell_class}__montages"
        / "{gene}"
        / "{sgrna}"
        / get_filename(
            {},
            "overlay_montage",
            "tiff",
        ),
        "montage_flag": AGGREGATE_FP
        / "montages"
        / "{cell_class}__montages_complete.flag",
    }
cell_classes = aggregate_wildcard_combos["cell_class"].unique()
MONTAGE_TARGETS_ALL = [
    str(MONTAGE_OUTPUTS["montage_flag"]).format(cell_class=cell_class)
    for cell_class in cell_classes
]


# Define bootstrap outputs
BOOTSTRAP_OUTPUTS = {
    "combined_construct_results": AGGREGATE_FP / "bootstrap" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "all_construct_bootstrap_results", "tsv"
    ),
    "combined_gene_results": AGGREGATE_FP / "bootstrap" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "all_gene_bootstrap_results", "tsv"
    ),
}

# Bootstrap target combinations
bootstrap_combos = config.get("aggregate", {}).get("bootstrap_combinations", [])
BOOTSTRAP_TARGETS_ALL = [
    str(output_path).format(cell_class=combo["cell_class"], channel_combo=combo["channel_combo"])
    for combo in bootstrap_combos
    for output_path in [
        BOOTSTRAP_OUTPUTS["combined_construct_results"],
        BOOTSTRAP_OUTPUTS["combined_gene_results"]
    ]
]
