"""Utility functions for handling cell data in the brieflow aggregation pipeline.

This module provides helper functions for manipulating cell data, including
loading metadata columns, splitting cell data into metadata and features,
and filtering features based on channel combinations.
"""

import pandas as pd

from lib.phenotype.constants import DEFAULT_METADATA_COLS


def load_metadata_cols(metadata_cols_fp, include_classification_cols=False):
    """Load metadata column names from a file.

    Args:
        metadata_cols_fp (str): File path to the metadata columns list.
        include_classification_cols (bool, optional): Whether to include
            classification columns. Defaults to False.

    Returns:
        list: List of metadata column names.
    """
    metadata_cols = pd.read_csv(metadata_cols_fp, header=None, sep="\t")[0].tolist()

    if include_classification_cols:
        metadata_cols += [
            "class",
            "confidence",
            "cell_stage_confidence",
        ]

    return metadata_cols


def split_cell_data(
    cell_data, metadata_cols, validate_dtypes=True, raise_on_invalid=True
):
    """Splits the cell data into metadata and features.

    Args:
        cell_data (pd.DataFrame): Input DataFrame containing cell data.
        metadata_cols (list): List of column names that represent metadata.
        validate_dtypes (bool, optional): Whether to validate that feature columns
            have numeric dtypes. Defaults to True.
        raise_on_invalid (bool, optional): Whether to raise an error if invalid
            dtypes are found. If False, only prints a warning. Defaults to True.

    Returns:
        tuple: (metadata, features) where metadata is a DataFrame containing
            only metadata columns and features is a DataFrame containing all
            non-metadata columns.

    Raises:
        ValueError: If validate_dtypes=True, raise_on_invalid=True, and non-numeric
            feature columns are detected.
    """
    # Ensure all metadata columns exist in the data
    existing_metadata_cols = [col for col in metadata_cols if col in cell_data.columns]

    # Get metadata columns
    metadata = cell_data[existing_metadata_cols].copy()

    # Get feature columns (all columns not in metadata)
    features = cell_data.drop(columns=existing_metadata_cols).copy()

    # Validate feature dtypes
    if validate_dtypes:
        print("Validating feature columns ...")
        invalid_cols = []
        for col in features.columns:
            dtype = features[col].dtype
            if dtype == "object" or dtype.name == "object":
                invalid_cols.append(col)

        if invalid_cols:
            error_msg = f"\nWARNING: Found {len(invalid_cols)} non-numeric columns in features!\n"
            error_msg += "These columns should be added to METADATA_COLS:\n\n"
            for col in invalid_cols:
                sample_val = features[col].iloc[0] if len(features) > 0 else "N/A"
                error_msg += (
                    f"  - {col}: dtype={features[col].dtype}, sample='{sample_val}'\n"
                )
            error_msg += f"\nAdd these to CANDIDATE_METADATA_COLS (or the metadata_cols parameter) to fix."

            if raise_on_invalid:
                raise ValueError(
                    f"Invalid feature dtypes detected: {invalid_cols}\n{error_msg}"
                )
            else:
                print(error_msg)
        else:
            print("All feature columns have valid numeric dtypes")

    return metadata, features


def parse_channel_combo(channel_combo, all_channels):
    """Split a channel-combo wildcard such as ``"TDP43_STMN2_CE_DAPI"`` into channel names.

    Channel names may themselves contain underscores (``STMN2_CE``, ``DAPI_round2``), so
    a plain ``split("_")`` is wrong; this matches known channel names greedily, longest
    first, from the start of the string.

    Args:
        channel_combo (str): Wildcard value, channel names joined by ``_``, or ``"all"``.
        all_channels (list): Every channel name of the screen.

    Returns:
        list: Channel names in the order written; ``["all"]`` for the sentinel.

    Raises:
        ValueError: If any part of the string is not a known channel name.
    """
    if channel_combo == "all":
        return ["all"]
    all_sorted = sorted(all_channels, key=len, reverse=True)
    rest, channels = channel_combo, []
    while rest:
        for ch in all_sorted:
            if rest == ch or rest.startswith(ch + "_"):
                channels.append(ch)
                rest = rest[len(ch) + 1 :]
                break
        else:
            raise ValueError(
                f"channel combo {channel_combo!r} does not split into channels {all_channels}"
            )
    return channels


def channels_of(col, all_channels):
    """Every channel a feature name refers to; empty for channel-agnostic features.

    Channel names are matched at token boundaries, longest first, and each match is
    consumed so ``DAPI`` is not also found inside ``DAPI_round2``.
    """
    found, s = set(), f"{col}_"
    for ch in sorted(all_channels, key=len, reverse=True):
        token = f"_{ch}_"
        if token in s:
            found.add(ch)
            s = s.replace(token, "_#_")
    return found


def channel_combo_subset(features, channel_combo, all_channels):
    """Filter features to the columns that belong to a channel combination.

    A column is kept when it is channel-agnostic (shape features such as ``cell_area``)
    or when every channel it refers to is in the combination. A two-channel feature such
    as ``nucleus_correlation_DAPI_GFP`` therefore needs both ``DAPI`` and ``GFP``; keeping
    it under the first matching channel would leak the other channel into the partition.

    Args:
        features (pd.DataFrame): DataFrame containing feature data.
        channel_combo (list | str): List of channels to include, or "all" to
            keep every column.
        all_channels (list): List of all available channels.

    Returns:
        pd.DataFrame: DataFrame with features filtered to include only
            columns from the specified channel combination.
    """
    if channel_combo == "all" or tuple(channel_combo) == ("all",):
        return features

    keep_channels = set(channel_combo)
    columns_to_keep = [
        col for col in features.columns if channels_of(col, all_channels) <= keep_channels
    ]

    return features[columns_to_keep]


def get_feature_table_cols(feature_cols):
    """Filter feature columns based on specific tags and compartments.

    Args:
        feature_cols (list): List of feature column names.

    Returns:
        list: Filtered list of feature column names.
    """
    # Define the specific tags to look for
    intensity_tags = [
        "mean",
        "integrated",
        "mass_displacement",
        "mean_edge",
        "std_edge",
        "mean_frac_0",
        "mean_frac_3",
    ]
    shape_tags = ["area", "solidity", "form_factor", "eccentricity"]
    # Substring match — picks up *_correlation_* (Pearson/Spearman) and *manders* features
    overlap_tags = ["manders", "correlation"]

    # Define the specific compartments to look for
    compartments = ["nucleus", "cell"]

    # Initialize lists to store columns for each feature type
    intensity_cols = []
    shape_cols = []
    overlap_cols = []

    # Filter columns based on compartments and tags
    for col in feature_cols:
        # Only include columns for nucleus or cell compartments
        if any(compartment in col for compartment in compartments):
            # Intensity features - must be at END of string
            if any(col.lower().endswith(tag) for tag in intensity_tags):
                intensity_cols.append(col)

            # Shape features - must be at END of string
            elif any(col.lower().endswith(tag) for tag in shape_tags):
                shape_cols.append(col)

            # Overlap features - can be anywhere in string
            elif any(tag in col.lower() for tag in overlap_tags):
                overlap_cols.append(col)

    # Create a new DataFrame with selected columns, preserving the label column if it exists
    selected_columns = []
    if "label" in feature_cols:
        selected_columns.append("label")

    # Add columns in an organized way with clear section breaks
    selected_columns.extend(intensity_cols)
    selected_columns.extend(shape_cols)
    selected_columns.extend(overlap_cols)

    return selected_columns
