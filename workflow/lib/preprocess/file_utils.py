"""Utility functions for handling files during preprocessing."""

import pandas as pd
from typing import List, Union


def get_sample_fps(
    samples_df: pd.DataFrame,
    plate: Union[int, str] = None,
    well: Union[int, str] = None,
    tile: Union[int, str] = None,
    cycle: Union[int, str] = None,
    channel: str = None,
    round_order: List[int] = None,
    channel_order: List[str] = None,
    verbose: bool = False,
) -> Union[str, List[str]]:
    """Filters the samples DataFrame and ensures consistent channel and round order.

    Args:
        samples_df (pd.DataFrame): DataFrame containing sample data.
        plate (Union[int, str], optional): Plate number to filter by. Defaults to None.
        well (Union[int, str], optional): Well identifier to filter by. Defaults to None.
        tile (Union[int, str], optional): Tile number to filter by. Defaults to None.
        cycle (Union[int, str], optional): Cycle number to filter by. Defaults to None.
        channel (str, optional): Channel to filter by. Defaults to None.
        round_order (List[int], optional): Order of rounds to return. Defaults to None.
        channel_order (List[str], optional): Order of channels. Defaults to None.
        verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
        Union[str, List[str]]: Either a single filepath or ordered list of filepaths
    """
    filtered_df = samples_df
    
    # Handle type conversion for numeric wildcards (Snakemake passes them as strings)
    if plate is not None:
        plate_val = int(plate) if isinstance(plate, str) else plate
        filtered_df = filtered_df[filtered_df["plate"] == plate_val]
    if well is not None:
        well_val = int(well) if isinstance(well, str) else well
        filtered_df = filtered_df[filtered_df["well"] == well_val]
    if tile is not None:
        tile_val = int(tile) if isinstance(tile, str) else tile
        filtered_df = filtered_df[filtered_df["tile"] == tile_val]
    if cycle is not None:
        cycle_val = int(cycle) if isinstance(cycle, str) else cycle
        filtered_df = filtered_df[filtered_df["cycle"] == cycle_val]
    if channel is not None:
        filtered_df = filtered_df[filtered_df["channel"] == channel]

    if round_order is not None:
        # Filter to only include specified rounds
        filtered_df = filtered_df[filtered_df["round"].isin(round_order)]

        # If no data after filtering, return results based on available rounds
        if len(filtered_df) == 0:
            print(
                f"No data found for specified rounds {round_order}. Using available rounds."
            )
            filtered_df = samples_df
            if plate is not None:
                plate_val = int(plate) if isinstance(plate, str) else plate
                filtered_df = filtered_df[filtered_df["plate"] == plate_val]
            if well is not None:
                well_val = int(well) if isinstance(well, str) else well
                filtered_df = filtered_df[filtered_df["well"] == well_val]
            if tile is not None:
                tile_val = int(tile) if isinstance(tile, str) else tile
                filtered_df = filtered_df[filtered_df["tile"] == tile_val]
            if cycle is not None:
                cycle_val = int(cycle) if isinstance(cycle, str) else cycle
                filtered_df = filtered_df[filtered_df["cycle"] == cycle_val]
            if channel is not None:
                filtered_df = filtered_df[filtered_df["channel"] == channel]

        # Create dictionary mapping round to DataFrame rows
        round_groups = {
            round_num: group for round_num, group in filtered_df.groupby("round")
        }

        # Initialize lists to store files and channels
        all_files = []
        final_channel_order = []

        # Get available rounds that exist in the data
        available_rounds = sorted(round_groups.keys())

        # Process each available round
        for round_num in available_rounds:
            round_df = round_groups[round_num]

            # If channel order is specified, get files in that order for this round
            if "channel" in round_df.columns and channel_order is not None:
                # Create mapping of available channels to files for this round
                channel_to_file = dict(zip(round_df["channel"], round_df["sample_fp"]))
                # Add files for each requested channel if available in this round
                for channel in channel_order:
                    if channel in channel_to_file:
                        all_files.append(channel_to_file[channel])
                        final_channel_order.append(f"Round {round_num}: {channel}")
            else:
                # If no channel order, just take the first file from this round
                all_files.append(round_df["sample_fp"].iloc[0])
                if "channel" in round_df.columns:
                    final_channel_order.append(
                        f"Round {round_num}: {round_df['channel'].iloc[0]}"
                    )
                else:
                    final_channel_order.append(f"Round {round_num}")

        if verbose:
            print("\nFinal channel order:")
            for chan in final_channel_order:
                print(f"  {chan}")

        return all_files

    # If no rounds specified but we have channels and channel order
    if "channel" in filtered_df.columns and channel_order is not None:
        channel_to_file = dict(zip(filtered_df["channel"], filtered_df["sample_fp"]))
        return [
            channel_to_file[channel]
            for channel in channel_order
            if channel in channel_to_file
        ]

    # Otherwise return single file path
    return filtered_df["sample_fp"].iloc[0]
