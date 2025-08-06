"""Functions for preprocessing ND2 files in preparation for downstream BrieFlow steps."""

import pandas as pd
import numpy as np
import nd2
from typing import Union, List
from pathlib import Path

# Libraries for OME-Zarr
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image
import xarray as xr


def extract_tile_metadata(
    tile_fp: str,
    plate: int,
    well: str,
    tile: int,
    cycle: int = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extracts metadata from a single ND2 file for a specific tile.

    Args:
        tile_fp (str): File path pointing to the ND2 file for the tile.
        plate (int): Plate number to associate with this metadata.
        well (str): Well to associate with this metadata.
        tile (int): Tile number to associate with this metadata.
        cycle (int, optional): Cycle number to associate with this metadata. Defaults to None.
        z_interval (int, optional): If set, samples z-planes at this interval to ensure metadata is one line per position. Defaults to 4.
        verbose (bool, optional): If True, prints metadata information. Defaults to False.

    Returns:
        pd.DataFrame: Extracted metadata for the given tile.
    """
    if verbose:
        print(f"Processing tile {tile} from file {tile_fp}")

    with nd2.ND2File(tile_fp) as images:
        frame_meta = images.frame_metadata(0)

        if verbose:
            print(f"File shape: {images.shape}")
            print(f"Number of dimensions: {images.ndim}")
            print(f"Data type: {images.dtype}")
            print(f"Sizes (by axes): {images.sizes}")

        # Get position data from first channel's position information
        if frame_meta.channels and hasattr(frame_meta.channels[0], "position"):
            stage_pos = frame_meta.channels[0].position.stagePositionUm
            metadata = {
                "x_pos": stage_pos.x,
                "y_pos": stage_pos.y,
                "z_pos": stage_pos.z,
                "pfs_offset": frame_meta.channels[0].position.pfsOffset,
            }
        else:
            metadata = {
                "x_pos": None,
                "y_pos": None,
                "z_pos": None,
                "pfs_offset": None,
            }

        # Add basic metadata
        metadata.update(
            {
                "plate": plate,
                "well": well,
                "tile": tile,
            }
        )

        # Conditionally add cycle after tile
        if cycle is not None:
            metadata["cycle"] = cycle

        # Add remaining metadata
        metadata.update(
            {
                "filename": tile_fp,
                "channels": frame_meta.contents.channelCount,
            }
        )

        # Get pixel size from first channel's volume information
        if frame_meta.channels and hasattr(frame_meta.channels[0], "volume"):
            x_cal, y_cal, _ = frame_meta.channels[0].volume.axesCalibration
            metadata.update(
                {
                    "pixel_size_x": x_cal,
                    "pixel_size_y": y_cal,
                }
            )
        else:
            metadata.update(
                {
                    "pixel_size_x": None,
                    "pixel_size_y": None,
                }
            )

        df = pd.DataFrame([metadata])

    return df


def nd2_to_tiff(
    files: Union[str, List[str], Path, List[Path]],
    channel_order_flip: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Converts one or multiple ND2 files to a multidimensional numpy array, ensuring CYX structure.

    Args:
        files: Path(s) to the ND2 file(s). Can be a single path or list of paths.
        channel_order_flip: If True, flips the channel order. Defaults to False.
        verbose: If True, prints dimension information. Defaults to False.

    Returns:
        np.ndarray: Image data as a multidimensional numpy array in CYX format.

    Raises:
        ValueError: If files have incompatible dimensions.
    """
    # Convert input to list of Path objects
    if isinstance(files, (str, Path)):
        files = [Path(files)]
    else:
        files = [Path(f) for f in files]

    # Process all files
    image_arrays = []
    for i, file in enumerate(files, 1):
        if verbose:
            print(f"Processing file {i}/{len(files)}: {file}")

        image = nd2.imread(str(file), xarray=True)

        if verbose:
            print(f"Original dimensions for {file}: {image.dims}")

        # Handle Z-stack if present
        if "Z" in image.dims:
            image = image.max(dim="Z")

        # Convert to numpy array based on dimensions present
        if "C" in image.dims:
            # If C dimension exists, ensure CYX order
            img_array = image.transpose("C", "Y", "X").values

            # Flip channel order if needed
            if channel_order_flip:
                img_array = np.flip(img_array, axis=0)
        else:
            # If no C dimension, assume YX and add channel dimension
            img_array = image.transpose("Y", "X").values
            img_array = np.expand_dims(img_array, axis=0)  # Add channel dimension

        if verbose:
            print(f"Array shape after processing: {img_array.shape}")

        # Check dimensions match if not first image
        if image_arrays and img_array.shape[1:] != image_arrays[0].shape[1:]:
            raise ValueError(
                f"File {file} has incompatible dimensions: {img_array.shape} vs {image_arrays[0].shape}"
            )

        image_arrays.append(img_array)

    # Concatenate along channel axis (axis 0)
    result = np.concatenate(image_arrays, axis=0)

    if verbose:
        print(f"Final dimensions (CYX): {result.shape}")

    return result.astype(np.uint16)

def nd2_to_ome_zarr(
    input_file: Union[str, Path],
    output_dir: Union[str, Path],
    chunk_dims: tuple = (1, 1, 256, 256),
    verbose: bool = False,
) -> bool:
    """Converts a single ND2 file to a 5D OME-Zarr file (TCZYX).

    This function reads an ND2 file, extracts its metadata and pixel data,
    and writes it to a pyramid-less OME-Zarr store. It also saves the
    full raw metadata from the ND2 file into a separate JSON file for verification.

    Args:
        input_file (Union[str, Path]): Path to the input ND2 file.
        output_dir (Union[str, Path]): Directory to save the OME-Zarr output.
        chunk_dims (tuple, optional): The chunk size for the Zarr array along (C, Z, Y, X).
                                      Defaults to (1, 1, 256, 256).
        verbose (bool, optional): If True, prints processing information. Defaults to False.
    
    Returns:
        bool: True if conversion was successful, False otherwise.
    
    Raises:
        FileNotFoundError: If input file doesn't exist.
        ValueError: If input file is not a valid ND2 file.
        RuntimeError: If conversion fails due to unsupported dimensions or other issues.
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    # Validate input file
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    if not input_file.suffix.lower() == '.nd2':
        raise ValueError(f"Input file must be an ND2 file, got: {input_file.suffix}")
    
    # Create output directory
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create output directory {output_dir}: {e}")

    if verbose:
        print(f"Processing {input_file.name}...")

    try:
        with nd2.ND2File(input_file) as images:
            # --- 1. Extract data as a dask-backed xarray for memory efficiency ---
            try:
                xarr = images.to_xarray(delayed=True, squeeze=False)
            except Exception as e:
                raise RuntimeError(f"Failed to read ND2 file as xarray: {e}")
            
            if verbose:
                print(f"Original dimensions: {xarr.dims} with shape {xarr.shape}")

            # --- 2. Handle dimension mapping and standardization ---
            # Map common ND2 dimensions to OME-Zarr dimensions
            dim_mapping = {
                'P': 'T',  # Position -> Time (for single position files)
                'T': 'T',  # Time -> Time
                'C': 'C',  # Channel -> Channel
                'Z': 'Z',  # Z -> Z
                'Y': 'Y',  # Y -> Y
                'X': 'X',  # X -> X
            }
            
            # Rename dimensions to standard names
            rename_dict = {}
            for old_dim in xarr.dims:
                if old_dim in dim_mapping:
                    rename_dict[old_dim] = dim_mapping[old_dim]
                else:
                    if verbose:
                        print(f"Warning: Unknown dimension '{old_dim}' will be treated as additional dimension")
            
            if rename_dict:
                xarr = xarr.rename(rename_dict)
                if verbose:
                    print(f"Renamed dimensions: {rename_dict}")
            
            # --- 3. Standardize to 5D TCZYX format ---
            # Handle position dimension by taking first position if multiple exist
            if 'P' in xarr.dims and xarr.dims['P'] > 1:
                if verbose:
                    print(f"Multiple positions detected ({xarr.dims['P']}), using first position")
                xarr = xarr.isel(P=0)
            elif 'P' in xarr.dims:
                # Single position, rename to T
                xarr = xarr.rename({'P': 'T'})
            
            # Ensure all 5 dimensions are present, adding dummy ones if necessary
            for dim in "TCZYX":
                if dim not in xarr.dims:
                    xarr = xarr.expand_dims(dim, axis=0)
                    if verbose:
                        print(f"Added missing dimension '{dim}'")
            
            # Check if we have the correct dimensions for OME-Zarr
            expected_dims = set("TCZYX")
            actual_dims = set(xarr.dims)
            
            if not expected_dims.issubset(actual_dims):
                missing_dims = expected_dims - actual_dims
                extra_dims = actual_dims - expected_dims
                error_msg = f"Cannot convert to OME-Zarr format. "
                if missing_dims:
                    error_msg += f"Missing dimensions: {missing_dims}. "
                if extra_dims:
                    error_msg += f"Extra dimensions: {extra_dims}. "
                error_msg += f"Expected: {expected_dims}, Got: {actual_dims}"
                raise RuntimeError(error_msg)
            
            # Transpose to the standard OME-Zarr order
            try:
                data = xarr.transpose("T", "C", "Z", "Y", "X")
            except Exception as e:
                raise RuntimeError(f"Failed to transpose dimensions to TCZYX order: {e}")

            if verbose:
                print(f"Standardized 5D shape (TCZYX): {data.shape}")

            # --- 4. Prepare OME-Zarr metadata ---
            axes = [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ]

            # Get voxel sizes for scaling transformation
            try:
                voxel_size = images.voxel_size()
                transformations = [
                    {
                        "type": "scale",
                        "scale": [
                            images.experiment[0].period.total_seconds() if images.experiment else 1.0,
                            1.0,  # Channel scale
                            voxel_size.z,
                            voxel_size.y,
                            voxel_size.x,
                        ],
                    }
                ]
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not extract voxel size, using default scaling: {e}")
                transformations = [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0, 1.0, 1.0],
                    }
                ]
            
            # --- 5. Write to OME-Zarr ---
            zarr_path = output_dir / f"{input_file.stem}.zarr"
            try:
                store = parse_url(str(zarr_path), mode="w").store
                root_group = zarr.group(store=store, overwrite=True)

                # Define storage options with chunking
                # Adjust chunks to match the final 5D data shape
                final_chunks = (1,) + chunk_dims # Add time chunk
                
                write_image(
                    image=data.data,  # Pass the underlying dask array
                    group=root_group,
                    axes=axes,
                    storage_options={'chunks': final_chunks},
                )

                if verbose:
                    print(f"Successfully wrote OME-Zarr to: {zarr_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to write OME-Zarr file: {e}")

            # --- 6. Export raw metadata for verification ---
            metadata_filename = output_dir / f"{input_file.stem}_metadata.json"
            try:
                # nd2's repr is quite informative and serializes well
                with open(metadata_filename, "w") as f:
                    f.write(repr(images.metadata))
                if verbose:
                    print(f"Full metadata exported to: {metadata_filename}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not export full metadata due to serialization error: {e}")

            return True

    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        error_msg = f"Unexpected error occurred while processing {input_file.name}: {e}"
        if verbose:
            print(error_msg)
        raise RuntimeError(error_msg)
