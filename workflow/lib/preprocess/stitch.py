"""Stitching adapter module for Brieflow preprocessing.

This module provides a thin wrapper around the external GPU-accelerated stitching
library, adapting it for use within Brieflow's preprocessing pipeline.

Key functions:
- validate_stitch_config(): Validate stitch configuration parameters
- is_gpu_available(): Check if GPU acceleration is available
- estimate_stitch_from_tiles(): Estimate tile positions via phase correlation
- estimate_stitch_from_metadata(): Estimate positions from stage coordinates
- stitch_tiles_to_well(): Assemble tiles into stitched well image
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import warnings


# GPU availability detection with lazy evaluation
_GPU_AVAILABLE: Optional[bool] = None


def is_gpu_available() -> bool:
    """Check if GPU acceleration is available for stitching operations.

    Performs a one-time check for CuPy availability and GPU accessibility.
    Result is cached for subsequent calls.

    Returns:
        True if CuPy is installed and a GPU is accessible, False otherwise.
    """
    global _GPU_AVAILABLE

    if _GPU_AVAILABLE is not None:
        return _GPU_AVAILABLE

    try:
        import cupy as cp

        # Actually test GPU access
        _ = cp.array([1.0])
        _GPU_AVAILABLE = True
    except (ImportError, ModuleNotFoundError):
        _GPU_AVAILABLE = False
    except Exception:
        # CuPy installed but GPU not accessible
        _GPU_AVAILABLE = False

    return _GPU_AVAILABLE


def get_compute_backend() -> str:
    """Get the current compute backend for stitching operations.

    Returns:
        'gpu' if CuPy/CUDA is available, 'cpu' otherwise.
    """
    return "gpu" if is_gpu_available() else "cpu"


# Valid configuration values
VALID_METHODS = {"phase_correlation", "coordinate_based"}
VALID_OUTPUT_FORMATS = {"omezarr"}
VALID_BLENDING_METHODS = {"edt", "average"}


def validate_stitch_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize stitch configuration.

    Checks that all required parameters are present and valid, applies defaults
    for optional parameters, and validates parameter combinations.

    Args:
        config: Stitch configuration dictionary. Expected structure:
            {
                "enabled": bool,
                "method": str,  # "phase_correlation" or "coordinate_based"
                "use_gpu": bool,
                "overlap_pixels": int,
                "flipud": bool,
                "fliplr": bool,
                "rot90": int,
                "output_format": str,  # "omezarr"
                "blending_method": str,  # "edt" or "average"
                "phenotype": {"enabled": bool, "reference_channel": int},
                "sbs": {"enabled": bool, "reference_cycle": int, "reference_channel": int}
            }

    Returns:
        Validated and normalized configuration dictionary with defaults applied.

    Raises:
        ValueError: If configuration is invalid.
    """
    if not isinstance(config, dict):
        raise ValueError("Stitch config must be a dictionary")

    validated = {}

    # Required: enabled flag (defaults to False)
    validated["enabled"] = bool(config.get("enabled", False))

    if not validated["enabled"]:
        # If disabled, return minimal valid config
        return validated

    # Method: phase_correlation or coordinate_based
    method = config.get("method", "phase_correlation")
    if method not in VALID_METHODS:
        raise ValueError(
            f"Invalid stitch method '{method}'. Must be one of: {VALID_METHODS}"
        )
    validated["method"] = method

    # GPU usage
    use_gpu = config.get("use_gpu", True)
    if use_gpu and not is_gpu_available():
        warnings.warn(
            "GPU requested but not available. Falling back to CPU processing."
        )
        use_gpu = False
    validated["use_gpu"] = use_gpu

    # Overlap pixels (must be positive)
    overlap = config.get("overlap_pixels", 150)
    if not isinstance(overlap, int) or overlap <= 0:
        raise ValueError(f"overlap_pixels must be a positive integer, got {overlap}")
    validated["overlap_pixels"] = overlap

    # Image augmentation parameters
    validated["flipud"] = bool(config.get("flipud", False))
    validated["fliplr"] = bool(config.get("fliplr", False))

    rot90 = config.get("rot90", 0)
    if not isinstance(rot90, int) or rot90 not in {0, 1, 2, 3}:
        raise ValueError(f"rot90 must be 0, 1, 2, or 3, got {rot90}")
    validated["rot90"] = rot90

    # Output format
    output_format = config.get("output_format", "omezarr")
    if output_format not in VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output_format '{output_format}'. Must be one of: {VALID_OUTPUT_FORMATS}"
        )
    validated["output_format"] = output_format

    # Blending method
    blending = config.get("blending_method", "edt")
    if blending not in VALID_BLENDING_METHODS:
        raise ValueError(
            f"Invalid blending_method '{blending}'. Must be one of: {VALID_BLENDING_METHODS}"
        )
    validated["blending_method"] = blending

    # Phenotype-specific settings
    phenotype_config = config.get("phenotype", {})
    validated["phenotype"] = {
        "enabled": bool(phenotype_config.get("enabled", True)),
        "reference_channel": int(phenotype_config.get("reference_channel", 0)),
    }

    # SBS-specific settings
    sbs_config = config.get("sbs", {})
    validated["sbs"] = {
        "enabled": bool(sbs_config.get("enabled", True)),
        "reference_cycle": int(sbs_config.get("reference_cycle", 1)),
        "reference_channel": int(sbs_config.get("reference_channel", 0)),
    }

    return validated


def get_stitch_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and validate stitch configuration from main config.

    Args:
        config: Main Brieflow configuration dictionary.

    Returns:
        Validated stitch configuration dictionary.
    """
    preprocess_config = config.get("preprocess", {})
    stitch_config = preprocess_config.get("stitch", {})
    return validate_stitch_config(stitch_config)


def is_stitching_enabled(config: Dict[str, Any], image_type: str = None) -> bool:
    """Check if stitching is enabled in the configuration.

    Args:
        config: Main Brieflow configuration dictionary.
        image_type: Optional image type ('phenotype' or 'sbs') to check
                   type-specific enablement.

    Returns:
        True if stitching is enabled (globally and for the specific image type
        if provided).
    """
    stitch_config = get_stitch_config(config)

    if not stitch_config.get("enabled", False):
        return False

    if image_type is not None:
        type_config = stitch_config.get(image_type, {})
        return type_config.get("enabled", True)

    return True


def estimate_stitch_from_metadata(
    metadata_df: "pd.DataFrame",
    tile_size: Tuple[int, int],
    pixel_size: float,
    well: str,
    output_path: Union[str, Path],
) -> Dict[str, List[int]]:
    """Estimate tile positions from stage coordinate metadata.

    Converts microscope stage coordinates (in µm) to pixel positions for stitching.
    This is faster than phase correlation but may be less accurate if stage
    coordinates are imprecise.

    Args:
        metadata_df: DataFrame with columns 'tile', 'x_pos', 'y_pos' containing
                    stage coordinates in micrometers.
        tile_size: Tuple of (height, width) in pixels for each tile.
        pixel_size: Physical pixel size in µm/pixel.
        well: Well identifier (e.g., "A/01" for HCS layout).
        output_path: Path to write the stitch configuration YAML file.

    Returns:
        Dictionary mapping tile paths to [y_shift, x_shift] in pixels.

    Raises:
        ValueError: If required columns are missing or coordinates are invalid.
    """
    import pandas as pd
    import yaml

    if not isinstance(metadata_df, pd.DataFrame):
        raise ValueError("metadata_df must be a pandas DataFrame")

    required_cols = {"tile", "x_pos", "y_pos"}
    missing = required_cols - set(metadata_df.columns)
    if missing:
        raise ValueError(f"metadata_df missing required columns: {missing}")

    # Filter out rows with missing coordinates
    valid_df = metadata_df.dropna(subset=["x_pos", "y_pos"])
    if len(valid_df) == 0:
        raise ValueError("No valid stage coordinates found in metadata")

    # Convert stage coordinates (µm) to pixel positions
    # Stage coordinates are typically absolute positions; we need relative shifts
    x_coords = valid_df["x_pos"].values / pixel_size
    y_coords = valid_df["y_pos"].values / pixel_size

    # Normalize to origin (minimum becomes 0)
    x_shifts = x_coords - x_coords.min()
    y_shifts = y_coords - y_coords.min()

    # Round to integers
    x_shifts = [int(round(x)) for x in x_shifts]
    y_shifts = [int(round(y)) for y in y_shifts]

    # Build shift dictionary with tile path format expected by stitching library
    # Format: "well/tile_name" -> [y_shift, x_shift]
    shifts = {}
    for i, row in enumerate(valid_df.itertuples()):
        tile_name = _format_tile_name(row.tile)
        tile_path = f"{well}/{tile_name}"
        shifts[tile_path] = [y_shifts[i], x_shifts[i]]

    # Write configuration file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    config_data = {
        "total_translation": shifts,
        "method": "coordinate_based",
        "pixel_size_um": pixel_size,
        "tile_size": list(tile_size),
    }

    with open(output_path, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    return shifts


def estimate_stitch_from_tiles(
    input_store_path: Union[str, Path],
    output_path: Union[str, Path],
    tile_size: Tuple[int, int],
    overlap_pixels: int = 150,
    flipud: bool = False,
    fliplr: bool = False,
    rot90: int = 0,
    reference_channel: int = 0,
    limit_positions: Optional[int] = None,
) -> Dict[str, List[int]]:
    """Estimate tile positions using phase correlation registration.

    Uses GPU-accelerated phase correlation to find optimal tile alignment.
    More accurate than coordinate-based estimation but slower.

    Args:
        input_store_path: Path to OME-Zarr store containing tile images.
        output_path: Path to write the stitch configuration YAML file.
        tile_size: Tuple of (height, width) in pixels for each tile.
        overlap_pixels: Expected overlap between adjacent tiles in pixels.
        flipud: Flip tiles vertically before registration.
        fliplr: Flip tiles horizontally before registration.
        rot90: Number of 90-degree rotations to apply to tiles.
        reference_channel: Channel index to use for registration.
        limit_positions: Optional limit on number of positions to process
                        (useful for testing/debugging).

    Returns:
        Dictionary mapping tile paths to [y_shift, x_shift] in pixels.

    Raises:
        ImportError: If stitch library is not available.
        FileNotFoundError: If input store does not exist.
    """
    try:
        from stitch.stitch.assemble import estimate_stitch
    except ImportError as e:
        raise ImportError(
            "Stitch library not found. Install with: pip install stitch"
        ) from e

    input_store_path = Path(input_store_path)
    output_path = Path(output_path)

    if not input_store_path.exists():
        raise FileNotFoundError(f"Input store not found: {input_store_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Call the stitching library's estimate function
    shifts = estimate_stitch(
        input_store_path=str(input_store_path),
        output_config_path=output_path,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        tile_size=tile_size,
        overlap=overlap_pixels,
        limit_positions=limit_positions,
    )

    return shifts


def _format_tile_name(tile_id: Union[int, str]) -> str:
    """Format tile identifier to 6-digit name used by stitching library.

    The stitching library expects tile names in format 'RRRCC' where:
    - RRR: 3-digit row number (000-999)
    - CCC: 3-digit column number (000-999)

    For simple numeric tile IDs, we convert to a row-major grid layout.

    Args:
        tile_id: Tile identifier (integer or string).

    Returns:
        6-digit tile name string.
    """
    if isinstance(tile_id, str):
        # If already formatted, return as-is
        if len(tile_id) == 6 and tile_id.isdigit():
            return tile_id
        # Try to parse as integer
        try:
            tile_id = int(tile_id)
        except ValueError:
            # Return original if can't parse
            return str(tile_id)

    # Convert integer to row/col assuming row-major order
    # This assumes a square-ish grid; actual layout depends on acquisition
    tile_id = int(tile_id)

    # Simple linear mapping for now - row = tile_id, col = 0
    # This will be overridden by actual grid layout in metadata
    row = tile_id
    col = 0

    return f"{row:03d}{col:03d}"
