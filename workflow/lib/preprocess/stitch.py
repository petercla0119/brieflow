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
