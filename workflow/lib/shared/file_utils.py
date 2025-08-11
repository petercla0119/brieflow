"""Utility functions for handling and filtering sample file paths in the BrieFlow pipeline."""

from pathlib import Path

from pyarrow.parquet import ParquetFile
import pyarrow as pa
import pandas as pd
import numpy as np

# Mapping of metadata keys to filename prefixes and data types
FILENAME_METADATA_MAPPING = {
    "plate": ["P-", str],
    "well": ["W-", str],
    "tile": ["T-", int],
    "cycle": ["C-", int],
    "cell_class": ["CeCl-", str],
    "channel_combo": ["ChCo-", str],
    "gene": ["G-", str],
    "sgrna": ["SG-", str],
    "channel": ["CH-", str],
    "leiden_resolution": ["LR-", float],
    "cluster_benchmark": ["CB-", str],
}


def get_filename(data_location: dict, info_type: str, file_type: str) -> str:
    """Generate a structured filename based on data location, information type, and file type.

    Args:
        data_location (dict): Dictionary containing location info like well, tile, and cycle.
        info_type (str): Type of information (e.g., 'cell_features', 'sbs_reads').
        file_type (str): File extension/type (e.g., 'tsv', 'parquet', 'tiff').

    Returns:
        str: Structured filename.
    """
    parts = []

    for metadata_key, metadata_value in data_location.items():
        if metadata_key in FILENAME_METADATA_MAPPING:
            prefix, _ = FILENAME_METADATA_MAPPING[metadata_key]
            parts.append(f"{prefix}{metadata_value}")
        else:
            print(f"Unknown metadata key: {metadata_key}")

    prefix = "_".join(parts)
    filename = (
        f"{prefix}__{info_type}.{file_type}" if prefix else f"{info_type}.{file_type}"
    )

    return filename


def parse_filename(file_path: str) -> tuple:
    """Parse a structured filename from a file path to extract data location, information type, and file type.

    Args:
        file_path (str): Full file path or filename, e.g., '/path/to/W_A1_T02_C03__cell_features.tsv'.

    Returns:
        tuple: A tuple containing:
            - metadata (dict): Dictionary with keys like 'well', 'tile', 'cycle' as applicable.
            - info_type (str): The type of information (e.g., 'cell_features').
            - file_type (str): The file extension/type (e.g., 'tsv').
    """
    # Convert the input to a Path object
    path = Path(file_path)

    # Extract the filename and file extension
    filename = path.stem
    file_type = path.suffix.lstrip(".")

    # Split the filename into main parts
    parts = filename.split("__")

    # Initialize metadata dictionary and info_type variable
    metadata = {}
    info_type = None

    # Parse data location part
    if len(parts) == 2:
        location_part, info_type = parts
        elements = location_part.split("_")

        for element in elements:
            for key, (prefix, data_type) in FILENAME_METADATA_MAPPING.items():
                if element.startswith(prefix):
                    # Extract and convert the value based on the data type
                    value = element[len(prefix) :]
                    metadata[key] = data_type(value)
                    break  # Stop checking other prefixes for this element
    else:
        # If no location part, the first part is the info_type
        info_type = parts[0]

    return metadata, info_type, file_type


def load_parquet_subset(full_df_fp, n_rows=50000):
    """Load a fixed number of rows from an parquet file without loading entire file into memory.

    Args:
        full_df_fp (str): Path to parquet file.
        n_rows (int): Number of rows to get.

    Returns:
        pd.DataFrame: Subset of the data with combined blocks.
    """
    print(f"Reading first {n_rows:,} rows from {full_df_fp}")

    # read the first n_rows of the file path
    df = ParquetFile(full_df_fp)
    row_subset = next(df.iter_batches(batch_size=n_rows))
    df = pa.Table.from_batches([row_subset]).to_pandas()

    return df


def validate_dtypes(df):
    """Convert DataFrame columns to the most specific data type possible with the following rules.

    - Convert object to bool or string if possible
    - Convert strings to int float if possible
    - Convert floats to int if possible

    Args:
        df : pandas.DataFrame
            The DataFrame to optimize

    Returns:
        pandas.DataFrame
            A new DataFrame with optimized dtypes
    """
    for col in df.columns:
        # Skip columns that are already int
        if pd.api.types.is_integer_dtype(df[col]):
            continue

        # Convert object to bool if possible, else to string
        if pd.api.types.is_object_dtype(df[col]):
            lowered = df[col].dropna().astype(str).str.lower()
            if lowered.isin(["true", "false"]).all():
                df[col] = (
                    df[col].astype(str).str.lower().map({"true": True, "false": False})
                )
            else:
                try:
                    df[col] = df[col].astype("string")
                except ValueError:
                    pass

        # Convert string to float if possible
        if pd.api.types.is_string_dtype(df[col]):
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                pass

        # Convert float to int if possible
        if pd.api.types.is_float_dtype(df[col]):
            col_nonan = df[col].dropna()
            if len(col_nonan) == 0 or np.allclose(
                col_nonan, col_nonan.round(), rtol=1e-10, atol=1e-10
            ):
                try:
                    df[col] = df[col].astype("Int64")
                except TypeError:
                    pass

    return df


def fix_combo_file(input_file: Path, output_file: Path, plate_value: str = "1") -> None:
    """Add missing 'plate' column to combo TSV file.
    
    This function resolves the Snakemake wildcard error: "No values given for wildcard 'plate'"
    by ensuring the combo TSV files have the required 'plate' column.
    
    Args:
        input_file (Path): Path to the input TSV file that may be missing the 'plate' column.
        output_file (Path): Path where the fixed TSV file will be saved.
        plate_value (str): The value to use for the 'plate' column. Defaults to "1".
    
    Returns:
        None: The function modifies the file in place.
    """
    print(f"Processing {input_file}...")
    
    # Read the TSV file
    df = pd.read_csv(input_file, sep='\t')
    print(f"Original columns: {list(df.columns)}")
    
    # Check if plate column already exists
    if 'plate' in df.columns:
        print(f"Plate column already exists in {input_file}")
        return
    
    # Add plate column at the beginning
    df.insert(0, 'plate', plate_value)
    print(f"Added 'plate' column with value '{plate_value}'")
    
    # Save the fixed file
    df.to_csv(output_file, sep='\t', index=False)
    print(f"Fixed file saved to {output_file}")


def fix_combo_files(config_dir: Path = None, plate_value: str = "1", backup: bool = True) -> None:
    """Fix missing 'plate' column in both combo TSV files.
    
    This function resolves the Snakemake wildcard error: "No values given for wildcard 'plate'"
    by ensuring both sbs_combo.tsv and phenotype_combo.tsv have the required 'plate' column.
    
    Args:
        config_dir (Path): Directory containing the combo TSV files. 
                          Defaults to current working directory / "config".
        plate_value (str): The value to use for the 'plate' column. Defaults to "1".
        backup (bool): Whether to create backup files before overwriting. Defaults to True.
    
    Returns:
        None: The function modifies the files in place.
    """
    if config_dir is None:
        config_dir = Path.cwd() / "config"
    
    print(f"Fixing combo files in: {config_dir}")
    
    # Fix SBS combo file
    sbs_input = config_dir / "sbs_combo.tsv"
    sbs_output = config_dir / "sbs_combo_fixed.tsv"
    
    if sbs_input.exists():
        if backup:
            backup_file = config_dir / "sbs_combo.tsv.backup"
            backup_file.write_text(sbs_input.read_text())
            print(f"Original SBS file backed up to: {backup_file}")
        
        fix_combo_file(sbs_input, sbs_output, plate_value)
        
        # Replace original with fixed version
        sbs_output.replace(sbs_input)
        print(f"SBS combo file fixed successfully")
    else:
        print(f"Warning: {sbs_input} not found")
    
    # Fix phenotype combo file
    phenotype_input = config_dir / "phenotype_combo.tsv"
    phenotype_output = config_dir / "phenotype_combo_fixed.tsv"
    
    if phenotype_input.exists():
        if backup:
            backup_file = config_dir / "phenotype_combo.tsv.backup"
            backup_file.write_text(phenotype_input.read_text())
            print(f"Original phenotype file backed up to: {backup_file}")
        
        fix_combo_file(phenotype_input, phenotype_output, plate_value)
        
        # Replace original with fixed version
        phenotype_output.replace(phenotype_input)
        print(f"Phenotype combo file fixed successfully")
    else:
        print(f"Warning: {phenotype_input} not found")
    
    print("\n Combo files fixed successfully!")
    print("You can now run the preprocessing script again.")


