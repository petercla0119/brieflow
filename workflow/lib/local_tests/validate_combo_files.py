#!/usr/bin/env python3
"""
Validation script for combo files before Snakemake workflow execution.
This script checks the structure and content of combo files to prevent wildcard errors.
"""

import sys
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComboFileValidator:
    """Validator for combo files to ensure they meet Snakemake wildcard requirements."""
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        # Updated required columns to match new metadata structure
        self.required_columns = {
            'sbs_combo.tsv': ['plate', 'cycle', 'well', 'tile', 'channel'],
            'phenotype_combo.tsv': ['plate', 'round', 'well', 'tile', 'channel']
        }
        # Updated expected data types to match new metadata structure
        self.expected_data_types = {
            'plate': int,
            'well': (int, str),  # Accept both int and str for well column
            'tile': int,
            'cycle': int,
            'round': int,
            'channel': str
        }
    
    def validate_file_structure(self, filename: str) -> Tuple[bool, List[str]]:
        """Validate that a combo file has the required structure."""
        file_path = self.config_dir / filename
        
        if not file_path.exists():
            return False, [f"File {filename} does not exist"]
        
        try:
            df = pd.read_csv(file_path, sep='\t')
        except Exception as e:
            return False, [f"Failed to read {filename}: {str(e)}"]
        
        errors = []
        
        # Check required columns
        required_cols = self.required_columns.get(filename, [])
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {list(missing_cols)}")
        
        # Check for unexpected columns
        unexpected_cols = set(df.columns) - set(required_cols)
        if unexpected_cols:
            errors.append(f"Unexpected columns found: {list(unexpected_cols)}")
        
        # Check data types
        for col in df.columns:
            if col in self.expected_data_types:
                expected_type = self.expected_data_types[col]
                if not self._check_column_type(df[col], expected_type):
                    errors.append(f"Column '{col}' has incorrect data type. Expected {expected_type.__name__}")
        
        # Check for empty data
        if df.empty:
            errors.append("File contains no data")
        
        # Check for missing values in critical columns
        critical_cols = ['plate', 'well', 'tile']
        for col in critical_cols:
            if col in df.columns and df[col].isnull().any():
                errors.append(f"Column '{col}' contains missing values")
        
        return len(errors) == 0, errors
    
    def _check_column_type(self, series: pd.Series, expected_type: type) -> bool:
        """Check if a pandas series matches the expected data type."""
        if expected_type == int:
            # Check if all non-null values can be converted to int
            try:
                series.dropna().astype(int)
                return True
            except (ValueError, TypeError):
                return False
        elif expected_type == str:
            # Check if all non-null values are strings
            return series.dropna().apply(lambda x: isinstance(x, str)).all()
        elif expected_type == (int, str):
            # Accept both int and str for well column
            try:
                # Check if all non-null values can be converted to int
                series.dropna().astype(int)
                return True
            except (ValueError, TypeError):
                # If not int, check if they're strings
                return series.dropna().apply(lambda x: isinstance(x, str)).all()
        return True
    
    def validate_wildcard_consistency(self) -> Tuple[bool, List[str]]:
        """Validate that wildcard combinations are consistent across files."""
        errors = []
        
        # Load both files
        try:
            sbs_df = pd.read_csv(self.config_dir / 'sbs_combo.tsv', sep='\t')
            phenotype_df = pd.read_csv(self.config_dir / 'phenotype_combo.tsv', sep='\t')
        except Exception as e:
            return False, [f"Failed to load combo files: {str(e)}"]
        
        # Check that plate numbers are consistent
        sbs_plates = set(sbs_df['plate'].unique())
        phenotype_plates = set(phenotype_df['plate'].unique())
        
        if sbs_plates != phenotype_plates:
            errors.append(f"Plate numbers inconsistent: SBS has {sbs_plates}, Phenotype has {phenotype_plates}")
        
        # Check that well numbers are consistent
        sbs_wells = set(sbs_df['well'].unique())
        phenotype_wells = set(phenotype_df['well'].unique())
        
        if sbs_wells != phenotype_wells:
            errors.append(f"Well numbers inconsistent: SBS has {sbs_wells}, Phenotype has {phenotype_wells}")
        
        # Check that tile numbers are consistent
        sbs_tiles = set(sbs_df['tile'].unique())
        phenotype_tiles = set(phenotype_df['tile'].unique())
        
        # Note: SBS and phenotype files may have different tile coverage
        # This is not an error, just different datasets
        if sbs_tiles != phenotype_tiles:
            print(f"ℹ️  Tile numbers differ between SBS and phenotype files:")
            print(f"   SBS tiles: {sorted(sbs_tiles)}")
            print(f"   Phenotype tiles: {sorted(phenotype_tiles)}")
            print(f"   This is normal - different datasets may have different tile coverage")
            # Don't treat this as an error
        
        # Check that channel names are consistent
        sbs_channels = set(sbs_df['channel'].unique())
        phenotype_channels = set(phenotype_df['channel'].unique())
        
        if sbs_channels != phenotype_channels:
            errors.append(f"Channel names inconsistent: SBS has {sbs_channels}, Phenotype has {phenotype_channels}")
        
        return len(errors) == 0, errors
    
    def validate_data_integrity(self) -> Tuple[bool, List[str]]:
        """Validate data integrity and consistency."""
        errors = []
        
        # Load both files
        try:
            sbs_df = pd.read_csv(self.config_dir / 'sbs_combo.tsv', sep='\t')
            phenotype_df = pd.read_csv(self.config_dir / 'phenotype_combo.tsv', sep='\t')
        except Exception as e:
            return False, [f"Failed to load combo files: {str(e)}"]
        
        # Check for duplicate combinations
        sbs_duplicates = sbs_df.duplicated().sum()
        phenotype_duplicates = phenotype_df.duplicated().sum()
        
        if sbs_duplicates > 0:
            errors.append(f"SBS combo file has {sbs_duplicates} duplicate rows")
        
        if phenotype_duplicates > 0:
            errors.append(f"Phenotype combo file has {phenotype_duplicates} duplicate rows")
        
        # Check that all combinations have the expected number of channels
        expected_channels = ['DAPI', 'Cy3', 'Far-red', 'GFP']
        
        # Check SBS combinations
        for _, row in sbs_df.groupby(['plate', 'cycle', 'well', 'tile']).size().reset_index().iterrows():
            plate, cycle, well, tile = row['plate'], row['cycle'], row['well'], row['tile']
            channels = sbs_df[(sbs_df['plate'] == plate) & 
                             (sbs_df['cycle'] == cycle) & 
                             (sbs_df['well'] == well) & 
                             (sbs_df['tile'] == tile)]['channel'].tolist()
            
            if len(channels) != len(expected_channels):
                errors.append(f"SBS combination (plate={plate}, cycle={cycle}, well={well}, tile={tile}) "
                            f"has {len(channels)} channels, expected {len(expected_channels)}")
        
        # Check phenotype combinations
        for _, row in phenotype_df.groupby(['plate', 'round', 'well', 'tile']).size().reset_index().iterrows():
            plate, round_num, well, tile = row['plate'], row['round'], row['well'], row['tile']
            channels = phenotype_df[(phenotype_df['plate'] == plate) & 
                                  (phenotype_df['round'] == round_num) & 
                                  (phenotype_df['well'] == well) & 
                                  (phenotype_df['tile'] == tile)]['channel'].tolist()
            
            if len(channels) != len(expected_channels):
                errors.append(f"Phenotype combination (plate={plate}, round={round_num}, well={well}, tile={tile}) "
                            f"has {len(channels)} channels, expected {len(expected_channels)}")
        
        return len(errors) == 0, errors
    
    def generate_validation_report(self) -> Dict[str, any]:
        """Generate a comprehensive validation report."""
        report = {
            'overall_status': 'PASS',
            'file_validation': {},
            'wildcard_consistency': False,
            'data_integrity': False,
            'errors': []
        }
        
        # Validate individual files
        for filename in self.required_columns.keys():
            is_valid, errors = self.validate_file_structure(filename)
            report['file_validation'][filename] = {
                'status': 'PASS' if is_valid else 'FAIL',
                'errors': errors
            }
            if not is_valid:
                report['overall_status'] = 'FAIL'
                report['errors'].extend([f"{filename}: {error}" for error in errors])
        
        # Validate wildcard consistency
        wildcard_valid, wildcard_errors = self.validate_wildcard_consistency()
        report['wildcard_consistency'] = wildcard_valid
        if not wildcard_valid:
            report['overall_status'] = 'FAIL'
            report['errors'].extend([f"Wildcard consistency: {error}" for error in wildcard_errors])
        
        # Validate data integrity
        data_valid, data_errors = self.validate_data_integrity()
        report['data_integrity'] = data_valid
        if not data_valid:
            report['overall_status'] = 'FAIL'
            report['errors'].extend([f"Data integrity: {error}" for error in data_errors])
        
        return report
    
    def print_validation_report(self, report: Dict[str, any]):
        """Print a formatted validation report."""
        print("=" * 60)
        print("COBO FILES VALIDATION REPORT")
        print("=" * 60)
        print(f"Overall Status: {report['overall_status']}")
        print()
        
        print("File Validation Results:")
        print("-" * 30)
        for filename, result in report['file_validation'].items():
            status_icon = "✅" if result['status'] == 'PASS' else "❌"
            print(f"{filename}: {status_icon} {result['status']}")
            if result['errors']:
                for error in result['errors']:
                    print(f"  └─ {error}")
        
        print()
        print("Validation Errors:")
        print("-" * 20)
        if report['errors']:
            for error in report['errors']:
                print(f"❌ {error}")
        else:
            print("✅ No validation errors found")
        
        print()
        print("Recommendations:")
        print("-" * 20)
        if report['overall_status'] == 'PASS':
            print("✅ All validation checks passed")
            print("✅ Combo files are ready for workflow execution")
        else:
            print("1. Fix the errors listed above")
            print("2. Ensure all required columns are present")
            print("3. Check data types match expected values")
            print("4. Verify wildcard consistency across files")
            print("5. Run validation again before executing workflow")
        
        print("=" * 60)


def main():
    """Main function to run validation."""
    if len(sys.argv) > 1:
        config_dir = Path(sys.argv[1])
    else:
        config_dir = Path("config")
    
    if not config_dir.exists():
        print(f"Error: Config directory {config_dir} does not exist")
        sys.exit(1)
    
    validator = ComboFileValidator(config_dir)
    report = validator.generate_validation_report()
    validator.print_validation_report(report)
    
    # Exit with appropriate code
    sys.exit(0 if report['overall_status'] == 'PASS' else 1)


if __name__ == "__main__":
    main() 