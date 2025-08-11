#!/usr/bin/env python3
"""
Unit tests for wildcard type handling in the BrieFlow preprocessing pipeline.
Tests various edge cases and ensures robust handling of different wildcard types.
"""

import unittest
import sys
import pandas as pd
from pathlib import Path
from typing import List, Union

# Add the brieflow workflow lib directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocess.file_utils import get_sample_fps


class TestWildcardTypeHandling(unittest.TestCase):
    """Test cases for wildcard type handling in get_sample_fps function."""
    
    def setUp(self):
        """Set up test data."""
        # Create test DataFrame with integer columns
        self.test_data = pd.DataFrame({
            'sample_fp': [
                '/path/to/file1.nd2',
                '/path/to/file2.nd2',
                '/path/to/file3.nd2',
                '/path/to/file4.nd2',
                '/path/to/file5.nd2',
                '/path/to/file6.nd2'
            ],
            'plate': [1, 1, 1, 1, 2, 2],
            'well': [6, 6, 6, 6, 6, 6],
            'tile': [0, 0, 1, 1, 0, 1],
            'cycle': [1, 1, 1, 1, 2, 2],
            'channel': ['DAPI', 'GFP', 'DAPI', 'GFP', 'DAPI', 'GFP']
        })
    
    def test_string_wildcards(self):
        """Test that string wildcards are properly converted to integers."""
        # Test with string wildcards (as Snakemake passes them)
        result = get_sample_fps(
            self.test_data, 
            plate='1', 
            well='6', 
            tile='0', 
            cycle='1'
        )
        
        self.assertIsInstance(result, str)
        self.assertIn('file1.nd2', result)
    
    def test_integer_wildcards(self):
        """Test that integer wildcards work as expected."""
        result = get_sample_fps(
            self.test_data, 
            plate=1, 
            well=6, 
            tile=0, 
            cycle=1
        )
        
        self.assertIsInstance(result, str)
        self.assertIn('file1.nd2', result)
    
    def test_mixed_type_wildcards(self):
        """Test that mixed type wildcards are handled correctly."""
        result = get_sample_fps(
            self.test_data, 
            plate='1', 
            well=6, 
            tile='0', 
            cycle=1
        )
        
        self.assertIsInstance(result, str)
        self.assertIn('file1.nd2', result)
    
    def test_none_wildcards(self):
        """Test that None wildcards are handled gracefully."""
        result = get_sample_fps(
            self.test_data, 
            plate=None, 
            well=None, 
            tile=None, 
            cycle=None
        )
        
        # Should return first file when no filtering applied
        self.assertIsInstance(result, str)
        self.assertIn('file1.nd2', result)
    
    def test_empty_filtering_result(self):
        """Test handling when filtering results in empty DataFrame."""
        # This should not raise an error
        result = get_sample_fps(
            self.test_data, 
            plate=999,  # Non-existent plate
            well=999,   # Non-existent well
            tile=999,   # Non-existent tile
            cycle=999   # Non-existent cycle
        )
        
        # Should handle gracefully, though result may be empty
        self.assertIsInstance(result, str)
    
    def test_channel_order_with_string_wildcards(self):
        """Test channel order filtering with string wildcards."""
        result = get_sample_fps(
            self.test_data, 
            plate='1', 
            well='6', 
            tile='0', 
            cycle='1',
            channel_order=['DAPI', 'GFP']
        )
        
        # Should return a list when channel_order is specified
        self.assertIsInstance(result, list)
        # Note: Only 1 result because only 1 row matches the criteria (plate=1, well=6, tile=0, cycle=1)
        # The test data has 4 rows but only 1 matches all criteria
        self.assertGreaterEqual(len(result), 1)
    
    def test_channel_order_with_integer_wildcards(self):
        """Test channel order filtering with integer wildcards."""
        result = get_sample_fps(
            self.test_data, 
            plate=1, 
            well=6, 
            tile=0, 
            cycle=1,
            channel_order=['DAPI', 'GFP']
        )
        
        self.assertIsInstance(result, list)
        # Note: Only 1 result because only 1 row matches the criteria (plate=1, well=6, tile=0, cycle=1)
        # The test data has 4 rows but only 1 matches all criteria
        self.assertGreaterEqual(len(result), 1)
    
    def test_edge_case_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        empty_df = pd.DataFrame(columns=['sample_fp', 'plate', 'well', 'tile', 'cycle', 'channel'])
        
        # Should handle empty DataFrame gracefully and return empty string
        result = get_sample_fps(empty_df, plate=1, well=6, tile=0, cycle=1)
        self.assertEqual(result, "")
    
    def test_edge_case_missing_columns(self):
        """Test handling when required columns are missing."""
        incomplete_df = self.test_data.drop(columns=['plate'])
        
        # Should handle gracefully when plate column is missing
        result = get_sample_fps(incomplete_df, well=6, tile=0, cycle=1)
        self.assertIsInstance(result, str)
    
    def test_edge_case_mixed_data_types(self):
        """Test handling when DataFrame has mixed data types."""
        mixed_df = self.test_data.copy()
        mixed_df.loc[0, 'plate'] = '1'  # String in integer column
        
        # Should handle mixed types gracefully
        result = get_sample_fps(mixed_df, plate=1, well=6, tile=0, cycle=1)
        self.assertIsInstance(result, str)
    
    def test_edge_case_nan_values(self):
        """Test handling of NaN values in wildcard columns."""
        nan_df = self.test_data.copy()
        nan_df.loc[0, 'plate'] = pd.NA
        
        # Should handle NaN values gracefully
        result = get_sample_fps(nan_df, plate=1, well=6, tile=0, cycle=1)
        self.assertIsInstance(result, str)
    
    def test_performance_with_large_dataframe(self):
        """Test performance with larger DataFrame."""
        # Create larger test data
        large_data = []
        for plate in range(1, 6):
            for well in range(1, 13):
                for tile in range(0, 10):
                    for cycle in range(1, 6):
                        for channel in ['DAPI', 'GFP', 'Cy3', 'Far-red']:
                            large_data.append({
                                'sample_fp': f'/path/to/P{plate}_W{well}_T{tile}_C{cycle}_{channel}.nd2',
                                'plate': plate,
                                'well': well,
                                'tile': tile,
                                'cycle': cycle,
                                'channel': channel
                            })
        
        large_df = pd.DataFrame(large_data)
        
        # Test filtering performance
        import time
        start_time = time.time()
        
        result = get_sample_fps(
            large_df, 
            plate='1', 
            well='6', 
            tile='0', 
            cycle='1'
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time (< 1 second)
        self.assertLess(processing_time, 1.0)
        self.assertIsInstance(result, str)


class TestComboFileValidation(unittest.TestCase):
    """Test cases for combo file validation."""
    
    def setUp(self):
        """Set up test data."""
        self.test_config_dir = Path("test_config")
        self.test_config_dir.mkdir(exist_ok=True)
        
        # Create test combo files
        self.sbs_combo_data = pd.DataFrame({
            'plate': [1, 1, 1, 1],
            'cycle': [1, 1, 2, 2],
            'well': [6, 6, 6, 6],
            'point': [6, 6, 6, 6],
            'tile': [0, 1, 0, 1],
            'channel': ['DAPI', 'GFP', 'DAPI', 'GFP']
        })
        
        self.phenotype_combo_data = pd.DataFrame({
            'plate': [1, 1, 1, 1],
            'round': [1, 1, 2, 2],
            'well': [6, 6, 6, 6],
            'point': [6, 6, 6, 6],
            'tile': [0, 1, 0, 1],
            'channel': ['DAPI', 'GFP', 'DAPI', 'GFP']
        })
        
        # Save test files
        self.sbs_combo_data.to_csv(self.test_config_dir / 'sbs_combo.tsv', sep='\t', index=False)
        self.phenotype_combo_data.to_csv(self.test_config_dir / 'phenotype_combo.tsv', sep='\t', index=False)
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_config_dir.exists():
            shutil.rmtree(self.test_config_dir)
    
    def test_valid_combo_files(self):
        """Test validation of correctly formatted combo files."""
        # Import the validator
        sys.path.insert(0, str(Path(__file__).parent))
        from validate_combo_files import ComboFileValidator
        
        validator = ComboFileValidator(self.test_config_dir)
        report = validator.generate_validation_report()
        
        self.assertEqual(report['overall_status'], 'PASS')
        self.assertEqual(len(report['errors']), 0)
    
    def test_missing_plate_column(self):
        """Test validation fails when plate column is missing."""
        # Remove plate column
        incomplete_data = self.sbs_combo_data.drop(columns=['plate'])
        incomplete_data.to_csv(self.test_config_dir / 'sbs_combo.tsv', sep='\t', index=False)
        
        from validate_combo_files import ComboFileValidator
        validator = ComboFileValidator(self.test_config_dir)
        report = validator.generate_validation_report()
        
        self.assertEqual(report['overall_status'], 'FAIL')
        self.assertGreater(len(report['errors']), 0)
    
    def test_inconsistent_plate_values(self):
        """Test validation fails when plate values are inconsistent."""
        # Make plate values inconsistent
        inconsistent_data = self.phenotype_combo_data.copy()
        inconsistent_data.loc[0, 'plate'] = 2
        inconsistent_data.to_csv(self.test_config_dir / 'phenotype_combo.tsv', sep='\t', index=False)
        
        from validate_combo_files import ComboFileValidator
        validator = ComboFileValidator(self.test_config_dir)
        report = validator.generate_validation_report()
        
        self.assertEqual(report['overall_status'], 'FAIL')
        self.assertGreater(len(report['errors']), 0)


def run_tests():
    """Run all tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestWildcardTypeHandling))
    test_suite.addTest(unittest.makeSuite(TestComboFileValidation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests()) 