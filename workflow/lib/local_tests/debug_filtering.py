#!/usr/bin/env python3
"""
Debug script to test the get_sample_fps function and identify why filtering is failing.
"""

import sys
from pathlib import Path

# Add the brieflow workflow lib directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from preprocess.file_utils import get_sample_fps

def test_filtering():
    """Test the filtering function with different parameters."""
    
    # Load the samples data
    sbs_df = pd.read_csv("config/sbs_samples.tsv", sep="\t")
    phenotype_df = pd.read_csv("config/phenotype_samples.tsv", sep="\t")
    
    print("=== SBS Samples DataFrame ===")
    print(f"Shape: {sbs_df.shape}")
    print(f"Columns: {list(sbs_df.columns)}")
    print(f"Data types: {sbs_df.dtypes.to_dict()}")
    print(f"First few rows:")
    print(sbs_df.head())
    
    print("\n=== Phenotype Samples DataFrame ===")
    print(f"Shape: {phenotype_df.shape}")
    print(f"Columns: {list(phenotype_df.columns)}")
    print(f"Data types: {phenotype_df.dtypes.to_dict()}")
    print(f"First few rows:")
    print(phenotype_df.head())
    
    # Test filtering with specific wildcard values
    print("\n=== Testing SBS Filtering ===")
    
    # Test case 1: plate=1, well=6, tile=0, cycle=1
    print("\nTest 1: plate=1, well=6, tile=0, cycle=1")
    try:
        result = get_sample_fps(sbs_df, plate=1, well=6, tile=0, cycle=1)
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        if isinstance(result, list):
            print(f"Length: {len(result)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 2: plate=1, well=6, tile=0, cycle=1, channel_order specified
    print("\nTest 2: plate=1, well=6, tile=0, cycle=1, channel_order=['DAPI', 'GFP']")
    try:
        result = get_sample_fps(sbs_df, plate=1, well=6, tile=0, cycle=1, channel_order=['DAPI', 'GFP'])
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        if isinstance(result, list):
            print(f"Length: {len(result)}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n=== Testing Phenotype Filtering ===")
    
    # Test case 3: plate=1, well=6, tile=0
    print("\nTest 3: plate=1, well=6, tile=0")
    try:
        result = get_sample_fps(phenotype_df, plate=1, well=6, tile=0)
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        if isinstance(result, list):
            print(f"Length: {len(result)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test case 4: plate=1, well=6, tile=0, round_order=[1, 2], channel_order specified
    print("\nTest 4: plate=1, well=6, tile=0, round_order=[1, 2], channel_order=['DAPI', 'Cy3']")
    try:
        result = get_sample_fps(phenotype_df, plate=1, well=6, tile=0, round_order=[1, 2], channel_order=['DAPI', 'Cy3'])
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
        if isinstance(result, list):
            print(f"Length: {len(result)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_filtering() 