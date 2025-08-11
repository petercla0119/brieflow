#!/usr/bin/env python3
"""
Simple test script to check wildcard types and input values.
"""

import sys
from pathlib import Path

# Add the brieflow workflow lib directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from preprocess.file_utils import get_sample_fps

def test_wildcard_types():
    """Test with different wildcard types to see what's causing the issue."""
    
    # Load the samples data
    sbs_df = pd.read_csv("config/sbs_samples.tsv", sep="\t")
    
    print("=== Testing Wildcard Type Handling ===")
    
    # Test with string wildcards (as Snakemake might pass them)
    print("\nTest with string wildcards:")
    print("plate='1', well='6', tile='0', cycle='1'")
    try:
        result = get_sample_fps(sbs_df, plate='1', well='6', tile='0', cycle='1')
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with integer wildcards
    print("\nTest with integer wildcards:")
    print("plate=1, well=6, tile=0, cycle=1")
    try:
        result = get_sample_fps(sbs_df, plate=1, well=6, tile=0, cycle=1)
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test with mixed types
    print("\nTest with mixed types:")
    print("plate='1', well=6, tile='0', cycle=1")
    try:
        result = get_sample_fps(sbs_df, plate='1', well=6, tile='0', cycle=1)
        print(f"Result: {result}")
        print(f"Type: {type(result)}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Check DataFrame filtering behavior
    print("\n=== DataFrame Filtering Test ===")
    
    # Test string vs integer filtering
    print("\nFiltering with string '1' vs integer 1:")
    print(f"plate == '1': {len(sbs_df[sbs_df['plate'] == '1'])} rows")
    print(f"plate == 1: {len(sbs_df[sbs_df['plate'] == 1])} rows")
    
    print(f"well == '6': {len(sbs_df[sbs_df['well'] == '6'])} rows")
    print(f"well == 6: {len(sbs_df[sbs_df['well'] == 6])} rows")
    
    print(f"tile == '0': {len(sbs_df[sbs_df['tile'] == '0'])} rows")
    print(f"tile == 0: {len(sbs_df[sbs_df['tile'] == 0])} rows")
    
    print(f"cycle == '1': {len(sbs_df[sbs_df['cycle'] == '1'])} rows")
    print(f"cycle == 1: {len(sbs_df[sbs_df['cycle'] == 1])} rows")

if __name__ == "__main__":
    test_wildcard_types() 