#!/usr/bin/env python3
"""Test script for improved nd2_to_ome_zarr function"""

import sys
import os
from pathlib import Path

# Add the preprocess module to path
sys.path.append('analysis/seq_processing/Brieflow/workflow/lib/preprocess')

from preprocess import nd2_to_ome_zarr

def test_improved_nd2_to_ome_zarr():
    """Test the improved nd2_to_ome_zarr function"""
    print("Testing improved nd2_to_ome_zarr...")
    
    # Use a real ND2 file from the test data
    test_file = "./brieflow/tests/small_test_analysis/small_test_data/phenotype/real_images/P001_Pheno_20x_Wells-A1_Points-005__Channel_AF750,Cy3,GFP,DAPI.nd2"
    output_dir = "./test_output_zarr_improved"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    try:
        # Test the improved function
        success = nd2_to_ome_zarr(
            input_file=test_file,
            output_dir=output_dir,
            chunk_dims=(1, 1, 256, 256),
            verbose=True
        )
        
        if success:
            # Check if output was created
            zarr_path = Path(output_dir) / f"{Path(test_file).stem}.zarr"
            metadata_path = Path(output_dir) / f"{Path(test_file).stem}_metadata.json"
            
            print(f"OME-Zarr conversion completed successfully:")
            print(f"Zarr path exists: {zarr_path.exists()}")
            print(f"Metadata path exists: {metadata_path.exists()}")
            
            if zarr_path.exists():
                print(f"Zarr directory contents: {list(zarr_path.iterdir())}")
            
            return True
        else:
            print("Function returned False - conversion failed")
            return False
        
    except Exception as e:
        print(f"Error in improved nd2_to_ome_zarr: {e}")
        return False

def test_error_handling():
    """Test error handling with invalid inputs"""
    print("\nTesting error handling...")
    
    # Test with non-existent file
    try:
        nd2_to_ome_zarr(
            input_file="non_existent_file.nd2",
            output_dir="./test_error_output",
            verbose=True
        )
        print("ERROR: Should have raised FileNotFoundError")
        return False
    except FileNotFoundError as e:
        print(f"✓ Correctly caught FileNotFoundError: {e}")
    except Exception as e:
        print(f"ERROR: Expected FileNotFoundError, got {type(e).__name__}: {e}")
        return False
    
    # Test with non-ND2 file
    try:
        nd2_to_ome_zarr(
            input_file="test_improved_ome_zarr.py",  # This script file
            output_dir="./test_error_output",
            verbose=True
        )
        print("ERROR: Should have raised ValueError")
        return False
    except ValueError as e:
        print(f"✓ Correctly caught ValueError: {e}")
    except Exception as e:
        print(f"ERROR: Expected ValueError, got {type(e).__name__}: {e}")
        return False
    
    return True

def cleanup():
    """Clean up test files"""
    print("\nCleaning up test files...")
    
    test_output_dirs = ["./test_output_zarr_improved", "./test_error_output"]
    for output_dir in test_output_dirs:
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
            print(f"Removed {output_dir}")

def main():
    """Run all tests"""
    print("Starting improved nd2_to_ome_zarr function tests...")
    print("=" * 60)
    
    results = []
    
    # Test the improved function
    results.append(test_improved_nd2_to_ome_zarr())
    
    # Test error handling
    results.append(test_error_handling())
    
    # Clean up
    cleanup()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary:")
    print(f"Improved nd2_to_ome_zarr: {'PASS' if results[0] else 'FAIL'}")
    print(f"Error handling: {'PASS' if results[1] else 'FAIL'}")
    
    all_passed = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main() 