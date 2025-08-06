#!/usr/bin/env python3
"""Test script for preprocess.py functions"""

import sys
import os
from pathlib import Path

# Add the preprocess module to path
sys.path.append('analysis/seq_processing/Brieflow/workflow/lib/preprocess')

from preprocess import extract_tile_metadata, nd2_to_tiff, nd2_to_ome_zarr

def test_extract_tile_metadata():
    """Test extract_tile_metadata function"""
    print("Testing extract_tile_metadata...")
    
    # Use a real ND2 file from the test data
    test_file = "./brieflow/tests/small_test_analysis/small_test_data/phenotype/real_images/P001_Pheno_20x_Wells-A1_Points-005__Channel_AF750,Cy3,GFP,DAPI.nd2"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    try:
        # Test the function
        metadata = extract_tile_metadata(
            tile_fp=test_file,
            plate=1,
            well="A1",
            tile=5,
            cycle=1,
            verbose=True
        )
        
        print(f"Metadata extracted successfully:")
        print(f"Shape: {metadata.shape}")
        print(f"Columns: {list(metadata.columns)}")
        print(f"First row: {metadata.iloc[0].to_dict()}")
        
        return True
        
    except Exception as e:
        print(f"Error in extract_tile_metadata: {e}")
        return False

def test_nd2_to_tiff():
    """Test nd2_to_tiff function"""
    print("\nTesting nd2_to_tiff...")
    
    # Use a real ND2 file from the test data
    test_file = "./brieflow/tests/small_test_analysis/small_test_data/phenotype/real_images/P001_Pheno_20x_Wells-A1_Points-005__Channel_AF750,Cy3,GFP,DAPI.nd2"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    try:
        # Test the function
        image_array = nd2_to_tiff(
            files=test_file,
            channel_order_flip=False,
            verbose=True
        )
        
        print(f"Image converted successfully:")
        print(f"Array shape: {image_array.shape}")
        print(f"Data type: {image_array.dtype}")
        print(f"Min value: {image_array.min()}")
        print(f"Max value: {image_array.max()}")
        
        return True
        
    except Exception as e:
        print(f"Error in nd2_to_tiff: {e}")
        return False

def test_nd2_to_ome_zarr():
    """Test nd2_to_ome_zarr function"""
    print("\nTesting nd2_to_ome_zarr...")
    
    # Use a real ND2 file from the test data
    test_file = "./brieflow/tests/small_test_analysis/small_test_data/phenotype/real_images/P001_Pheno_20x_Wells-A1_Points-005__Channel_AF750,Cy3,GFP,DAPI.nd2"
    output_dir = "./test_output_zarr"
    
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}")
        return False
    
    try:
        # Test the function
        nd2_to_ome_zarr(
            input_file=test_file,
            output_dir=output_dir,
            chunk_dims=(1, 1, 256, 256),
            verbose=True
        )
        
        # Check if output was created
        zarr_path = Path(output_dir) / f"{Path(test_file).stem}.zarr"
        metadata_path = Path(output_dir) / f"{Path(test_file).stem}_metadata.json"
        
        print(f"OME-Zarr conversion completed:")
        print(f"Zarr path exists: {zarr_path.exists()}")
        print(f"Metadata path exists: {metadata_path.exists()}")
        
        if zarr_path.exists():
            print(f"Zarr directory contents: {list(zarr_path.iterdir())}")
        
        return True
        
    except Exception as e:
        print(f"Error in nd2_to_ome_zarr: {e}")
        return False

def cleanup():
    """Clean up test files"""
    print("\nCleaning up test files...")
    
    test_output_dir = "./test_output_zarr"
    if os.path.exists(test_output_dir):
        import shutil
        shutil.rmtree(test_output_dir)
        print(f"Removed {test_output_dir}")

def main():
    """Run all tests"""
    print("Starting preprocess.py function tests...")
    print("=" * 50)
    
    results = []
    
    # Test each function
    results.append(test_extract_tile_metadata())
    results.append(test_nd2_to_tiff())
    results.append(test_nd2_to_ome_zarr())
    
    # Clean up
    cleanup()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary:")
    print(f"extract_tile_metadata: {'PASS' if results[0] else 'FAIL'}")
    print(f"nd2_to_tiff: {'PASS' if results[1] else 'FAIL'}")
    print(f"nd2_to_ome_zarr: {'PASS' if results[2] else 'FAIL'}")
    
    all_passed = all(results)
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    
    return all_passed

if __name__ == "__main__":
    main() 