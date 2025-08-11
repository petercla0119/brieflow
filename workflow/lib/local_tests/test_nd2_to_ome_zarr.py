#!/usr/bin/env python3
"""Comprehensive test script for nd2_to_ome_zarr function

This script allows users to test the nd2_to_ome_zarr function with:
1. Custom ND2 files specified by the user
2. Built-in test images from the small_test_analysis directory
3. Control over whether to keep or clean up test outputs

Usage:
    python test_nd2_to_ome_zarr.py --custom-image /path/to/your/image.nd2 --keep-output
    python test_nd2_to_ome_zarr.py --use-test-image --cleanup
    python test_nd2_to_ome_zarr.py --help
"""

import sys
import os
import argparse
import shutil
from pathlib import Path

# Add the preprocess module to path
sys.path.append('brieflow/workflow/lib/preprocess')

from preprocess import nd2_to_ome_zarr

# Default test image path
DEFAULT_TEST_IMAGE = "./brieflow/tests/small_test_analysis/small_test_data/phenotype/real_images/P001_Pheno_20x_Wells-A1_Points-005__Channel_AF750,Cy3,GFP,DAPI.nd2"

def test_nd2_to_ome_zarr(input_file, output_dir, verbose=True):
    """Test the nd2_to_ome_zarr function with a given input file"""
    print(f"Testing nd2_to_ome_zarr with: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"❌ Test file not found: {input_file}")
        return False
    
    try:
        # Test the function
        success = nd2_to_ome_zarr(
            input_file=input_file,
            output_dir=output_dir,
            chunk_dims=(1, 1, 256, 256),
            verbose=verbose
        )
        
        if success:
            # Check if output was created
            zarr_path = Path(output_dir) / f"{Path(input_file).stem}.zarr"
            metadata_path = Path(output_dir) / f"{Path(input_file).stem}_metadata.json"
            
            print(f"✅ OME-Zarr conversion completed successfully:")
            print(f"   Zarr path exists: {zarr_path.exists()}")
            print(f"   Metadata path exists: {metadata_path.exists()}")
            
            if zarr_path.exists():
                print(f"   Zarr directory contents: {list(zarr_path.iterdir())}")
            
            return True
        else:
            print("❌ Function returned False - conversion failed")
            return False
        
    except Exception as e:
        print(f"❌ Error in nd2_to_ome_zarr: {e}")
        return False

def test_error_handling(output_dir):
    """Test error handling with invalid inputs"""
    print("\n🧪 Testing error handling...")
    
    results = []
    
    # Test with non-existent file
    try:
        nd2_to_ome_zarr(
            input_file="non_existent_file.nd2",
            output_dir=output_dir,
            verbose=False
        )
        print("❌ ERROR: Should have raised FileNotFoundError")
        results.append(False)
    except FileNotFoundError as e:
        print(f"✅ Correctly caught FileNotFoundError: {e}")
        results.append(True)
    except Exception as e:
        print(f"❌ ERROR: Expected FileNotFoundError, got {type(e).__name__}: {e}")
        results.append(False)
    
    # Test with non-ND2 file that exists
    # Create a temporary file with wrong extension for testing
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
        tmp_file.write(b"This is not an ND2 file")
        tmp_file_path = tmp_file.name
    
    try:
        nd2_to_ome_zarr(
            input_file=tmp_file_path,
            output_dir=output_dir,
            verbose=False
        )
        print("❌ ERROR: Should have raised ValueError")
        results.append(False)
    except ValueError as e:
        print(f"✅ Correctly caught ValueError: {e}")
        results.append(True)
    except Exception as e:
        print(f"❌ ERROR: Expected ValueError, got {type(e).__name__}: {e}")
        results.append(False)
    finally:
        # Clean up temporary file
        import os
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)
    
    return all(results)

def cleanup_outputs(output_dirs):
    """Clean up test output directories"""
    print("\n🧹 Cleaning up test files...")
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print(f"   Removed {output_dir}")
        else:
            print(f"   {output_dir} does not exist (already cleaned up)")

def validate_test_image():
    """Check if the default test image exists"""
    if not os.path.exists(DEFAULT_TEST_IMAGE):
        print(f"⚠️  Warning: Default test image not found: {DEFAULT_TEST_IMAGE}")
        print("   This might be because the brieflow test data is not installed.")
        return False
    return True

def main():
    """Main test function with command line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test the nd2_to_ome_zarr function with various options",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with custom image and keep output
  python test_nd2_to_ome_zarr.py --custom-image /path/to/image.nd2 --keep-output
  
  # Test with built-in test image and cleanup
  python test_nd2_to_ome_zarr.py --use-test-image --cleanup
  
  # Test with custom image and cleanup (default)
  python test_nd2_to_ome_zarr.py --custom-image /path/to/image.nd2
  
  # Test with built-in test image and keep output
  python test_nd2_to_ome_zarr.py --use-test-image --keep-output
        """
    )
    
    # Image source options (mutually exclusive)
    image_group = parser.add_mutually_exclusive_group(required=True)
    image_group.add_argument(
        '--custom-image',
        type=str,
        help='Path to custom ND2 file to test'
    )
    image_group.add_argument(
        '--use-test-image',
        action='store_true',
        help='Use the built-in test image from small_test_analysis'
    )
    
    # Output control options
    parser.add_argument(
        '--keep-output',
        action='store_true',
        help='Keep test output files (default: cleanup after testing)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Clean up test output files (default behavior)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./test_output_zarr_improved',
        help='Output directory for test results (default: ./test_output_zarr_improved)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Reduce verbose output during conversion'
    )
    
    args = parser.parse_args()
    
    # Determine image path
    if args.custom_image:
        test_image = args.custom_image
        print(f"🎯 Using custom image: {test_image}")
    else:
        test_image = DEFAULT_TEST_IMAGE
        if not validate_test_image():
            print("❌ Cannot proceed without a valid test image")
            return False
        print(f"🎯 Using built-in test image: {test_image}")
    
    # Determine cleanup behavior
    keep_output = args.keep_output
    if args.cleanup:
        keep_output = False
    
    # Set output directory
    output_dir = args.output_dir
    
    # Set verbosity
    verbose = not args.quiet
    
    print("=" * 80)
    print("🧪 ND2 to OME-Zarr Function Test Suite")
    print("=" * 80)
    print(f"Input file: {test_image}")
    print(f"Output directory: {output_dir}")
    print(f"Keep output: {keep_output}")
    print(f"Verbose: {verbose}")
    print("=" * 80)
    
    # Run tests
    results = []
    
    # Test 1: Main conversion test
    print("\n📋 Test 1: ND2 to OME-Zarr Conversion")
    print("-" * 50)
    result1 = test_nd2_to_ome_zarr(test_image, output_dir, verbose)
    results.append(result1)
    
    # Test 2: Error handling test
    print("\n📋 Test 2: Error Handling")
    print("-" * 50)
    result2 = test_error_handling(output_dir)
    results.append(result2)
    
    # Cleanup if requested
    if not keep_output:
        cleanup_outputs([output_dir])
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 Test Results Summary")
    print("=" * 80)
    print(f"ND2 to OME-Zarr conversion: {'✅ PASS' if results[0] else '❌ FAIL'}")
    print(f"Error handling: {'✅ PASS' if results[1] else '❌ FAIL'}")
    
    all_passed = all(results)
    print(f"\nOverall: {'🎉 ALL TESTS PASSED' if all_passed else '💥 SOME TESTS FAILED'}")
    
    if keep_output:
        print(f"\n📁 Test outputs preserved in: {output_dir}")
    else:
        print(f"\n🧹 Test outputs cleaned up")
    
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 