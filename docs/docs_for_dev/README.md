# Local Tests for Brieflow Functions

This directory contains test scripts for testing functions defined in the `lib` directory.

## Test Scripts

### `test_nd2_to_ome_zarr.py` - Comprehensive ND2 to OME-Zarr Testing

A comprehensive test script that allows users to test the `nd2_to_ome_zarr` function with various options.

#### Features

- **Custom Image Testing**: Test with your own ND2 files
- **Built-in Test Image**: Use the default test image from small_test_analysis
- **Output Control**: Choose to keep or clean up test outputs
- **Error Handling Tests**: Validates proper error handling
- **Verbose Output Control**: Reduce output noise if needed

🖼️ Flexible Image Source:
--custom-image PATH: Use your own ND2 files
--use-test-image: Use built-in test image from small_test_analysis
�� Output Control:
--keep-output: Keep test files for inspection
--cleanup: Remove test files after testing (default)
--output-dir PATH: Custom output directory
�� Additional Options:
--quiet: Reduce verbose output
--help: Comprehensive help with examples


#### Usage Examples

```bash
# Test with custom image and keep output
python test_nd2_to_ome_zarr.py --custom-image /path/to/your/image.nd2 --keep-output

# Test with built-in test image and cleanup (no files left behind)
python test_nd2_to_ome_zarr.py --use-test-image --cleanup

# Test with custom image and cleanup (default behavior)
python test_nd2_to_ome_zarr.py --custom-image /path/to/your/image.nd2

# Test with built-in test image and keep output
python test_nd2_to_ome_zarr.py --use-test-image --keep-output

# Test with custom output directory
python test_nd2_to_ome_zarr.py --custom-image /path/to/image.nd2 --output-dir ./my_test_output

# Test with reduced verbose output
python test_nd2_to_ome_zarr.py --custom-image /path/to/image.nd2 --quiet
```

#### Command Line Options

- `--custom-image PATH`: Path to custom ND2 file to test
- `--use-test-image`: Use the built-in test image from small_test_analysis
- `--keep-output`: Keep test output files (default: cleanup after testing)
- `--cleanup`: Clean up test output files (default behavior)
- `--output-dir PATH`: Output directory for test results (default: ./test_output_zarr_improved)
- `--quiet`: Reduce verbose output during conversion
- `--help`: Show help message and examples

#### Test Results

The script provides comprehensive test results including:

- ✅ **ND2 to OME-Zarr conversion test**: Validates the core function
- ✅ **Error handling test**: Ensures proper error handling for invalid inputs
- 📊 **Summary report**: Overall pass/fail status
- 📁 **Output verification**: Confirms files were created correctly
- 🧹 **Cleanup confirmation**: Shows what was cleaned up

#### Error Handling Test Details

The error handling test validates two critical scenarios:

1. **Non-existent file test**: Tests that `FileNotFoundError` is raised when the input file doesn't exist
2. **Invalid file type test**: Tests that `ValueError` is raised when the input file exists but has the wrong extension (not `.nd2`)

**Technical Note**: The `nd2_to_ome_zarr` function checks for file existence before checking file type. The test creates a temporary file with a `.txt` extension to properly test the file type validation, ensuring both error conditions are correctly handled.

#### Example Output

```
================================================================================
🧪 ND2 to OME-Zarr Function Test Suite
================================================================================
Input file: /path/to/your/image.nd2
Output directory: ./test_output_zarr_improved
Keep output: False
Verbose: True
================================================================================

📋 Test 1: ND2 to OME-Zarr Conversion
--------------------------------------------------
✅ OME-Zarr conversion completed successfully:
   Zarr path exists: True
   Metadata path exists: True

📋 Test 2: Error Handling
--------------------------------------------------
✅ Correctly caught FileNotFoundError: Input file not found: non_existent_file.nd2
✅ Correctly caught ValueError: Invalid file type

================================================================================
📊 Test Results Summary
================================================================================
ND2 to OME-Zarr conversion: ✅ PASS
Error handling: ✅ PASS

Overall: 🎉 ALL TESTS PASSED

🧹 Test outputs cleaned up
================================================================================
```

## Legacy Test Scripts

- `test_improved_ome_zarr.py`: Original test script (superseded by `test_nd2_to_ome_zarr.py`)
- `test_preprocess_functions.py`: Tests for other preprocess functions

## Recent Updates

### 2025-08-05: Error Handling Test Fix
- **Issue**: Error handling test was failing because it used an existing file with wrong extension, but the function checks file existence before file type
- **Fix**: Modified test to create a temporary `.txt` file to properly test the `ValueError` condition
- **Result**: All tests now pass consistently
- **Technical Details**: The `nd2_to_ome_zarr` function validates input in this order:
  1. Check if file exists → `FileNotFoundError` if not
  2. Check if file has `.nd2` extension → `ValueError` if not

## Notes

- All test scripts require the `brieflow_main_env` conda environment to be activated
- Test scripts are designed to be run from the `brieflow-analysis` root directory
- The `--cleanup` option is the default behavior to avoid leaving test files behind
- Use `--keep-output` if you want to inspect the generated OME-Zarr files 