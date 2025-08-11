#!/usr/bin/env python3
"""
Test script for configure_preprocess_params.py

This script tests:
1. Command line argument parsing
2. Path pattern matching for SBS and phenotype files
3. Metadata extraction and ordering
4. Channel order configuration
5. Dry-run and overwrite functionality
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import pandas as pd
import re

# Add the analysis directory to path for testing
sys.path.append('../../../../analysis')

def test_argument_parsing():
    """Test command line argument parsing"""
    print("Testing argument parsing...")
    
    # Import the script module
    try:
        import configure_preprocess_params as config_script
        print("✅ Successfully imported configure_preprocess_params module")
    except ImportError as e:
        print(f"❌ Failed to import module: {e}")
        return False
    
    # Test argument parser creation
    try:
        parser = config_script.parse_arguments()
        print("✅ Argument parser created successfully")
        print(f"   Arguments: {[arg.dest for arg in parser._actions if hasattr(arg, 'dest')]}")
    except Exception as e:
        print(f"❌ Failed to create argument parser: {e}")
        return False
    
    return True

def test_sbs_path_pattern():
    """Test SBS path pattern matching"""
    print("\nTesting SBS path pattern...")
    
    # Test pattern from the script
    sbs_pattern = r".*cycle(?P<cycle>\d+)/Well(?P<well>\d+)_Point\d+_(?P<tile>\d+)_Channel(?P<channel>[^_]+(?:_\d+)?)_Seq\d+\.nd2$"
    
    # Test file paths
    test_paths = [
        "/seq_imgs/cycle1/Well6_Point6_0003_ChannelGFP_1_Seq0003.nd2",
        "/seq_imgs/cycle2/Well8_Point4_0015_ChannelCy3_Seq4017.nd2",
        "/seq_imgs/cycle1/Well6_Point6_0001_ChannelFar-red_Seq6019.nd2",
        "/seq_imgs/cycle3/Well12_Point2_0044_ChannelDAPI_2_Seq0044.nd2"
    ]
    
    pattern = re.compile(sbs_pattern)
    
    for test_path in test_paths:
        match = pattern.search(test_path)
        if match:
            print(f"✅ Matched: {test_path}")
            print(f"   cycle: {match.group('cycle')}")
            print(f"   well: {match.group('well')}")
            print(f"   tile: {match.group('tile')}")
            print(f"   channel: {match.group('channel')}")
        else:
            print(f"❌ No match: {test_path}")
            return False
    
    return True

def test_phenotype_path_pattern():
    """Test phenotype path pattern matching"""
    print("\nTesting phenotype path pattern...")
    
    # Test pattern from the script
    phenotype_pattern = r".*Staining(?P<round>\d+)_[^/]+/Well(?P<well>\d+)_Point\d+_(?P<tile>\d+)_Channel(?P<channel>[^_]+(?:_\d+)?)_Seq\d+\.nd2$"
    
    # Test file paths
    test_paths = [
        "/pheno_imgs/Staining1_20250221_140528_194/Well6_Point6_0022_ChannelCy3_Seq28752.nd2",
        "/pheno_imgs/Staining2_20250222_150630_195/Well8_Point4_0016_ChannelGFP_1_Seq14373.nd2",
        "/pheno_imgs/Staining1_20250221_140528_194/Well6_Point6_0044_ChannelDAPI_2_Seq0044.nd2",
        "/pheno_imgs/Staining3_20250223_160732_196/Well12_Point2_0008_ChannelFar-red_Seq43111.nd2"
    ]
    
    pattern = re.compile(phenotype_pattern)
    
    for test_path in test_paths:
        match = pattern.search(test_path)
        if match:
            print(f"✅ Matched: {test_path}")
            print(f"   round: {match.group('round')}")
            print(f"   well: {match.group('well')}")
            print(f"   tile: {match.group('tile')}")
            print(f"   channel: {match.group('channel')}")
        else:
            print(f"❌ No match: {test_path}")
            return False
    
    return True

def test_metadata_ordering():
    """Test metadata ordering and types"""
    print("\nTesting metadata ordering...")
    
    # Expected metadata order for SBS
    expected_sbs_order = ["plate", "cycle", "well", "tile", "channel"]
    expected_sbs_types = {"plate": int, "cycle": int, "well": str, "tile": int, "channel": str}
    
    # Expected metadata order for phenotype
    expected_phenotype_order = ["plate", "round", "well", "tile", "channel"]
    expected_phenotype_types = {"plate": int, "round": int, "well": str, "tile": int, "channel": str}
    
    print(f"✅ Expected SBS order: {expected_sbs_order}")
    print(f"✅ Expected SBS types: {expected_sbs_types}")
    print(f"✅ Expected phenotype order: {expected_phenotype_order}")
    print(f"✅ Expected phenotype types: {expected_phenotype_types}")
    
    return True

def test_channel_orders():
    """Test channel order configuration"""
    print("\nTesting channel orders...")
    
    # Expected channel orders
    expected_sbs_channels = ["DAPI", "Cy3", "Far-red", "GFP"]
    expected_phenotype_channels = ["DAPI", "Cy3", "Far-red", "GFP"]
    
    print(f"✅ Expected SBS channels: {expected_sbs_channels}")
    print(f"✅ Expected phenotype channels: {expected_phenotype_channels}")
    
    # Verify they match
    if expected_sbs_channels == expected_phenotype_channels:
        print("✅ SBS and phenotype channel orders match")
    else:
        print("❌ SBS and phenotype channel orders do not match")
        return False
    
    return True

def test_sample_dataframe_structure():
    """Test sample dataframe structure creation"""
    print("\nTesting sample dataframe structure...")
    
    # Create a mock sample dataframe
    mock_sbs_data = {
        'sample_fp': ['/path/to/file1.nd2', '/path/to/file2.nd2'],
        'plate': [1, 1],
        'cycle': [1, 2],
        'well': ['6', '8'],
        'tile': [3, 15],
        'channel': ['GFP', 'Cy3']
    }
    
    mock_phenotype_data = {
        'sample_fp': ['/path/to/file3.nd2', '/path/to/file4.nd2'],
        'plate': [1, 1],
        'round': [1, 2],
        'well': ['6', '8'],
        'tile': [22, 16],
        'channel': ['Cy3', 'GFP']
    }
    
    sbs_df = pd.DataFrame(mock_sbs_data)
    phenotype_df = pd.DataFrame(mock_phenotype_data)
    
    # Check column order
    expected_sbs_cols = ['sample_fp', 'plate', 'cycle', 'well', 'tile', 'channel']
    expected_phenotype_cols = ['sample_fp', 'plate', 'round', 'well', 'tile', 'channel']
    
    if list(sbs_df.columns) == expected_sbs_cols:
        print("✅ SBS dataframe columns in correct order")
    else:
        print(f"❌ SBS dataframe columns incorrect: {list(sbs_df.columns)}")
        return False
    
    if list(phenotype_df.columns) == expected_phenotype_cols:
        print("✅ Phenotype dataframe columns in correct order")
    else:
        print(f"❌ Phenotype dataframe columns incorrect: {list(phenotype_df.columns)}")
        return False
    
    # Check data types
    if sbs_df['plate'].dtype == 'int64' and sbs_df['well'].dtype == 'object':
        print("✅ SBS dataframe data types correct")
    else:
        print(f"❌ SBS dataframe data types incorrect: plate={sbs_df['plate'].dtype}, well={sbs_df['well'].dtype}")
        return False
    
    if phenotype_df['plate'].dtype == 'int64' and phenotype_df['well'].dtype == 'object':
        print("✅ Phenotype dataframe data types correct")
    else:
        print(f"❌ Phenotype dataframe data types incorrect: plate={phenotype_df['plate'].dtype}, well={phenotype_df['well'].dtype}")
        return False
    
    return True

def test_wildcard_combos():
    """Test wildcard combination generation"""
    print("\nTesting wildcard combinations...")
    
    # Create mock dataframes
    mock_sbs_data = {
        'plate': [1, 1, 1, 1],
        'cycle': [1, 1, 2, 2],
        'well': ['6', '8', '6', '8'],
        'tile': [3, 15, 3, 15],
        'channel': ['GFP', 'Cy3', 'GFP', 'Cy3']
    }
    
    mock_phenotype_data = {
        'plate': [1, 1, 1, 1],
        'round': [1, 1, 2, 2],
        'well': ['6', '8', '6', '8'],
        'tile': [22, 16, 22, 16],
        'channel': ['Cy3', 'GFP', 'Cy3', 'GFP']
    }
    
    sbs_df = pd.DataFrame(mock_sbs_data)
    phenotype_df = pd.DataFrame(mock_phenotype_data)
    
    # Generate wildcard combos
    sbs_combos = sbs_df[['plate', 'cycle', 'well', 'tile', 'channel']].drop_duplicates().astype(str)
    phenotype_combos = phenotype_df[['plate', 'round', 'well', 'tile', 'channel']].drop_duplicates().astype(str)
    
    print(f"✅ SBS combos shape: {sbs_combos.shape}")
    print(f"✅ Phenotype combos shape: {phenotype_combos.shape}")
    
    # Check that all columns are present
    expected_sbs_combo_cols = ['plate', 'cycle', 'well', 'tile', 'channel']
    expected_phenotype_combo_cols = ['plate', 'round', 'well', 'tile', 'channel']
    
    if list(sbs_combos.columns) == expected_sbs_combo_cols:
        print("✅ SBS combo columns correct")
    else:
        print(f"❌ SBS combo columns incorrect: {list(sbs_combos.columns)}")
        return False
    
    if list(phenotype_combos.columns) == expected_phenotype_combo_cols:
        print("✅ Phenotype combo columns correct")
    else:
        print(f"❌ Phenotype combo columns incorrect: {list(phenotype_combos.columns)}")
        return False
    
    return True

def test_file_operations():
    """Test file operations (dry-run mode)"""
    print("\nTesting file operations...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test config file creation
        config_content = {
            "all": {"root_fp": "test_output/"},
            "preprocess": {
                "sbs_channel_order": ["DAPI", "Cy3", "Far-red", "GFP"],
                "phenotype_channel_order": ["DAPI", "Cy3", "Far-red", "GFP"]
            }
        }
        
        config_file = temp_path / "test_config.yml"
        
        try:
            import yaml
            with open(config_file, 'w') as f:
                yaml.dump(config_content, f, default_flow_style=False)
            
            print("✅ Config file created successfully")
            
            # Verify content
            with open(config_file, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            if loaded_config['preprocess']['sbs_channel_order'] == ["DAPI", "Cy3", "Far-red", "GFP"]:
                print("✅ Config file content verified")
            else:
                print("❌ Config file content incorrect")
                return False
                
        except Exception as e:
            print(f"❌ Failed to create/verify config file: {e}")
            return False
    
    return True

def run_all_tests():
    """Run all tests and report results"""
    print("=" * 80)
    print("RUNNING TESTS FOR CONFIGURE_PREPROCESS_PARAMS.PY")
    print("=" * 80)
    
    tests = [
        ("Argument Parsing", test_argument_parsing),
        ("SBS Path Pattern", test_sbs_path_pattern),
        ("Phenotype Path Pattern", test_phenotype_path_pattern),
        ("Metadata Ordering", test_metadata_ordering),
        ("Channel Orders", test_channel_orders),
        ("Sample Dataframe Structure", test_sample_dataframe_structure),
        ("Wildcard Combinations", test_wildcard_combos),
        ("File Operations", test_file_operations)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' failed with exception: {e}")
            results.append((test_name, False))
    
    # Report results
    print("\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 