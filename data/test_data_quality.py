"""
Data quality testing script for conversation JSON files.

Tests that the num_patient_turns field matches the actual count of 
non-empty assistant messages in each conversation file.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


def test_patient_turns_count(file_path: Path) -> Tuple[bool, str]:
    """Test if num_patient_turns matches actual non-empty assistant messages.
    
    Args:
        file_path: Path to conversation JSON file
        
    Returns:
        Tuple of (passed: bool, message: str)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Get expected count from metadata
        expected_count = data.get('num_patient_turns')
        if expected_count is None:
            return False, f"Missing 'num_patient_turns' field"
        
        # Count actual non-empty assistant messages
        messages = data.get('messages', [])
        actual_count = sum(
            1 for msg in messages 
            if msg.get('role') == 'assistant' and msg.get('content', '').strip()
        )
        
        # Compare
        if actual_count == expected_count:
            return True, f"✓ PASS: {actual_count} patient turns"
        else:
            return False, f"✗ FAIL: Expected {expected_count} patient turns, found {actual_count}"
            
    except json.JSONDecodeError as e:
        return False, f"✗ JSON decode error: {e}"
    except Exception as e:
        return False, f"✗ Error: {e}"


def test_folder(folder_path: str) -> Dict:
    """Test all JSON files in a folder for data quality.
    
    Args:
        folder_path: Path to folder containing conversation JSON files
        
    Returns:
        Dictionary with test results and summary statistics
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return {'error': 'Folder not found'}
    
    # Find all JSON files
    json_files = sorted(folder.glob('*.json'))
    
    if not json_files:
        print(f"Warning: No JSON files found in '{folder_path}'")
        return {'error': 'No JSON files found'}
    
    print(f"\nTesting {len(json_files)} files in: {folder_path}\n")
    print("=" * 80)
    
    results = []
    passed = 0
    failed = 0
    
    for json_file in json_files:
        success, message = test_patient_turns_count(json_file)
        
        # Print result
        # status = "✓" if success else "✗"
        # print(f"{status} {json_file.name}: {message}")
        
        results.append({
            'file': json_file.name,
            'passed': success,
            'message': message
        })
        
        if success:
            passed += 1
        else:
            failed += 1
    
    # Print summary
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Total files: {len(json_files)}")
    print(f"  Passed: {passed} ({passed/len(json_files)*100:.1f}%)")
    print(f"  Failed: {failed} ({failed/len(json_files)*100:.1f}%)")
    
    if failed > 0:
        print(f"\n⚠ {failed} file(s) failed validation")
    else:
        print(f"\n✓ All files passed validation!")
    
    return {
        'total': len(json_files),
        'passed': passed,
        'failed': failed,
        'results': results
    }


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test data quality of conversation JSON files'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing conversation JSON files'
    )
    
    args = parser.parse_args()
    
    test_folder(args.folder)


if __name__ == "__main__":
    main()
