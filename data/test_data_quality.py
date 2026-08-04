"""
Data quality testing script for conversation JSON files.

Tests that the num_patient_turns field matches the actual count of 
non-empty assistant messages in each conversation file.

Usage:
    # Test a single folder:
    python test_data_quality.py /path/to/folder
    
    # Test all subfolders recursively and generate summary report:
    python test_data_quality.py /path/to/parent_folder --all-subfolders
    
    # Example with nested structure (data/synthetic/patientpsi/hosted_vllm_*/):
    python test_data_quality.py data/synthetic --all-subfolders
"""

import json
import os
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


def test_therapist_messages(file_path: Path) -> Tuple[bool, str]:
    """Test that therapist (user role) messages are not empty.
    The first message can be empty, but more than 1 empty message indicates a problem.
    
    Args:
        file_path: Path to conversation JSON file
        
    Returns:
        Tuple of (passed: bool, message: str)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = data.get('messages', [])
        
        # Find all empty therapist messages
        empty_therapist_msgs = [
            i for i, msg in enumerate(messages)
            if msg.get('role') == 'user' and not msg.get('content', '').strip()
        ]
        
        # Allow only the first message to be empty (index 0)
        if not empty_therapist_msgs:
            return True, "✓ PASS: 0 empty therapist message(s)"
        if empty_therapist_msgs == [0]:
            return True, "✓ PASS: 1 empty therapist message (index 0)"
        indices_str = ', '.join(str(i) for i in empty_therapist_msgs)
        return False, f"✗ FAIL: {len(empty_therapist_msgs)} empty therapist messages at indices: {indices_str}"
            
    except json.JSONDecodeError as e:
        return False, f"✗ JSON decode error: {e}"
    except Exception as e:
        return False, f"✗ Error: {e}"


def test_folder(folder_path: str, print_details: bool = True) -> Dict:
    """Test all JSON files in a folder for data quality.
    
    Args:
        folder_path: Path to folder containing conversation JSON files
        print_details: Whether to print detailed results (default: True)
        
    Returns:
        Dictionary with test results and summary statistics
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        if print_details:
            print(f"Error: Folder '{folder_path}' does not exist")
        return {'error': 'Folder not found'}
    
    # Find all JSON files
    json_files = sorted(folder.glob('*.json'))
    
    if not json_files:
        if print_details:
            print(f"Warning: No JSON files found in '{folder_path}'")
        return {'error': 'No JSON files found', 'total': 0, 'passed': 0, 'failed': 0}
    
    if print_details:
        print(f"\nTesting {len(json_files)} files in: {folder_path}\n")
        print("=" * 80)
    
    results = []
    passed = 0
    failed = 0
    
    for json_file in json_files:
        # Run both tests
        turns_success, turns_message = test_patient_turns_count(json_file)
        therapist_success, therapist_message = test_therapist_messages(json_file)
        
        # File passes only if both tests pass
        file_passed = turns_success and therapist_success
        
        # Combine messages
        combined_message = f"{turns_message} | {therapist_message}"
        
        results.append({
            'file': json_file.name,
            'passed': file_passed,
            'message': combined_message
        })
        
        if file_passed:
            passed += 1
        else:
            failed += 1
            # Print failed files immediately (only if details enabled)
            if print_details:
                print(f"✗ {json_file.name}: {combined_message}")
    
    # Print summary
    if print_details:
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
        'results': results,
        'folder': str(folder)
    }


def test_all_subfolders(parent_folder: str) -> Dict:
    """Test all JSON files in all subfolders of a parent folder (recursive).
    
    Args:
        parent_folder: Path to parent folder containing subfolders with JSON files
        
    Returns:
        Dictionary with aggregated test results for all subfolders
    """
    parent = Path(parent_folder)
    
    if not parent.exists():
        print(f"Error: Parent folder '{parent_folder}' does not exist")
        return {'error': 'Parent folder not found'}
    
    if not parent.is_dir():
        print(f"Error: '{parent_folder}' is not a directory")
        return {'error': 'Not a directory'}
    
    # Recursively find all directories that contain JSON files
    folders_with_json = []
    for dirpath, dirnames, filenames in os.walk(parent):
        # Check if this directory contains any JSON files
        json_files = [f for f in filenames if f.endswith('.json')]
        if json_files:
            folders_with_json.append(Path(dirpath))
    
    if not folders_with_json:
        print(f"Warning: No folders with JSON files found in '{parent_folder}' or its subfolders")
        return {'error': 'No folders with JSON files found'}
    
    print(f"\n{'='*100}")
    print(f"Testing all subfolders in: {parent_folder}")
    print(f"Found {len(folders_with_json)} folder(s) with JSON files")
    print(f"{'='*100}\n")
    
    all_results = {}
    
    for subfolder in sorted(folders_with_json):
        # Get relative path from parent for display
        relative_path = subfolder.relative_to(parent)
        
        print(f"\n{'─'*100}")
        print(f"Processing: {relative_path}")
        print(f"{'─'*100}")
        
        result = test_folder(str(subfolder), print_details=True)
        all_results[str(relative_path)] = result
    
    # Generate final summary report
    print(f"\n\n{'='*100}")
    print(f"FINAL REPORT: All Subfolders Summary")
    print(f"{'='*100}\n")
    
    # Print table header
    print(f"{'Subfolder Path':<60} {'Files':<10} {'Passed':<10} {'Failed':<10} {'Pass %':<10} {'Fail %':<10}")
    print(f"{'-'*60} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    
    total_files = 0
    total_passed = 0
    total_failed = 0
    
    for subfolder_path in sorted(all_results.keys()):
        result = all_results[subfolder_path]
        
        if 'error' in result and result.get('total', 0) == 0:
            # Skip folders with no files
            continue
        
        files = result.get('total', 0)
        passed = result.get('passed', 0)
        failed = result.get('failed', 0)
        
        pass_pct = (passed / files * 100) if files > 0 else 0
        fail_pct = (failed / files * 100) if files > 0 else 0
        
        total_files += files
        total_passed += passed
        total_failed += failed
        
        status = "✓" if failed == 0 and files > 0 else "✗" if failed > 0 else "⚠"
        
        # Truncate path if too long for display
        display_path = subfolder_path if len(subfolder_path) <= 58 else "..." + subfolder_path[-55:]
        print(f"{status} {display_path:<58} {files:<10} {passed:<10} {failed:<10} {pass_pct:<9.1f}% {fail_pct:<9.1f}%")
    
    # Print totals
    print(f"{'-'*60} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    total_pass_pct = (total_passed / total_files * 100) if total_files > 0 else 0
    total_fail_pct = (total_failed / total_files * 100) if total_files > 0 else 0
    print(f"{'TOTAL':<60} {total_files:<10} {total_passed:<10} {total_failed:<10} {total_pass_pct:<9.1f}% {total_fail_pct:<9.1f}%")
    
    print(f"\n{'='*100}")
    
    if total_failed > 0:
        print(f"⚠  Overall: {total_failed} file(s) failed validation across {len([r for r in all_results.values() if r.get('total', 0) > 0])} subfolder(s)")
    else:
        print(f"✓  Overall: All files passed validation!")
    
    print(f"{'='*100}\n")
    
    return {
        'parent_folder': str(parent),
        'subfolders': all_results,
        'summary': {
            'total_files': total_files,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'pass_percentage': total_pass_pct,
            'fail_percentage': total_fail_pct
        }
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
        help='Path to folder containing conversation JSON files (or parent folder if --all-subfolders is used)'
    )
    parser.add_argument(
        '--all-subfolders',
        action='store_true',
        help='Recursively test all subfolders within the specified parent folder and generate a summary report'
    )
    
    args = parser.parse_args()
    
    if args.all_subfolders:
        test_all_subfolders(args.folder)
    else:
        test_folder(args.folder)


if __name__ == "__main__":
    main()
