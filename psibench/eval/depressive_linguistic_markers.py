"""
Analyze and compare depressive linguistic markers between real and synthetic conversations.

This script compares the frequency of depressive linguistic markers in patient responses
across real depressive patients and different LLM-based patient simulators.

Frequencies are calculated as:
- Raw count: Total occurrences of each marker
- Normalized by words: Count per 100 words
- Normalized by utterances: Average count per patient turn
- Normalized by unique speakers: Average count per conversation

#!/bin/bash
# Example script to run depressive linguistic marker analysis
# Automatically loads all real data (ESC, HOPE, AnnoMI) and all synthetic PSI-backend pairs

# Basic usage - load all data with all metrics
python psibench/eval/depressive_linguistic_markers.py \
    --output-dir output/depressive_markers \
    --metrics all \
    --save-csv

# Load with specific metric only
python psibench/eval/depressive_linguistic_markers.py \
    --output-dir output/depressive_markers/per_100_words \
    --metrics per_100_words \
    --save-csv

# Analyze only first 10 turns for all pairs
python psibench/eval/depressive_linguistic_markers.py \
    --max-turns 10 \
    --output-dir output/depressive_markers/first_10_turns \
    --metrics per_100_words \
    --save-csv

"""

import argparse
import json
import re
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from tabulate import tabulate

from psibench.data_loader.main_loader import load_eeyore_dataset, load_synthetic_hf_to_df
from psibench.eval.utils import extract_patient_messages_by_turn, get_all_psi_backend_pairs, safe_dir_name


# =============================================================================
# DEPRESSIVE LINGUISTIC MARKER TAXONOMY
# =============================================================================
# Based on validated taxonomies from depression research literature

DEPRESSIVE_MARKERS = {
    "absolutist": {
        "patterns": [
            r"\b(absolutely|all|always|complete|completely|constant|constantly)\b",
            r"\b(definitely|entire|ever|every|everyone|everything|full)\b",
            r"\b(must|never|nothing|totally|whole)\b"
        ],
        "case_sensitive": False
    },
    "depressive_words": {
        "patterns": [
            r"\b(depression|collapse|stress|suicide|apastia|anxious|sad|tired)\b",
            r"\b(death|lonely|insomnia|bad|desperate|give up|low|leave)\b",
            r"\b(fear|danger|close|sensitive|lost|shadow|destroy|suspect)\b",
            r"\b(crash|dark|helpless|guilt|negative|frustration|nervous)\b",
            r"\b(melancholy|rubbish|jump|forget|goodbye|cut wrist|edge|haze)\b",
            r"\b(antidepressant)\b"
        ],
        "case_sensitive": False
    },
    "non_fluencies": {
        "patterns": [
            r"\b(uh|um|er|ah|eh|oh|hmm|mm|mmm)\b",
            r"\b(you know|y'know|i mean|let's see)\b"
        ],
        "case_sensitive": False
    },
    "first_person_singular": {
        "patterns": [
            r"\bI\b", r"\bme\b", r"\bmine\b", r"\bmy\b", r"\bmyself\b"
        ],
        "case_sensitive": False
    },
    "social_pronouns": {
        "patterns": [
            r"\bwe\b", r"\bus\b", r"\bour\b", r"\bours\b", r"\bourselves\b"
        ],
        "case_sensitive": False
    }
}


# =============================================================================
# TEXT PROCESSING & MARKER DETECTION
# =============================================================================

def count_words(text: str) -> int:
    """Count total words in text."""
    return len(text.split())


def count_tokens(text: str) -> int:
    """Approximate token count (roughly 1 token = 4 characters)."""
    return len(text) // 4


def compile_marker_patterns(markers_dict: Dict) -> Dict[str, List[re.Pattern]]:
    """Compile regex patterns for each marker category."""
    compiled = {}
    for marker_name, marker_info in markers_dict.items():
        flags = 0 if marker_info.get("case_sensitive", False) else re.IGNORECASE
        compiled[marker_name] = [
            re.compile(pattern, flags) 
            for pattern in marker_info["patterns"]
        ]
    return compiled


def detect_markers(text: str, compiled_patterns: Dict[str, List[re.Pattern]]) -> Dict[str, int]:
    """
    Detect all markers in text and return counts for each category.
    
    Args:
        text: Input text to analyze
        compiled_patterns: Dictionary of compiled regex patterns
        
    Returns:
        Dictionary mapping marker name to count
    """
    counts = {}
    for marker_name, patterns in compiled_patterns.items():
        total_count = 0
        for pattern in patterns:
            matches = pattern.findall(text)
            total_count += len(matches)
        counts[marker_name] = total_count
    return counts


def extract_patient_messages(messages: List[Dict]) -> List[str]:
    """Extract content of messages from the 'assistant' role (patient)."""
    return [
        msg.get('content', '').strip() 
        for msg in messages 
        if msg.get('role') == 'assistant' and msg.get('content', '').strip()
    ]


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def analyze_conversations(
    conversations: List[Dict],
    compiled_patterns: Dict[str, List[re.Pattern]],
    max_turns: Optional[int] = None
) -> Dict:
    """
    Analyze all conversations and compute marker statistics.
    
    Args:
        conversations: List of conversation dictionaries
        compiled_patterns: Compiled regex patterns for markers
        max_turns: Optional maximum number of turns to analyze
        
    Returns:
        Dictionary containing:
        - raw_counts: Total count of each marker across all conversations
        - word_counts: Total words analyzed
        - utterance_counts: Total patient utterances
        - conversation_count: Total conversations
        - per_conversation_counts: List of marker counts per conversation
        - marker_details: Detailed breakdown per conversation
    """
    total_raw_counts = Counter()
    total_words = 0
    total_tokens = 0
    total_utterances = 0
    total_messages_with_markers = Counter()
    conversation_count = len(conversations)
    per_conversation_counts = []
    marker_details = []
    
    for conv_idx, conv in enumerate(conversations):
        patient_messages = extract_patient_messages(conv.get('messages', []))
        
        # Calculate actual patient turns and apply limit
        actual_patient_turns = len(patient_messages)
        if max_turns is not None:
            effective_max_turns = min(max_turns, actual_patient_turns)
            patient_messages = patient_messages[:effective_max_turns]
        
        if not patient_messages:
            continue
        
        # Aggregate all patient messages for this conversation
        full_text = " ".join(patient_messages)
        
        # Count markers in full conversation
        conv_marker_counts = detect_markers(full_text, compiled_patterns)
        
        # Count messages containing each marker type
        conv_messages_with_markers = Counter()
        for message in patient_messages:
            message_markers = detect_markers(message, compiled_patterns)
            for marker_name, count in message_markers.items():
                if count > 0:
                    conv_messages_with_markers[marker_name] += 1
        
        # Count words, tokens, and utterances
        conv_words = count_words(full_text)
        conv_tokens = count_tokens(full_text)
        conv_utterances = len(patient_messages)
        
        # Update totals
        total_raw_counts.update(conv_marker_counts)
        total_words += conv_words
        total_tokens += conv_tokens
        total_utterances += conv_utterances
        total_messages_with_markers.update(conv_messages_with_markers)
        
        # Store per-conversation data
        per_conversation_counts.append(conv_marker_counts)
        marker_details.append({
            'conversation_id': conv_idx,
            'words': conv_words,
            'utterances': conv_utterances,
            'markers': conv_marker_counts
        })
    
    return {
        'raw_counts': dict(total_raw_counts),
        'word_counts': total_words,
        'token_counts': total_tokens,
        'utterance_counts': total_utterances,
        'messages_with_markers': dict(total_messages_with_markers),
        'conversation_count': conversation_count,
        'per_conversation_counts': per_conversation_counts,
        'marker_details': marker_details
    }


def compute_normalized_metrics(analysis_results: Dict) -> Dict[str, Dict[str, float]]:
    """
    Compute normalized metrics from analysis results.
    
    Returns:
        Dictionary with keys:
        - 'per_100_tokens': Markers per 100 tokens
        - 'percentage_messages': Percentage of patient messages containing each marker
    """
    raw_counts = analysis_results['raw_counts']
    token_count = analysis_results['token_counts']
    utterance_count = analysis_results['utterance_counts']
    messages_with_markers = analysis_results['messages_with_markers']
    
    metrics = {}
    
    # Per 100 tokens (exclude first_person_singular and social_pronouns from display)
    if token_count > 0:
        metrics['per_100_tokens'] = {
            marker: (count / token_count) * 100
            for marker, count in raw_counts.items()
            if marker not in ['first_person_singular', 'social_pronouns']
        }
    else:
        metrics['per_100_tokens'] = {
            marker: 0.0 for marker in raw_counts 
            if marker not in ['first_person_singular', 'social_pronouns']
        }
    
    # Calculate self-focus ratio (first_person_singular / social_pronouns)
    first_person = raw_counts.get('first_person_singular', 0)
    social = raw_counts.get('social_pronouns', 0)
    
    if social > 0:
        self_focus_ratio = first_person / social
    elif first_person > 0:
        self_focus_ratio = float('inf')  # Only self-focus, no social
    else:
        self_focus_ratio = 0.0  # Neither present
    
    metrics['per_100_tokens']['self_focus_ratio'] = self_focus_ratio
    
    # Percentage of messages containing each marker (exclude first_person_singular and social_pronouns)
    if utterance_count > 0:
        metrics['percentage_messages'] = {
            marker: (messages_with_markers.get(marker, 0) / utterance_count) * 100
            for marker in raw_counts.keys()
            if marker not in ['first_person_singular', 'social_pronouns']
        }
    else:
        metrics['percentage_messages'] = {
            marker: 0.0 for marker in raw_counts
            if marker not in ['first_person_singular', 'social_pronouns']
        }
    
    # Add self-focus ratio to percentage_messages as well for consistency
    metrics['percentage_messages']['self_focus_ratio'] = self_focus_ratio
    
    return metrics


def compute_statistical_measures(per_conversation_counts: List[Dict[str, int]]) -> Dict[str, Dict[str, float]]:
    """
    Compute statistical measures (mean, std, median) for each marker across conversations.
    
    Args:
        per_conversation_counts: List of marker counts per conversation
        
    Returns:
        Dictionary mapping marker name to statistics (mean, std, median, min, max)
    """
    if not per_conversation_counts:
        return {}
    
    # Get all marker names
    all_markers = set()
    for counts in per_conversation_counts:
        all_markers.update(counts.keys())
    
    stats = {}
    for marker in all_markers:
        values = [counts.get(marker, 0) for counts in per_conversation_counts]
        stats[marker] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'median': np.median(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    return stats


# =============================================================================
# COMPARISON & VISUALIZATION
# =============================================================================

def create_comparison_dataframe(
    results_dict: Dict[str, Dict],
    metric_type: str = 'per_100_words'
) -> pd.DataFrame:
    """
    Create a comparison DataFrame for visualization.
    
    Args:
        results_dict: Dictionary mapping dataset name to analysis results
        metric_type: Type of metric ('per_100_words', 'per_utterance', 'per_conversation')
        
    Returns:
        DataFrame with markers as rows and datasets as columns
    """
    data = {}
    for dataset_name, results in results_dict.items():
        metrics = compute_normalized_metrics(results)
        data[dataset_name] = metrics[metric_type]
    
    df = pd.DataFrame(data)
    return df


def print_summary_table(
    results_dict: Dict[str, Dict],
    metric_type: str = 'per_100_words'
):
    """
    Print a formatted summary table of marker frequencies.
    
    Args:
        results_dict: Dictionary mapping dataset name to analysis results
        metric_type: Type of metric to display
    """
    comparison_df = create_comparison_dataframe(results_dict, metric_type)
    
    # Create table
    table_data = []
    for marker in comparison_df.index:
        row = [marker]
        row.extend([f"{comparison_df.loc[marker, dataset]:.2f}" for dataset in comparison_df.columns])
        table_data.append(row)
    
    headers = ["Marker"] + list(comparison_df.columns)
    
    print("\n" + "="*100)
    print(f"DEPRESSIVE LINGUISTIC MARKERS COMPARISON ({metric_type.replace('_', ' ').title()})")
    print("="*100)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()


def print_dataset_statistics(results_dict: Dict[str, Dict]):
    """
    Print basic statistics for each dataset.
    
    Args:
        results_dict: Dictionary mapping dataset name to analysis results
    """
    print("\n" + "="*100)
    print("DATASET STATISTICS")
    print("="*100)
    
    table_data = []
    for dataset_name, results in results_dict.items():
        table_data.append([
            dataset_name,
            results['conversation_count'],
            results['utterance_counts'],
            results['word_counts'],
            f"{results['word_counts'] / results['conversation_count']:.1f}" if results['conversation_count'] > 0 else "0",
            f"{results['utterance_counts'] / results['conversation_count']:.1f}" if results['conversation_count'] > 0 else "0"
        ])
    
    headers = ["Dataset", "Conversations", "Total Utterances", "Total Words", "Avg Words/Conv", "Avg Utterances/Conv"]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    print()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze depressive linguistic markers in patient conversations. "
                    "Automatically loads all real data and all synthetic PSI-backend pairs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/depressive_markers"),
        help="Directory to save output files"
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of patient turns to analyze per conversation (default: 20)"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        choices=["per_100_tokens", "percentage_messages", "all"],
        default=["per_100_tokens", "percentage_messages"],
        help="Normalization metrics to compute and visualize"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save results to CSV files"
    )
    
    args = parser.parse_args()
    
    # Expand "all" metrics
    if "all" in args.metrics:
        args.metrics = ["per_100_tokens", "percentage_messages"]
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Compile patterns
    print("[INFO] Compiling marker patterns...")
    compiled_patterns = compile_marker_patterns(DEPRESSIVE_MARKERS)
    
    # Load and analyze data
    results_dict = {}
    
    # Load all real data
    print("\n[INFO] Loading all real data (ESC, HOPE, AnnoMI)...")
    try:
        df = load_eeyore_dataset("all")
        real_conversations = df.to_dict('records')
        print(f"[INFO] Loaded {len(real_conversations)} real conversations")
    except Exception as e:
        print(f"[ERROR] Could not load real data: {e}")
        return
    
    print(f"[INFO] Analyzing {len(real_conversations)} real conversations...")
    results_dict['Real'] = analyze_conversations(real_conversations, compiled_patterns, args.max_turns)
    
    # Load all synthetic PSI-backend pairs
    print("\n[INFO] Loading all available PSI-backend pairs...")
    all_pairs = get_all_psi_backend_pairs()
    print(f"[INFO] Found {len(all_pairs)} unique PSI-backend pairs")
    
    # Load synthetic data for each pair
    for psi, backend_llm in sorted(all_pairs):
        label = f"{psi}-{safe_dir_name(backend_llm)}"
        
        print(f"\n[INFO] Loading synthetic data for {label}...")
        
        # Load synthetic conversations
        try:
            df = load_synthetic_hf_to_df(psi=psi, backend_llm=backend_llm)
            if df.empty:
                print(f"[WARNING] No data found for {label}")
                continue
            
            synth_conversations = df.to_dict('records')
            print(f"[INFO] Analyzing {len(synth_conversations)} synthetic conversations ({label})...")
            results_dict[label] = analyze_conversations(synth_conversations, compiled_patterns, args.max_turns)
        except Exception as e:
            print(f"[ERROR] Failed to load {label}: {e}")
            continue
    
    # Print statistics
    print_dataset_statistics(results_dict)
    
    # Compute and display results for each metric
    for metric_type in args.metrics:
        print_summary_table(results_dict, metric_type)
        
        # Save to CSV
        if args.save_csv:
            comparison_df = create_comparison_dataframe(results_dict, metric_type)
            csv_path = args.output_dir / f"markers_{metric_type}.csv"
            comparison_df.to_csv(csv_path)
            print(f"[CSV SAVED] {csv_path}")
    
    # Save raw counts
    if args.save_csv:
        raw_counts_data = {
            dataset: results['raw_counts']
            for dataset, results in results_dict.items()
        }
        raw_df = pd.DataFrame(raw_counts_data)
        raw_csv_path = args.output_dir / "markers_raw_counts.csv"
        raw_df.to_csv(raw_csv_path)
        print(f"[CSV SAVED] {raw_csv_path}")
    
    print(f"\n[DONE] Analysis complete. Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
