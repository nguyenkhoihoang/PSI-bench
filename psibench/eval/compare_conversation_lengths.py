"""Compare conversation lengths between synthetic and real datasets.

Usage - automatically extracts PSI and dataset from folder path:

python -m psibench.eval.compare_conversation_lengths --all \
  --data-folder /work/hdd/bfjp/data/synthetic/test/ \
  --model hosted_vllm_openai_gpt-oss-120b
  

python -m psibench.eval.compare_conversation_lengths \
--folder /work/hdd/bfjp/data/synthetic/test/patientpsi/hosted_vllm_openai_gpt-oss-120b/hope

python -m psibench.eval.compare_conversation_lengths \
--folder /work/hdd/bfjp/data/synthetic/test/roleplaydoh/hosted_vllm_openai_gpt-oss-120b/annomi
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import yaml
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

from psibench.data_loader.main_loader import load_eeyore_dataset, load_synthetic_data_to_df


def extract_psi_and_dataset_from_path(folder_path: Path) -> Tuple[str, str]:
    """Extract PSI type and dataset from folder path.
    
    Expected path format: .../psi_type/model_name/dataset/
    e.g., /work/hdd/bfjp/data/synthetic/test/eeyore/hosted_vllm_openai_gpt-oss-120b/hope
    
    Args:
        folder_path: Path to synthetic conversations folder
        
    Returns:
        Tuple of (psi_type, dataset) where psi_type is 'eeyore', 'roleplaydoh', or 'patientpsi'
        and dataset is 'esc', 'hope', or 'annomi'
        
    Raises:
        ValueError: If PSI type or dataset cannot be determined from path
    """
    parts = folder_path.parts
    
    # Look for PSI type and dataset in path
    psi_type = None
    dataset = None
    
    valid_psi = ['eeyore', 'roleplaydoh', 'patientpsi']
    valid_datasets = ['esc', 'hope', 'annomi']
    
    # Search through path parts
    for part in parts:
        part_lower = part.lower()
        if part_lower in valid_psi:
            psi_type = part_lower
        if part_lower in valid_datasets:
            dataset = part_lower
    
    if not psi_type:
        raise ValueError(f"Could not determine PSI type from path: {folder_path}. Expected one of {valid_psi}")
    if not dataset:
        raise ValueError(f"Could not determine dataset from path: {folder_path}. Expected one of {valid_datasets}")
    
    return psi_type, dataset


def count_words(text: str) -> int:
    """Count words in a text string."""
    return len(text.split())


def count_tokens_approx(text: str) -> int:
    """Approximate token count (roughly 1 token = 4 characters)."""
    return len(text) // 4


def extract_patient_messages_by_turn(conversations: List[Dict], max_turns: int = None) -> Dict[int, List[str]]:
    """Extract patient messages organized by turn index.
    
    Args:
        conversations: List of conversation dictionaries
        max_turns: Maximum turn to analyze (None = analyze all)
        
    Returns:
        Dictionary mapping turn_index -> list of patient messages at that turn
    """
    messages_by_turn = defaultdict(list)
    
    for conv in conversations:
        messages = conv.get('messages', [])
        patient_turn_idx = 0
        
        for msg in messages:
            # Patient messages have role 'assistant'
            if msg.get('role') == 'assistant':
                content = msg.get('content', '').strip()
                if content:  # Only count non-empty messages
                    if max_turns is None or patient_turn_idx < max_turns:
                        messages_by_turn[patient_turn_idx].append(content)
                    patient_turn_idx += 1
    
    return messages_by_turn


def calculate_average_lengths(messages_by_turn: Dict[int, List[str]]) -> pd.DataFrame:
    """Calculate average word and token counts for each turn.
    
    Args:
        messages_by_turn: Dictionary mapping turn_index -> list of messages
        
    Returns:
        DataFrame with columns: turn, count, avg_words, std_words, avg_tokens, std_tokens
    """
    results = []
    
    for turn_idx in sorted(messages_by_turn.keys()):
        messages = messages_by_turn[turn_idx]
        
        if not messages:
            continue
        
        word_counts = [count_words(msg) for msg in messages]
        token_counts = [count_tokens_approx(msg) for msg in messages]
        
        results.append({
            'turn': turn_idx,
            'count': len(messages),
            'avg_words': np.mean(word_counts),
            'std_words': np.std(word_counts),
            'avg_tokens': np.mean(token_counts),
            'std_tokens': np.std(token_counts),
        })
    
    return pd.DataFrame(results)


def load_real_conversations(dataset_type: str) -> List[Dict]:
    """Load real conversations from Eeyore dataset.
    
    Args:
        dataset_type: Dataset type (e.g., 'esc', 'hope', 'annomi')
        
    Returns:
        List of conversation dictionaries
    """
    df = load_eeyore_dataset(dataset_type)
    conversations = []
    
    for _, row in df.iterrows():
        conversations.append({
            'messages': row['messages']
        })
    
    return conversations


def compare_datasets(synthetic_folder: Path, dataset_type: str, max_turns: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compare synthetic and real dataset conversation lengths.
    
    Args:
        synthetic_folder: Path to synthetic conversations folder
        dataset_type: Real dataset type to compare against
        max_turns: Maximum turn to analyze
        
    Returns:
        Tuple of (synthetic_df, real_df, comparison_df)
    """
    print(f"\n{'='*70}")
    print(f"Comparing Synthetic vs Real Conversations")
    print(f"{'='*70}\n")
    
    # Load synthetic data
    print(f"Loading synthetic conversations from: {synthetic_folder}")
    synthetic_df_raw = load_synthetic_data_to_df(str(synthetic_folder))
    synthetic_convs = synthetic_df_raw.to_dict('records')
    print(f"Found {len(synthetic_convs)} synthetic conversations")
    
    # Load real data
    print(f"\nLoading real conversations from dataset: {dataset_type}")
    real_convs = load_real_conversations(dataset_type)
    print(f"Found {len(real_convs)} real conversations")
    
    # Check if counts match
    if len(synthetic_convs) != len(real_convs):
        print(f"\n⚠️  WARNING: Conversation counts do not match!")
        print(f"   Synthetic: {len(synthetic_convs)}, Real: {len(real_convs)}")
    else:
        print(f"\n✓ Conversation counts match: {len(synthetic_convs)}")
    
    # Extract patient messages by turn
    print(f"\nAnalyzing patient messages by turn (max_turns: {max_turns or 'all'})...")
    synthetic_by_turn = extract_patient_messages_by_turn(synthetic_convs, max_turns)
    real_by_turn = extract_patient_messages_by_turn(real_convs, max_turns)
    
    # Calculate average lengths
    synthetic_df = calculate_average_lengths(synthetic_by_turn)
    real_df = calculate_average_lengths(real_by_turn)
    
    # Create comparison DataFrame
    comparison_data = []
    all_turns = sorted(set(synthetic_df['turn'].tolist() + real_df['turn'].tolist()))
    
    for turn in all_turns:
        synth_row = synthetic_df[synthetic_df['turn'] == turn]
        real_row = real_df[real_df['turn'] == turn]
        
        synth_words = synth_row['avg_words'].values[0] if not synth_row.empty else np.nan
        real_words = real_row['avg_words'].values[0] if not real_row.empty else np.nan
        synth_tokens = synth_row['avg_tokens'].values[0] if not synth_row.empty else np.nan
        real_tokens = real_row['avg_tokens'].values[0] if not real_row.empty else np.nan
        synth_count = synth_row['count'].values[0] if not synth_row.empty else 0
        real_count = real_row['count'].values[0] if not real_row.empty else 0
        
        comparison_data.append({
            'turn': turn,
            'synthetic_count': synth_count,
            'real_count': real_count,
            'synthetic_avg_words': synth_words,
            'real_avg_words': real_words,
            'words_diff': synth_words - real_words if not np.isnan(synth_words) and not np.isnan(real_words) else np.nan,
            'synthetic_avg_tokens': synth_tokens,
            'real_avg_tokens': real_tokens,
            'tokens_diff': synth_tokens - real_tokens if not np.isnan(synth_tokens) and not np.isnan(real_tokens) else np.nan,
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    return synthetic_df, real_df, comparison_df, dataset_type





def plot_word_count_comparison(synthetic_df: pd.DataFrame, real_df: pd.DataFrame, output_path: Path, dataset_type: str, psi:str):
    """Create a line graph comparing word counts between synthetic and real datasets.
    
    Args:
        synthetic_df: DataFrame with synthetic data statistics
        real_df: DataFrame with real data statistics
        output_path: Path to save the plot
        dataset_type: Dataset type (e.g., 'esc', 'hope', 'annomi')
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot lines for both datasets
    ax.plot(real_df['turn'], real_df['avg_words'], 
            marker='s', linewidth=2.5, label='Real', alpha=0.8)
    ax.plot(synthetic_df['turn'], synthetic_df['avg_words'], 
            marker='o', linewidth=2.5, label='Synthetic', alpha=0.8)
    
    # Formatting with dataset name
    ax.set_xlabel('Turn Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Word Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Patient Message Length Comparison ({dataset_type.upper()}): Synthetic vs Real', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Ensure x-axis shows integer values only
    ax.set_xticks(range(int(max(synthetic_df['turn'].max(), real_df['turn'].max())) + 1))
    
    plt.tight_layout()
    filename = f'{psi}_{dataset_type}.png'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Line graph saved to: {output_path / filename}")
    plt.close()

def compare_all_datasets(config_path: str, output_dir: str, data_folder: str = 'data/synthetic', model: str = None):
    """Compare all datasets concatenated, with multiple PSI simulators.
    
    Args:
        config_path: Path to config file
        output_dir: Output directory for results
        data_folder: Path to synthetic data folder (default: data/synthetic)
        model: Model name to filter by (e.g., gpt-4.1-mini). If None, uses first match.
    """
    print(f"\n{'='*70}")
    print(f"Comparing All Datasets (Concatenated) - Multiple PSI Simulators")
    print(f"{'='*70}\n")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    max_turns = config.get('patient').get('max_turns')
    
    # Load real data - concatenate all datasets
    print("Loading real conversations from all datasets...")
    all_real_convs = []
    for dataset in ['esc', 'hope', 'annomi']:
        try:
            df = load_eeyore_dataset(dataset)
            for _, row in df.iterrows():
                all_real_convs.append({'messages': row['messages']})
            print(f"  ✓ Loaded {len(df)} conversations from {dataset} dataset")
        except Exception as e:
            print(f"  ⚠ Error loading {dataset}: {e}")
    
    print(f"Total real conversations: {len(all_real_convs)}\n")
    
    # Calculate real data statistics
    real_by_turn = extract_patient_messages_by_turn(all_real_convs, max_turns)
    real_df = calculate_average_lengths(real_by_turn)
    print(f"Real data: {len(real_df)} turns analyzed\n")
    
    # Load synthetic data for each PSI simulator
    synthetic_data = {}
    data_folder_path = Path(data_folder)
    psi_types = ['patientpsi', 'roleplaydoh']
    
    for psi in psi_types:
        print(f"Loading synthetic conversations for {psi}...")
        all_synth_convs = []
        
        for dataset in ['esc', 'hope', 'annomi']:
            # Try to find the folder with this PSI and dataset
            # Handle both: data_folder/psi/model/dataset and data_folder/*/psi/model/dataset
            if model:
                # Try exact pattern first: psi/model/dataset
                psi_folders = list(data_folder_path.glob(f'{psi}/{model}/{dataset}'))
                # If not found, try with wildcard: */psi/model/dataset
                if not psi_folders:
                    psi_folders = list(data_folder_path.glob(f'*/{psi}/{model}/{dataset}'))
            else:
                # Try exact pattern first: psi/*/dataset
                psi_folders = list(data_folder_path.glob(f'{psi}/*/{dataset}'))
                # If not found, try with wildcard: */psi/*/dataset
                if not psi_folders:
                    psi_folders = list(data_folder_path.glob(f'*/{psi}/*/{dataset}'))
            
            if psi_folders:
                # Use the first matching folder
                synth_folder = psi_folders[0]
                try:
                    synth_df_raw = load_synthetic_data_to_df(str(synth_folder))
                    synth_convs = synth_df_raw.to_dict('records')
                    all_synth_convs.extend(synth_convs)
                    print(f"  ✓ Loaded {len(synth_convs)} conversations from {dataset} dataset ({synth_folder.parent.name})")
                except Exception as e:
                    print(f"  ⚠ Error loading {dataset} for {psi}: {e}")
            else:
                print(f"  ⚠ No folder found for {psi}/{dataset}")
        
        if all_synth_convs:
            synth_by_turn = extract_patient_messages_by_turn(all_synth_convs, max_turns)
            synth_df = calculate_average_lengths(synth_by_turn)
            synthetic_data[psi] = synth_df
            print(f"  Total {psi}: {len(all_synth_convs)} conversations, {len(synth_df)} turns analyzed\n")
        else:
            print(f"  ⚠ No synthetic data found for {psi}\n")
    
    # Create combined visualization
    output_path = Path(output_dir)
    plot_multiple_psi_comparison(real_df, synthetic_data, output_path)


def plot_multiple_psi_comparison(real_df: pd.DataFrame, synthetic_data: Dict[str, pd.DataFrame], output_path: Path):
    """Create a line graph comparing multiple PSI simulators against real data.
    
    Args:
        real_df: DataFrame with real data statistics
        synthetic_data: Dictionary mapping PSI name -> DataFrame with synthetic statistics
        output_path: Path to save the plot
    """
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot real data (solid line)
    ax.plot(real_df['turn'], real_df['avg_words'], 
            linewidth=2.5, label='Real', alpha=0.8, linestyle='-')
    
    # Plot synthetic data for each PSI simulator (orange color, different line styles)
    linestyles = ['--', ':']  # Dashed, Dotted
    for idx, (psi, synth_df) in enumerate(synthetic_data.items()):
        ax.plot(synth_df['turn'], synth_df['avg_words'], 
                linewidth=2.5, label=psi.capitalize(), alpha=0.8, color='orange', linestyle=linestyles[idx])
    
    # Formatting
    ax.set_xlabel('Turn Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Word Count', fontsize=12, fontweight='bold')
    ax.set_title(f'Patient Message Length Comparison (All Datasets): Multiple PSI Simulators vs Real', 
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    # Ensure x-axis shows integer values only
    all_turns = []
    all_turns.extend(real_df['turn'].tolist())
    for synth_df in synthetic_data.values():
        all_turns.extend(synth_df['turn'].tolist())
    
    if all_turns:
        max_turn = int(max(all_turns)) + 1
        ax.set_xticks(range(max_turn))
    
    plt.tight_layout()
    filename = 'all_datasets_multiple_psi.png'
    plt.savefig(output_path / filename, dpi=300, bbox_inches='tight')
    print(f"✓ Multi-PSI comparison graph saved to: {output_path / filename}")
    plt.close()



def main():
    """Main function to run conversation length comparison."""
    parser = argparse.ArgumentParser(
        description='Compare conversation lengths between synthetic and real datasets'
    )
    parser.add_argument('--folder', type=str, default=None,
                       help='Path to folder containing synthetic conversations (session_*.json files). Required if --all is not used.')
    parser.add_argument('--all', action='store_true',
                       help='Compare all datasets and PSI simulators together (patientpsi and roleplaydoh with real)')
    parser.add_argument('--data-folder', type=str, default='data/synthetic',
                       help='Path to synthetic data folder (default: data/synthetic). Used with --all flag.')
    parser.add_argument('--model', type=str, default=None,
                       help='Model name to filter by (e.g., gpt-4.1-mini, hosted_vllm_openai_gpt-oss-120b). If not specified, uses first match.')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--output-dir', type=str, default='output/length_comparison',
                       help='Output directory for results (default: output/length_comparison)')
    
    args = parser.parse_args()
    
    if args.all:
        compare_all_datasets(args.config, args.output_dir, args.data_folder, args.model)
    elif args.folder:
        # Original single-folder comparison
        synthetic_folder = Path(args.folder)
        psi_type, dataset_type = extract_psi_and_dataset_from_path(synthetic_folder)
        print(f"Extracted from path - PSI: {psi_type}, Dataset: {dataset_type}\n")
        
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        max_turns = config.get('patient').get('max_turns')
        
        # Run comparison
        output_path = Path(args.output_dir)
        
        synthetic_df, real_df, comparison_df, dataset_type = compare_datasets(
            synthetic_folder=synthetic_folder,
            dataset_type=dataset_type,
            max_turns=max_turns
        )
        
        # Create visualization
        plot_word_count_comparison(synthetic_df, real_df, output_path, dataset_type, psi_type)
    else:
        parser.error("Either --folder or --all must be specified")


if __name__ == '__main__':
    main()
