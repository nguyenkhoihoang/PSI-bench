"""
PTC Classification: Classify patient turns as Problem, Transition, or Change.

This module uses an LLM-judge to classify each patient turn in therapy conversations
according to the Problem-Transition-Change framework.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv
import yaml

from psibench.agents.judge import JudgeAgent
from psibench.prompts.judge_prompt import create_ptc_judge_turn_prompt, create_ptc_judge_conversation_prompt
from psibench.data_loader.main_loader import load_eeyore_dataset

load_dotenv()

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class PTCClassifier(JudgeAgent):
    """Judge for Problem-Transition-Change framework classification."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the PTC judge.
        
        Args:
            config: Configuration dictionary containing eval.ptc_classifier settings
        """
        super().__init__("ptc_classifier", config)
        self.chain = create_ptc_judge_turn_prompt() | self.llm
    
    ## Archived: more fine-grained message classification
    # def classify_turn(self, patient_turn: str, context: List[Dict[str, str]] = None) -> str:
    #     """Classify a single patient turn.
        
    #     Args:
    #         patient_turn: The patient's message to classify
    #         context: Previous conversation messages for context
            
    #     Returns:
    #         Classification string: 'P', 'T', or 'C'
    #     """
    #     # Format context using base agent's method if needed, or do it inline
    #     if context:
    #         context_str = "\n".join([
    #             f"{msg['role'].capitalize()}: {msg['content']}"
    #             for msg in context[-4:]  # Use up to last 4 messages for context
    #         ])
    #     else:
    #         context_str = "No previous context"
        
        
    #     result = self.chain.invoke({
    #         "context": context_str,
    #         "patient_turn": patient_turn
    #     })
        
    #     # Parse the response - extract first occurrence of P, T, or C
    #     response_text = result.content.strip().upper()
    #     if response_text in ['P', 'T', 'C']:
    #         return response_text
        
    #     # If no valid classification found, raise an assertion error
    #     raise ValueError(f"Error: Could not parse classification from response: '{response_text}'. Expected one of 'P', 'T', or 'C'.")
    
    # def classify_conversation_by_turn(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    #     """Classify all patient turns in a conversation.
        
    #     Args:
    #         messages: List of conversation messages
            
    #     Returns:
    #         List of classifications for each patient turn
    #     """
    #     classifications = []
        
    #     for i, msg in enumerate(messages):
    #         if msg["role"] in ["assistant", "patient"]:
    #             # Get context (all messages before this one)
    #             context = messages[:i] if i > 0 else []
                
    #             classification = self.classify_turn(msg["content"], context)
                
    #             classifications.append({
    #                 # "turn_index": i // 2,  # Convert message index to turn number (1, 2, 3, ...)
    #                 "content": msg["content"],
    #                 "classification": classification,
    #             })
        
    #     return classifications
    ## END Archived
    
    def classify_conversation_by_conversation(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Classify all patient turns in a conversation in one LLM call with JSON output.
        
        Args:
            messages: List of conversation messages containing both patient and therapist turns
            
        Returns:
            List of classifications for each patient turn with turn_index, content, and classification
        """
        from json_repair import repair_json
        
        # Format the conversation history using parent's method
        conversation_str = self._format_history(messages)
        
        # Create conversation-level prompt and chain
        conversation_prompt = create_ptc_judge_conversation_prompt()
        conversation_chain = conversation_prompt | self.llm
        
        # Get response
        result = conversation_chain.invoke({
            "conversation": conversation_str
        })
        
        # Parse JSON response with repair_json
        response_text = result.content.strip()
        try:
            classifications = repair_json(response_text, return_objects=True)
            return classifications
            
        except Exception as e:
            raise ValueError(f"Error parsing JSON response: {e}\nResponse was: {response_text}")


def analyze_conversation_file(file_path: Path, judge: PTCClassifier) -> Dict[str, Any]:
    """Analyze a single conversation file.
    
    Args:
        file_path: Path to conversation JSON file
        judge: PTCClassifier instance
        
    Returns:
        Dictionary with conversation analysis
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    messages = data.get('messages', [])
    classifications = judge.classify_conversation_by_conversation(messages)
    
    # Calculate distribution (exclude F from ratios)
    ptc_counts = Counter([c['classification'] for c in classifications])
    total = len(classifications)
    # Count only P, T, C for ratio calculation (exclude F)
    non_filler_total = sum([ptc_counts.get(cat, 0) for cat in ['P', 'T', 'C']])
    
    return {
        'file': file_path.name,
        'total_patient_turns': total,
        'non_filler_turns': non_filler_total,
        'P_count': ptc_counts.get('P', 0),
        'T_count': ptc_counts.get('T', 0),
        'C_count': ptc_counts.get('C', 0),
        'F_count': ptc_counts.get('F', 0),
        'P_ratio': ptc_counts.get('P', 0) / non_filler_total if non_filler_total > 0 else 0,
        'T_ratio': ptc_counts.get('T', 0) / non_filler_total if non_filler_total > 0 else 0,
        'C_ratio': ptc_counts.get('C', 0) / non_filler_total if non_filler_total > 0 else 0,
        'F_ratio': ptc_counts.get('F', 0) / total if total > 0 else 0,
        'classifications': classifications
    }


def analyze_real_dataset(dataset: str, indices: List[int], judge: PTCClassifier, output_dir: Path) -> pd.DataFrame:
    """Analyze real conversations from eeyore dataset.
    
    Args:
        dataset: Dataset type (e.g., 'esc', 'hope')
        indices: List of conversation indices to analyze
        judge: PTCClassifier instance
        output_dir: Directory to save results
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    # Load the dataset
    eeyore_df = load_eeyore_dataset(dataset_type=dataset, indices=indices)
    
    print(f"Analyzing {len(eeyore_df)} real conversations from {dataset} dataset")
    
    for idx, row in tqdm(eeyore_df.iterrows(), total=len(eeyore_df), desc="Classifying conversations"):
        try:
            messages = row["messages"]
            classifications = judge.classify_conversation_by_conversation(messages)
            
            # Calculate distribution (exclude F from ratios)
            ptc_counts = Counter([c['classification'] for c in classifications])
            total = len(classifications)
            # Count only P, T, C for ratio calculation (exclude F)
            non_filler_total = sum([ptc_counts.get(cat, 0) for cat in ['P', 'T', 'C']])
            
            result = {
                'file': f'real_conversation_{idx}',
                'index': idx,
                'total_patient_turns': total,
                'non_filler_turns': non_filler_total,
                'P_count': ptc_counts.get('P', 0),
                'T_count': ptc_counts.get('T', 0),
                'C_count': ptc_counts.get('C', 0),
                'F_count': ptc_counts.get('F', 0),
                'P_ratio': ptc_counts.get('P', 0) / non_filler_total if non_filler_total > 0 else 0,
                'T_ratio': ptc_counts.get('T', 0) / non_filler_total if non_filler_total > 0 else 0,
                'C_ratio': ptc_counts.get('C', 0) / non_filler_total if non_filler_total > 0 else 0,
                'F_ratio': ptc_counts.get('F', 0) / total if total > 0 else 0,
                'classifications': classifications
            }
            
            results.append(result)
            
            # Save detailed classification for this conversation
            detail_path = output_dir / 'details' / f"conversation_{idx}_ptc.json"
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(detail_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing conversation {idx}: {e}")
            continue
    
    return pd.DataFrame(results)


def analyze_dataset(data_dir: Path, judge: PTCClassifier, output_dir: Path) -> pd.DataFrame:
    """Analyze all conversations in a dataset (synthetic data from files).
    
    Args:
        data_dir: Directory containing conversation JSON files
        judge: PTCClassifier instance
        output_dir: Directory to save results
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    # Find all session files
    session_files = sorted(data_dir.glob('session_*.json'))
    
    print(f"Analyzing {len(session_files)} conversations from {data_dir}")
    
    for session_file in tqdm(session_files, desc="Classifying conversations"):
        try:
            result = analyze_conversation_file(session_file, judge)
            results.append(result)
            
            # Save detailed classification for this conversation
            detail_path = output_dir / 'details' / f"{session_file.stem}_ptc.json"
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(detail_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            print(f"Error processing {session_file}: {e}")
            continue
    
    return pd.DataFrame(results)


def compare_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, output_dir: Path):
    """Compare PTC distributions between real and synthetic data.
    
    Args:
        real_df: DataFrame with real conversation analysis
        synthetic_df: DataFrame with synthetic conversation analysis
        output_dir: Directory to save comparison results
    """
    # Calculate aggregate statistics (P, T, C averaged over non-filler turns only)
    real_stats = {
        'P_mean': real_df['P_ratio'].mean(),  # Already calculated over non-filler turns
        'T_mean': real_df['T_ratio'].mean(),
        'C_mean': real_df['C_ratio'].mean(),
        'F_mean': real_df['F_ratio'].mean(),  # Calculated over all turns
        'P_std': real_df['P_ratio'].std(),
        'T_std': real_df['T_ratio'].std(),
        'C_std': real_df['C_ratio'].std(),
        'F_std': real_df['F_ratio'].std(),
        'total_turns': real_df['total_patient_turns'].sum(),
        'non_filler_turns': real_df['non_filler_turns'].sum()
    }
    
    synthetic_stats = {
        'P_mean': synthetic_df['P_ratio'].mean(),  # Already calculated over non-filler turns
        'T_mean': synthetic_df['T_ratio'].mean(),
        'C_mean': synthetic_df['C_ratio'].mean(),
        'F_mean': synthetic_df['F_ratio'].mean(),  # Calculated over all turns
        'P_std': synthetic_df['P_ratio'].std(),
        'T_std': synthetic_df['T_ratio'].std(),
        'C_std': synthetic_df['C_ratio'].std(),
        'F_std': synthetic_df['F_ratio'].std(),
        'total_turns': synthetic_df['total_patient_turns'].sum(),
        'non_filler_turns': synthetic_df['non_filler_turns'].sum()
    }
    
    # Create comparison DataFrame (P, T, C are proportions of non-filler turns)
    comparison = pd.DataFrame({
        'Real_Mean': [real_stats['P_mean'], real_stats['T_mean'], real_stats['C_mean']],
        'Real_Std': [real_stats['P_std'], real_stats['T_std'], real_stats['C_std']],
        'Synthetic_Mean': [synthetic_stats['P_mean'], synthetic_stats['T_mean'], synthetic_stats['C_mean']],
        'Synthetic_Std': [synthetic_stats['P_std'], synthetic_stats['T_std'], synthetic_stats['C_std']],
        'Difference': [
            synthetic_stats['P_mean'] - real_stats['P_mean'],
            synthetic_stats['T_mean'] - real_stats['T_mean'],
            synthetic_stats['C_mean'] - real_stats['C_mean']
        ]
    }, index=['Problem (P)', 'Transition (T)', 'Change (C)'])
    
    with open(output_dir / 'ptc_comparison.txt', 'w') as f:
        f.write("PTC Distribution Comparison: Real vs Synthetic\n")
        f.write("=" * 70 + "\n\n")
        f.write("Note: P, T, C ratios are calculated over non-filler turns only\n")
        f.write("=" * 70 + "\n\n")
        f.write(comparison.to_string())
        f.write(f"\n\n--- Turn Statistics ---")
        f.write(f"\nTotal Real Turns: {real_stats['total_turns']}")
        f.write(f"\nNon-Filler Real Turns: {real_stats['non_filler_turns']}")
        f.write(f"\nReal Filler Ratio: {real_stats['F_mean']:.3f}")
        f.write(f"\n\nTotal Synthetic Turns: {synthetic_stats['total_turns']}")
        f.write(f"\nNon-Filler Synthetic Turns: {synthetic_stats['non_filler_turns']}")
        f.write(f"\nSynthetic Filler Ratio: {synthetic_stats['F_mean']:.3f}\n")
    
    print("\nPTC Distribution Comparison:")
    print(comparison)
    
    return comparison


def get_turn_classifications(df: pd.DataFrame) -> pd.DataFrame:
    """Extract turn-by-turn classifications from analysis results.
    
    Args:
        df: DataFrame with conversation analysis including classifications
        
    Returns:
        DataFrame with columns: conversation_id, turn_index, classification
    """
    turn_data = []
    
    for idx, row in df.iterrows():
        classifications = row['classifications']
        for turn_idx, turn in enumerate(classifications):
            turn_data.append({
                'conversation_id': idx,
                'turn_index': turn_idx,
                'classification': turn['classification']
            })
    
    return pd.DataFrame(turn_data)


def visualize_distributions(real_df: pd.DataFrame, synthetic_df: pd.DataFrame, 
                           output_dir: Path):
    """Create visualizations comparing PTC distributions.
    
    Args:
        real_df: DataFrame with real conversation analysis
        synthetic_df: DataFrame with synthetic conversation analysis
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate average counts as ratios of total turns (including fillers)
    real_p_ratio = real_df['P_count'].sum() / real_df['total_patient_turns'].sum()
    real_t_ratio = real_df['T_count'].sum() / real_df['total_patient_turns'].sum()
    real_c_ratio = real_df['C_count'].sum() / real_df['total_patient_turns'].sum()
    real_f_ratio = real_df['F_count'].sum() / real_df['total_patient_turns'].sum()
    
    synthetic_p_ratio = synthetic_df['P_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_t_ratio = synthetic_df['T_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_c_ratio = synthetic_df['C_count'].sum() / synthetic_df['total_patient_turns'].sum()
    synthetic_f_ratio = synthetic_df['F_count'].sum() / synthetic_df['total_patient_turns'].sum()
    
    real_means = [real_p_ratio, real_t_ratio, real_c_ratio]
    synthetic_means = [synthetic_p_ratio, synthetic_t_ratio, synthetic_c_ratio]
    real_filler_mean = real_f_ratio
    synthetic_filler_mean = synthetic_f_ratio
    
    # 1. Stacked bar chart of average P, T, C, F ratios
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    colors = ['#e74c3c', '#f39c12', '#27ae60', '#95a5a6']
    labels = ['Problem', 'Transition', 'Change', 'Filler']
    
    # Real stacked bar
    bottom = 0
    real_values = real_means + [real_filler_mean]
    for i, (value, color, label) in enumerate(zip(real_values, colors, labels)):
        axes[0].bar(0, value, bottom=bottom, label=label, alpha=0.8, color=color, width=0.5)
        # Add value label
        if value > 0.02:  # Only show label if segment is large enough
            axes[0].text(0, bottom + value/2, f'{value:.2f}', 
                    ha='center', va='center', fontsize=12, fontweight='bold')
        bottom += value
    
    axes[0].set_xlim([-0.5, 0.5])
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Ratio', fontsize=12)
    axes[0].set_title('Real Conversations', fontsize=14, fontweight='bold')
    axes[0].set_xticks([])
    axes[0].legend(loc='upper right')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Synthetic stacked bar
    bottom = 0
    synthetic_values = synthetic_means + [synthetic_filler_mean]
    for i, (value, color, label) in enumerate(zip(synthetic_values, colors, labels)):
        axes[1].bar(0, value, bottom=bottom, label=label, alpha=0.8, color=color, width=0.5)
        # Add value label
        if value > 0.02:  # Only show label if segment is large enough
            axes[1].text(0, bottom + value/2, f'{value:.2f}', 
                        ha='center', va='center', fontsize=12, fontweight='bold')
        bottom += value
    
    axes[1].set_xlim([-0.5, 0.5])
    axes[1].set_ylim([0, 1])
    axes[1].set_ylabel('Ratio', fontsize=12)
    axes[1].set_title('Synthetic Conversations', fontsize=14, fontweight='bold')
    axes[1].set_xticks([])
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Average PTCF Distribution', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / 'ptc_stacked_average.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Step line plot showing PTC progression through conversation turns
    # Filter to conversations with at least 10 patient messages
    real_df_filtered = real_df[real_df['total_patient_turns'] >= 10].copy()
    synthetic_df_filtered = synthetic_df[synthetic_df['total_patient_turns'] >= 10].copy()
    
    print(f"\nFiltered to {len(real_df_filtered)} real and {len(synthetic_df_filtered)} synthetic conversations with ≥10 turns")
    
    # Get turn-by-turn data
    real_turns = get_turn_classifications(real_df_filtered)
    synthetic_turns = get_turn_classifications(synthetic_df_filtered)
    
    # Filter to first 10 turns only
    real_turns = real_turns[real_turns['turn_index'] < 10].copy()
    synthetic_turns = synthetic_turns[synthetic_turns['turn_index'] < 10].copy()
    
    # Map classifications to numeric values: P=1, T=2, C=3, F=NaN
    ptc_map = {'P': 1, 'T': 2, 'C': 3, 'F': np.nan}
    real_turns['ptc_numeric'] = real_turns['classification'].map(ptc_map)
    synthetic_turns['ptc_numeric'] = synthetic_turns['classification'].map(ptc_map)
    
    # Average PTC value per turn index across all conversations (F will be NaN and excluded)
    real_avg = real_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
    synthetic_avg = synthetic_turns.groupby('turn_index')['ptc_numeric'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot step lines for non-filler turns only
    ax.step(real_avg['turn_index'], real_avg['ptc_numeric'], 
            where='mid', label='Real', linewidth=2.5, alpha=0.8, color='steelblue')
    ax.step(synthetic_avg['turn_index'], synthetic_avg['ptc_numeric'], 
            where='mid', label='Synthetic', linewidth=2.5, alpha=0.8, color='coral')
    
    ax.set_xlabel('Turn Index', fontsize=12)
    ax.set_ylabel('PTC Classification', fontsize=12)
    ax.set_title('Average PTC Progression (First 10 Turns, Conversations with ≥10 Turns)', 
                fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Problem (P)', 'Transition (T)', 'Change (C)'])
    ax.set_ylim([0.5, 3.5])
    ax.set_xlim([-0.5, 9.5])
    
    # Force x-axis to show only integer values
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ptc_progression_stepline.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization plots saved to {output_dir}/")
    print("  - ptc_stacked_average.png: Stacked bar chart of average P/T/C/F ratios (all conversations)")
    print("  - ptc_progression_stepline.png: Step line plot showing PTC progression (first 10 turns, conversations ≥10 turns)")


def main():
    """Main function to run PTC classification analysis."""
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Classify conversations using PTC framework')
    parser.add_argument('--dataset', type=str, default='esc', 
                       help='Dataset type for real conversations (default: esc)')
    parser.add_argument('--synthetic-dir', type=str, help='Directory with synthetic conversations')
    parser.add_argument('--output-dir', type=str, default='output/ptc_analysis',
                       help='Output directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare real vs synthetic')
    
    args = parser.parse_args()
    
    # Clean string arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    
    # Load configuration
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize PTC judge
    judge = PTCClassifier(config)
    
    if args.compare and args.synthetic_dir:
        # Compare mode - need to get indices from synthetic data
        synthetic_dir = Path(args.synthetic_dir)
        
        # Extract indices from synthetic session files
        session_files = sorted(synthetic_dir.glob('session_*.json'))
        indices = [int(f.stem.split('_')[1]) for f in session_files]
        
        print(f"Found {len(indices)} synthetic conversations")
        print(f"Analyzing corresponding real conversations from {args.dataset} dataset...")
        real_df = analyze_real_dataset(args.dataset, indices, judge, output_dir / 'real')
        
        print("\nAnalyzing synthetic conversations...")
        synthetic_df = analyze_dataset(synthetic_dir, judge, output_dir / 'synthetic')
        
        print("\nComparing distributions...")
        compare_distributions(real_df, synthetic_df, output_dir)
        
        print("\nGenerating visualizations...")
        visualize_distributions(real_df, synthetic_df, output_dir)
        
    elif args.synthetic_dir:
        # Analyze synthetic only
        print("Analyzing synthetic conversations...")
        df = analyze_dataset(Path(args.synthetic_dir), judge, output_dir)    
    else:
        print("Error: Please provide --real-dir and/or --synthetic-dir")
        return
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()