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
from litellm import batch_completion
from json_repair import repair_json

from psibench.prompts.judge_prompt import create_ptc_judge_conversation_prompt
from psibench.data_loader.main_loader import load_eeyore_dataset

load_dotenv()

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class PTCClassifier:
    """Judge for Problem-Transition-Change framework classification."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the PTC judge.
        
        Args:
            config: Configuration dictionary containing eval.ptc_classifier settings
        """
        # Store model settings for batch completion
        judge_config = config.get("eval", {}).get("ptc_classifier", {})
        self.model_name = judge_config.get("model")
        self.temperature = judge_config.get("temperature", 0.3)
        
        # Get API settings
        if judge_config.get("api_base"):
            self.api_base = judge_config.get("api_base")
            self.api_key = "sk-no-key-required"
        else:
            self.api_base = os.getenv("OPENAI_BASE_URL")
            self.api_key = os.getenv("OPENAI_API_KEY")
            
    def _format_history(self, history: list[Dict[str, str]]) -> str:
        """Format conversation history for the prompt.
        Switch to Therapist/Assistant labels, makes life easier for judges
        
        Args:
            history: List of conversation messages
            
        Returns:
            Formatted conversation history string
        """
        formatted = []
        for msg in history:
            role = "Therapist" if msg["role"] == "user" else "Assistant"
            content = msg["content"]
            formatted.append(f"{role}: {content}")
            
        return "\n".join(formatted)
    
    def classify_conversation_by_conversation(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Classify all patient turns in a conversation in one LLM call with JSON output.
        
        Args:
            messages: List of conversation messages containing both patient and therapist turns
            
        Returns:
            List of classifications for each patient turn with turn_index, content, and classification
        """
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

    def classify_conversations_batch(self, conversations: List[List[Dict[str, str]]]) -> List[List[Dict[str, Any]]]:
        """Batch classify multiple conversations using litellm.batch_completion.
        
        Args:
            conversations: List of conversations, each being a list of messages
            
        Returns:
            List of classification results for each conversation
        """
        # Get the prompt template
        conversation_prompt = create_ptc_judge_conversation_prompt()
        
        # Build messages for each conversation
        all_messages = []
        for messages in conversations:
            conversation_str = self._format_history(messages)
            # Format prompt messages
            formatted_prompt = conversation_prompt.format_messages(conversation=conversation_str)
            # Convert to litellm format - map 'human' to 'user' for OpenAI compatibility
            litellm_messages = []
            for i, msg in enumerate(formatted_prompt):
                if hasattr(msg, 'type'):
                    role = msg.type
                    # Fix: Map 'human' to 'user' for OpenAI API
                    if role == 'human':
                        role = 'user'
                    elif role not in ['system', 'assistant', 'user', 'function', 'tool', 'developer']:
                        role = 'system' if i == 0 else 'user'
                else:
                    role = 'system' if i == 0 else 'user'
                
                litellm_messages.append({
                    "role": role,
                    "content": msg.content
                })
            all_messages.append(litellm_messages)
        
        # Batch completion call with error handling
        try:
            responses = batch_completion(
                model=self.model_name,
                messages=all_messages,
                temperature=self.temperature,
                api_key=self.api_key,
                api_base=self.api_base,
            )
        except Exception as e:
            print(f"Error in batch_completion call: {e}")
            return [[] for _ in conversations]
        
        # Parse all responses
        results = []
        for i, response in enumerate(responses):
            try:
                # Handle different response formats
                if isinstance(response, str):
                    response_text = response.strip()
                elif hasattr(response, 'choices') and len(response.choices) > 0:
                    response_text = response.choices[0].message.content.strip()
                elif hasattr(response, 'message') and hasattr(response.message, 'content'):
                    response_text = response.message.content.strip()
                elif hasattr(response, 'content'):
                    response_text = response.content.strip()
                else:
                    raise ValueError(f"Unexpected response format: {type(response)}")
                
                classifications = repair_json(response_text, return_objects=True)
                results.append(classifications)
            except Exception as e:
                print(f"Error parsing batch response {i}: {e}")
                print(f"Response type: {type(response)}")
                results.append([])  # Empty list on error
        
        return results


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


def analyze_real_dataset(dataset: str, indices: List[int], judge: PTCClassifier, output_dir: Path, batch_size: int = 1) -> pd.DataFrame:
    """Analyze real conversations from eeyore dataset.
    
    Args:
        dataset: Dataset type (e.g., 'esc', 'hope')
        indices: List of conversation indices to analyze
        judge: PTCClassifier instance
        output_dir: Directory to save results
        batch_size: Number of conversations to process in parallel
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    # Load the dataset
    eeyore_df = load_eeyore_dataset(dataset_type=dataset, indices=indices)
    
    print(f"Analyzing {len(eeyore_df)} real conversations from {dataset} dataset")
    print(f"Batch size: {batch_size}")
    
    # Prepare all conversations
    all_conversations = []
    all_indices = []
    for idx, row in eeyore_df.iterrows():
        all_conversations.append(row["messages"])
        all_indices.append(idx)
    
    # Process in batches
    for batch_start in tqdm(range(0, len(all_conversations), batch_size), desc="Classifying conversations"):
        batch_end = min(batch_start + batch_size, len(all_conversations))
        batch_conversations = all_conversations[batch_start:batch_end]
        batch_indices = all_indices[batch_start:batch_end]
        
        try:
            if batch_size == 1:
                # Single conversation processing
                classifications_list = [judge.classify_conversation_by_conversation(batch_conversations[0])]
            else:
                # Batch processing
                classifications_list = judge.classify_conversations_batch(batch_conversations)
            
            # Process results
            for idx, classifications in zip(batch_indices, classifications_list):
                ptc_counts = Counter([c['classification'] for c in classifications])
                total = len(classifications)
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
            print(f"Error processing batch starting at {batch_start}: {e}")
            continue
    
    return pd.DataFrame(results)


def analyze_dataset(data_dir: Path, judge: PTCClassifier, output_dir: Path, batch_size: int = 1) -> pd.DataFrame:
    """Analyze all conversations in a dataset (synthetic data from files).
    
    Args:
        data_dir: Directory containing conversation JSON files
        judge: PTCClassifier instance
        output_dir: Directory to save results
        batch_size: Number of conversations to process in parallel
        
    Returns:
        DataFrame with analysis results
    """
    results = []
    
    # Find all session files
    session_files = sorted(data_dir.glob('session_*.json'))
    
    print(f"Analyzing {len(session_files)} conversations from {data_dir}")
    print(f"Batch size: {batch_size}")
    
    # Load all conversations
    all_conversations = []
    all_files = []
    for session_file in session_files:
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_conversations.append(data.get('messages', []))
            all_files.append(session_file)
        except Exception as e:
            print(f"Error loading {session_file}: {e}")
            continue
    
    # Process in batches
    for batch_start in tqdm(range(0, len(all_conversations), batch_size), desc="Classifying conversations"):
        batch_end = min(batch_start + batch_size, len(all_conversations))
        batch_conversations = all_conversations[batch_start:batch_end]
        batch_files = all_files[batch_start:batch_end]
        
        try:
            if batch_size == 1:
                # Single conversation processing
                classifications_list = [judge.classify_conversation_by_conversation(batch_conversations[0])]
            else:
                # Batch processing
                classifications_list = judge.classify_conversations_batch(batch_conversations)
            
            # Process results
            for session_file, classifications in zip(batch_files, classifications_list):
                ptc_counts = Counter([c['classification'] for c in classifications])
                total = len(classifications)
                non_filler_total = sum([ptc_counts.get(cat, 0) for cat in ['P', 'T', 'C']])
                
                result = {
                    'file': session_file.name,
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
                detail_path = output_dir / 'details' / f"{session_file.stem}_ptc.json"
                detail_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(detail_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                    
        except Exception as e:
            print(f"Error processing batch starting at {batch_start}: {e}")
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
                           output_dir: Path, turn_threshold: int = 12):
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
    # Filter to conversations with at least turn_threshold patient messages
    real_df_filtered = real_df[real_df['total_patient_turns'] >= turn_threshold].copy()
    synthetic_df_filtered = synthetic_df[synthetic_df['total_patient_turns'] >= turn_threshold].copy()
    
    print(f"\nFiltered to {len(real_df_filtered)} real and {len(synthetic_df_filtered)} synthetic conversations with ≥turn_threshold turns")
    
    # Get turn-by-turn data
    real_turns = get_turn_classifications(real_df_filtered)
    synthetic_turns = get_turn_classifications(synthetic_df_filtered)
    
    # Filter to first 10 turns only
    real_turns = real_turns[real_turns['turn_index'] < turn_threshold].copy()
    synthetic_turns = synthetic_turns[synthetic_turns['turn_index'] < turn_threshold].copy()
    
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
    ax.set_title(f'Average PTC Progression (First {turn_threshold} Turns, Conversations with ≥{turn_threshold} Turns)', 
                fontsize=14, fontweight='bold')
    ax.set_yticks([1, 2, 3])
    ax.set_yticklabels(['Problem (P)', 'Transition (T)', 'Change (C)'])
    ax.set_ylim([0.5, 3.5])
    ax.set_xlim([-0.5, turn_threshold - 0.5])
    
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
    print(f"  - ptc_progression_stepline.png: Step line plot showing PTC progression (first {turn_threshold} turns, conversations ≥{turn_threshold} turns)")


def main():
    """Main function to run PTC classification analysis."""
    import argparse
    from datetime import datetime
    import time
    
    parser = argparse.ArgumentParser(description='Classify conversations using PTC framework')
    parser.add_argument('--dataset', type=str, default='esc', 
                       help='Dataset type for real conversations (default: esc)')
    parser.add_argument('--synthetic-dir', type=str, help='Directory with synthetic conversations')
    parser.add_argument('--output-dir', type=str, default='output/ptc_analysis',
                       help='Output directory')
    parser.add_argument('--compare', action='store_true',
                       help='Compare real vs synthetic')
    parser.add_argument('--N', type=int, default=None,
                       help='Number of conversations to classify (default: all available samples)')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file (default: configs/default.yaml)')
    parser.add_argument('--batch-size', type=int, default=1,
                       help='Number of parallel tasks to run (default: 1)')
    parser.add_argument('--turn-threshold', type=int, default=12,
                       help='Minimum number of turns for progression analysis (default: 12)')
    
    args = parser.parse_args()
    
    # Clean string arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize PTC judge
    judge = PTCClassifier(config)
    
    start_time = time.time()
    
    if args.compare and args.synthetic_dir:
        # Compare mode - need to get indices from synthetic data
        synthetic_dir = Path(args.synthetic_dir)
        
        # Extract indices from synthetic session files
        session_files = sorted(synthetic_dir.glob('session_*.json'))
        indices = [int(f.stem.split('_')[1]) for f in session_files]
        
        # Limit number of conversations if specified
        if args.N:
            indices = indices[:args.N]
            session_files = session_files[:args.N]
        
        print(f"Found {len(indices)} conversations to analyze")
        print(f"Batch size: {args.batch_size}")
        print(f"Analyzing corresponding real conversations from {args.dataset} dataset...")
        real_df = analyze_real_dataset(args.dataset, indices, judge, output_dir / 'real', batch_size=args.batch_size)
        
        print("\nAnalyzing synthetic conversations...")
        synthetic_df = analyze_dataset(synthetic_dir, judge, output_dir / 'synthetic', batch_size=args.batch_size)
        
        print("\nComparing distributions...")
        compare_distributions(real_df, synthetic_df, output_dir)
        
        print("\nGenerating visualizations...")
        visualize_distributions(real_df, synthetic_df, output_dir, turn_threshold=args.turn_threshold)
        
    elif args.synthetic_dir:
        # Analyze synthetic only
        print("Analyzing synthetic conversations...")
        df = analyze_dataset(Path(args.synthetic_dir), judge, output_dir, batch_size=args.batch_size)    
    else:
        print("Error: Please provide --real-dir and/or --synthetic-dir")
        return
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nResults saved to {output_dir}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    main()