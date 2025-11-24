"""
Calculate and compare Lexical Diversity (MTLD, MATTR) between real and synthetic conversations
using strict preprocessing and length control strategies.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lexicalrichness import LexicalRichness

from data_loader.main_loader import load_eeyore_dataset
from data_loader.utils import merge_consecutive_messages

# Common disfluencies to remove
DISFLUENCIES = {
    "um", "uh", "hm", "hmm", "ah", "er", "huh", "mm", "mhm", "oh", 
    "like", "you know", "i mean" 
}
STRICT_DISFLUENCIES = {"um", "uh", "hm", "hmm", "ah", "er", "mm", "mhm", "huh"}


def preprocess(text: str, remove_disfluencies: bool = True) -> str:
    """
    Preprocess text: lowercase, remove punctuation, optionally remove disfluencies.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    words = text.split()
    
    if remove_disfluencies:
        words = [w for w in words if w not in STRICT_DISFLUENCIES]
        
    return " ".join(words)


def calculate_bidirectional_mtld(text: str, threshold: float = 0.72) -> float:
    """
    Calculate bidirectional MTLD (Forward + Backward) / 2.
    """
    if not text or len(text.split()) < 1: # Basic sanity check
        return float('nan')

    try:
        lex = LexicalRichness(text)
        
        # Forward MTLD
        mtld_fwd = lex.mtld(threshold=threshold)
        
        # Backward MTLD (reverse the word list)
        words = text.split()
        reversed_text = " ".join(words[::-1])
        lex_rev = LexicalRichness(reversed_text)
        mtld_bwd = lex_rev.mtld(threshold=threshold)
        
        if math.isnan(mtld_fwd) or math.isnan(mtld_bwd):
            return float('nan')
            
        return (mtld_fwd + mtld_bwd) / 2.0
        
    except Exception:
        return float('nan')





def get_assistant_messages(messages: List[Dict]) -> List[str]:
    """Extract content of messages from the 'assistant' role."""
    return [msg.get('content', '') for msg in messages if msg.get('role') == 'assistant']


def aggregate_messages(messages: List[str]) -> str:
    """Concatenate all messages into a single string."""
    return " ".join(messages)


def truncate_to_length(text: str, target_length: int) -> str:
    """Truncate text to the first `target_length` tokens."""
    words = text.split()
    if len(words) <= target_length:
        return text
    return " ".join(words[:target_length])


def analyze_session(
    real_msgs: List[str],
    synth_msgs: List[str],
    session_id: int,
    min_tokens: int = 100,
    synth_turn_msgs: Optional[List[str]] = None
) -> List[Dict]:
    """
    Analyze a single session using Raw, Matched, Cumulative, and Turn-Level strategies.
    """
    results = []
    
    # 1. Aggregation
    real_text_raw = preprocess(aggregate_messages(real_msgs))
    synth_text_raw = preprocess(aggregate_messages(synth_msgs))
    
    real_len = len(real_text_raw.split())
    synth_len = len(synth_text_raw.split())
    
    # --- Strategy A: Raw Aggregated ---
    results.append({
        'session_id': session_id,
        'speaker_type': 'Real',
        'analysis_type': 'Raw',
        'turn_index': -1, # N/A
        'token_count': real_len,
        'mtld': calculate_bidirectional_mtld(real_text_raw),
        'meets_threshold': real_len >= min_tokens
    })
        
    results.append({
        'session_id': session_id,
        'speaker_type': 'Synthetic',
        'analysis_type': 'Raw',
        'turn_index': -1, # N/A
        'token_count': synth_len,
        'mtld': calculate_bidirectional_mtld(synth_text_raw),
        'meets_threshold': synth_len >= min_tokens
    })

    # --- Strategy B: Matched Truncated ---
    # For matched, we still need some content to work with. 
    # If either is 0, we can't really match. But let's assume > 0 for now or handle gracefully.
    if real_len > 0 and synth_len > 0:
        target_len = min(real_len, synth_len)
        
        real_text_trunc = truncate_to_length(real_text_raw, target_len)
        synth_text_trunc = truncate_to_length(synth_text_raw, target_len)
        
        # Matched usually implies we want to compare "fairly". 
        # If the common length is < min_tokens, we flag it.
        meets_threshold = target_len >= min_tokens
        
        results.append({
            'session_id': session_id,
            'speaker_type': 'Real',
            'analysis_type': 'Matched',
            'turn_index': -1,
            'token_count': len(real_text_trunc.split()),
            'mtld': calculate_bidirectional_mtld(real_text_trunc),

            'meets_threshold': meets_threshold
        })
        
        results.append({
            'session_id': session_id,
            'speaker_type': 'Synthetic',
            'analysis_type': 'Matched',
            'turn_index': -1,
            'token_count': len(synth_text_trunc.split()),
            'mtld': calculate_bidirectional_mtld(synth_text_trunc),

            'meets_threshold': meets_threshold
        })
        
    # --- Strategy C: Cumulative Analysis ---
    # Real
    cum_text = ""
    for i, msg in enumerate(real_msgs):
        processed_msg = preprocess(msg)
        if not processed_msg: continue
        cum_text += " " + processed_msg
        
        current_len = len(cum_text.split())
        results.append({
            'session_id': session_id,
            'speaker_type': 'Real',
            'analysis_type': 'Cumulative',
            'turn_index': i + 1,
            'token_count': current_len,
            'mtld': calculate_bidirectional_mtld(cum_text),

            'meets_threshold': current_len >= min_tokens
        })

    # Synthetic
    cum_text = ""
    for i, msg in enumerate(synth_msgs):
        processed_msg = preprocess(msg)
        if not processed_msg: continue
        cum_text += " " + processed_msg
        
        current_len = len(cum_text.split())
        results.append({
            'session_id': session_id,
            'speaker_type': 'Synthetic',
            'analysis_type': 'Cumulative',
            'turn_index': i + 1,
            'token_count': current_len,
            'mtld': calculate_bidirectional_mtld(cum_text),

            'meets_threshold': current_len >= min_tokens
        })

    # --- Strategy D: Turn-Level Analysis ---
    # Real
    for i, msg in enumerate(real_msgs):
        processed_msg = preprocess(msg)
        if not processed_msg: continue
        
        current_len = len(processed_msg.split())
        results.append({
            'session_id': session_id,
            'speaker_type': 'Real',
            'analysis_type': 'Turn-Level',
            'turn_index': i + 1,
            'token_count': current_len,
            'mtld': calculate_bidirectional_mtld(processed_msg),

            'meets_threshold': current_len >= min_tokens
        })
            
    # Synthetic - ONLY if synth_turn_msgs is provided
    if synth_turn_msgs:
        for i, msg in enumerate(synth_turn_msgs):
            processed_msg = preprocess(msg)
            if not processed_msg: continue
            
            current_len = len(processed_msg.split())
            results.append({
                'session_id': session_id,
                'speaker_type': 'Synthetic',
                'analysis_type': 'Turn-Level',
                'turn_index': i + 1,
                'token_count': current_len,
                'mtld': calculate_bidirectional_mtld(processed_msg),
    
                'meets_threshold': current_len >= min_tokens
            })
            
    return results


def load_local_real_data(data_dir: Path, dataset: str, indices: Optional[List[int]] = None) -> pd.DataFrame:
    """Load real conversation data from local directory."""
    if dataset == 'esc':
        filename = 'ESConv.json'
    else:
        # TODO: Add support for other datasets if needed
        print(f"Warning: Local loading for {dataset} not explicitly implemented, trying {dataset}.json")
        filename = f"{dataset}.json"
        
    file_path = data_dir / filename
    if not file_path.exists():
        print(f"Error: Real data file not found at {file_path}")
        return pd.DataFrame()
        
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # Filter by indices if provided
        # Note: ESConv.json is a list of dicts, index corresponds to list index
        if indices:
            # Filter indices that are within range
            valid_indices = [i for i in indices if 0 <= i < len(data)]
            data = [data[i] for i in valid_indices]
            
        # Convert to DataFrame format expected by analysis
        rows = []
        for i, session in enumerate(data):
            # Map 'seeker'/'supporter' to 'user'/'assistant' if needed, 
            # or just ensure 'role' exists. 
            # ESConv uses 'speaker': 'seeker'/'supporter'
            messages = []
            for msg in session.get('dialog', []):
                role = 'assistant' if msg['speaker'] == 'supporter' else 'user'
                messages.append({
                    'role': role,
                    'content': msg['content']
                })
            
            # Merge consecutive messages
            merged_messages = merge_consecutive_messages(messages)
            
            rows.append({
                'messages': merged_messages,
                # Preserve original index if we filtered, otherwise use loop index
                'original_index': indices[i] if indices else i 
            })
            
        return pd.DataFrame(rows)
        
    except Exception as e:
        print(f"Error loading local real data: {e}")
        return pd.DataFrame()



def create_visualizations(df: pd.DataFrame, output_dir: Path):
    """Generate plots for the analysis results."""
    sns.set_theme(style="whitegrid")
    
    # 1. Boxplots: MTLD by Speaker Type (Raw vs Matched)
    # Filter out Cumulative/Turn-Level for this plot
    main_df = df[df['analysis_type'].isin(['Raw', 'Matched'])]
    
    if not main_df.empty:
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=main_df, x='analysis_type', y='mtld', hue='speaker_type')
        plt.title('MTLD Distribution: Real vs Synthetic (Raw vs Matched)')
        plt.ylabel('Bidirectional MTLD')
        plt.xlabel('Analysis Strategy')
        plt.tight_layout()
        plt.savefig(output_dir / 'boxplot_mtld_comparison.png')
        plt.close()
        


    # 2. Scatter: MTLD vs Token Count (Raw Data Only)
    raw_df = df[df['analysis_type'] == 'Raw']
    if not raw_df.empty:
        plt.figure(figsize=(10, 6))
        # Color by meets_threshold
        palette = {True: 'blue', False: 'red'}
        sns.scatterplot(
            data=raw_df, 
            x='token_count', 
            y='mtld', 
            hue='meets_threshold', 
            style='speaker_type',
            palette=palette,
            alpha=0.7
        )
        plt.title('MTLD vs Token Count (Raw Aggregated Data)')
        plt.xlabel('Token Count')
        plt.ylabel('Bidirectional MTLD')
        plt.tight_layout()
        plt.savefig(output_dir / 'scatter_mtld_vs_tokens.png')
        plt.close()

    # 3. Line Plots: Cumulative MTLD Progression (Aggregate)
    cum_df = df[df['analysis_type'] == 'Cumulative']
    if not cum_df.empty:
        plt.figure(figsize=(12, 8))
        sns.lineplot(
            data=cum_df, 
            x='turn_index', 
            y='mtld', 
            hue='speaker_type', 
            units='session_id', 
            estimator=None, 
            lw=1, 
            alpha=0.3
        )
        # Add average lines
        sns.lineplot(
            data=cum_df, 
            x='turn_index', 
            y='mtld', 
            hue='speaker_type', 
            estimator='mean', 
            errorbar=None, 
            lw=3,
            legend=False
        )
        
        plt.title('Cumulative MTLD Progression (All Sessions)')
        plt.xlabel('Turn Index')
        plt.ylabel('Cumulative MTLD')
        plt.tight_layout()
        plt.savefig(output_dir / 'lineplot_cumulative_mtld_all.png')
        plt.close()

    # 4. Per-Session Plots
    session_plot_dir = output_dir / 'session_plots'
    session_plot_dir.mkdir(exist_ok=True)
    
    unique_sessions = df['session_id'].unique()
    print(f"Generating per-session plots in {session_plot_dir}...")
    
    for sess_id in unique_sessions:
        sess_df = df[df['session_id'] == sess_id]
        
        # Create session-specific subdirectory
        sess_dir = session_plot_dir / f"session_{sess_id}"
        sess_dir.mkdir(exist_ok=True)
        
        # Helper to plot with annotations
        def plot_with_annotations(data, x_col, y_col, title, filename):
            plt.figure(figsize=(12, 7))
            
            # Draw lines first
            sns.lineplot(data=data, x=x_col, y=y_col, hue='speaker_type', marker=None, zorder=1)
            
            # Draw points with custom colors based on threshold
            # We iterate to handle colors and annotations
            for stype in data['speaker_type'].unique():
                subset = data[data['speaker_type'] == stype]
                
                # Plot points
                colors = subset['meets_threshold'].map({True: 'blue' if stype == 'Real' else 'orange', False: 'red'})
                plt.scatter(subset[x_col], subset[y_col], c=colors, s=30, zorder=2, label=f"{stype} (Points)")
                
                # Annotate token counts
                for _, row in subset.iterrows():
                    plt.text(
                        row[x_col], 
                        row[y_col], 
                        f"{int(row['token_count'])}", 
                        fontsize=8, 
                        ha='left', 
                        va='bottom',
                        alpha=0.7
                    )

            plt.title(title)
            plt.xlabel('Turn Index')
            plt.ylabel(y_col.upper())
            plt.tight_layout()
            plt.savefig(sess_dir / filename)
            plt.close()

        # A. Cumulative Plot (Per Session)
        cum_data = sess_df[sess_df['analysis_type'] == 'Cumulative']
        if not cum_data.empty:
            plot_with_annotations(
                cum_data, 
                'turn_index', 
                'mtld', 
                f'Session {sess_id}: Cumulative MTLD Progression', 
                'cumulative_mtld.png'
            )
            
        # B. Turn-Level Plot (Per Session)
        turn_data = sess_df[sess_df['analysis_type'] == 'Turn-Level']
        if not turn_data.empty:
            plot_with_annotations(
                turn_data, 
                'turn_index', 
                'mtld', 
                f'Session {sess_id}: Turn-Level MTLD', 
                'turn_mtld.png'
            )


def main():
    parser = argparse.ArgumentParser(description='Strict Lexical Diversity Analysis')
    parser.add_argument("--dataset", type=str, default="esc", help="Dataset type (default: esc)")
    parser.add_argument("--data-dir", type=str, required=True, help="Synthetic data directory")
    parser.add_argument("--real-data-dir", type=str, help="Real data directory (optional, overrides HF)")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    parser.add_argument("--output-dir", type=str, default="output/lexical_diversity_strict", help="Output directory")
    parser.add_argument("-k", "--num-samples", type=int, help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    # Setup paths
    data_path = Path(args.data_dir)
    extended_path = data_path / args.psi / args.dataset
    if extended_path.exists() and not data_path.name == args.dataset:
        data_path = extended_path
    
    print(f"Loading synthetic data from: {data_path}")
    if not data_path.exists():
        print(f"Error: Directory {data_path} does not exist.")
        return

    # Setup Output Directory
    output_dir = Path(args.output_dir) / args.psi / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all unique indices from session files
    all_files = sorted(data_path.glob('session_*.json'))
    indices = set()
    for f in all_files:
        # Match session_X.json but exclude _turn.json
        if '_turn' not in f.name:
            match = re.search(r'session_(\d+)\.json', f.name)
            if match:
                indices.add(int(match.group(1)))
            
    sorted_indices = sorted(list(indices))
    
    if args.num_samples:
        sorted_indices = sorted_indices[:args.num_samples]
        
    print(f"Processing {len(sorted_indices)} sessions...")
    
    all_results = []
    
    for idx in sorted_indices:
        # Load Real Data
        try:
            if args.real_data_dir:
                real_path = Path(args.real_data_dir)
                # We pass [idx] but for local file we need to handle it carefully.
                # The load_local_real_data implementation above handles a list of indices.
                # However, indices in ESConv.json are implicit (list position).
                # If the synthetic data uses the same indices as the full ESConv dataset, this works.
                real_df = load_local_real_data(real_path, args.dataset, indices=[idx])
            else:
                real_df = load_eeyore_dataset(args.dataset, indices=[idx])
                
            if real_df.empty:
                continue
            real_msgs = get_assistant_messages(real_df.iloc[0]['messages'])
        except Exception as e:
            print(f"Error loading real data for {idx}: {e}")
            continue

        # Load Synthetic Data
        synth_file = data_path / f"session_{idx}.json"
        if not synth_file.exists():
            continue
            
        try:
            with open(synth_file, 'r') as f:
                data = json.load(f)
            synth_msgs = get_assistant_messages(data['messages'])
        except Exception as e:
            print(f"Error loading synthetic data for {idx}: {e}")
            continue
            
        # Load Synthetic Turn Data (Optional)
        synth_turn_msgs = None
        # Turn data is in a 'turn' subdirectory with the same filename 'session_{idx}.json'
        turn_file = data_path / "turn" / f"session_{idx}.json"
        
        # Fallback to old naming if not found (just in case)
        if not turn_file.exists():
             turn_file = data_path / f"session_{idx}_turn.json"
             
        if turn_file.exists():
            try:
                with open(turn_file, 'r') as f:
                    turn_data = json.load(f)
                synth_turn_msgs = get_assistant_messages(turn_data['messages'])
            except Exception as e:
                print(f"Error loading synthetic turn data for {idx}: {e}")
                # Fallback to None, which means skipping turn-level analysis
                synth_turn_msgs = None
            
        # Analyze
        session_results = analyze_session(real_msgs, synth_msgs, idx, synth_turn_msgs=synth_turn_msgs)
        all_results.extend(session_results)
        
    if not all_results:
        print("No results generated.")
        return
        
    # Save Results
    df = pd.DataFrame(all_results)
    csv_path = output_dir / 'lexical_diversity_strict_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Visualize
    create_visualizations(df, output_dir)
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
