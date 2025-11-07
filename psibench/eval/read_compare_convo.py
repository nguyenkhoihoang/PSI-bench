"""Compare real and synthetic conversations side by side for a given index."""

import json
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from data_loader.esc import load_esc_data_with_indices, load_esc_original_data
from data_loader.main_loader import load_eeyore_dataset
 
def load_synthetic_conversation(idx: int, data_dir: str):
    """Load synthetic conversation for given index."""
    try:
        file_path = Path(data_dir) / f'session_{idx}.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
            return {
                'messages': data['messages'],
                'situation': data['profile'].get('situation of the client', '')
            }
    except FileNotFoundError:
        print(f"[ERROR] No synthetic conversation found for index {idx}")
        return None


def format_messages(messages):
    """Format conversation messages with clear role separation."""
    formatted = []
    for msg in messages:
        role = msg.get('role', '').upper()
        # Replace multiple spaces and newlines with single space
        content = ' '.join(msg.get('content', '').split())
        formatted.append((role, content))
    return formatted


def get_real_conversation(idx: int, dataset: str):
    """Get real conversation and situation for given index."""
    
    try:
        eeyore_df = load_eeyore_dataset(dataset_type=dataset, indices=[idx])
        row = eeyore_df.iloc[0]  # Use first row since we only loaded one specific index
        id_source = row["id_source"]
        messages = row["messages"]
        if dataset == "esc":
            esc_original = load_esc_original_data()
            original_situation = esc_original[int(id_source)].get("situation", "")
            return {
                'messages': messages,
                'situation': original_situation
            }
        else:
            return {
                'messages': messages,
                'situation': ''
                }

    except Exception as e:
        print(f"[ERROR] Failed to get real conversation at index {idx}: {e}")
        return None


def compare_conversations(idx: int, dataset: str, data_dir: str):
    """Compare real and synthetic conversations side by side."""
    real_data = get_real_conversation(idx, dataset)
    if not real_data:
        print("[ERROR] Could not load real conversation")
        return

    synthetic_data = load_synthetic_conversation(idx, data_dir)
    if not synthetic_data:
        print("[ERROR] Could not load synthetic conversation")
        return
        
    real_messages = format_messages(real_data['messages'])
    synthetic_messages = format_messages(synthetic_data['messages'])
    table_data = [["Situation", real_data['situation'], synthetic_data['situation']]]
    
    # Add messages as rows, padding shorter conversation if needed
    max_turns = max(len(real_messages), len(synthetic_messages))
    for i in range(max_turns):
        real_msg = real_messages[i] if i < len(real_messages) else ('', '')
        syn_msg = synthetic_messages[i] if i < len(synthetic_messages) else ('', '')
        
        # Format each message as "ROLE: content"
        real_formatted = f"{real_msg[0]}: {real_msg[1]}" if real_msg[0] else ''
        syn_formatted = f"{syn_msg[0]}: {syn_msg[1]}" if syn_msg[0] else ''
        
        table_data.append([f"Message {i+1}", real_formatted, syn_formatted])
    
    comparison_table = tabulate(
        table_data,
        headers=["", "Real", "Synthetic"],
        tablefmt="grid",
        maxcolwidths=[10, 60, 60],  # Adjust column widths
        numalign="left"  # Align turn numbers to the left
    )
    
    output_dir = Path('output/convos')
    output_dir.mkdir(parents=True, exist_ok=True)    
    output_path = output_dir / f'convo_{idx}.txt'
    with open(output_path, 'w') as f:
        f.write(comparison_table)    
    print(f"\nComparison saved to: {output_path}")
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare real and synthetic conversations')
    parser.add_argument('idx', type=int, help='Index of the conversation to compare')
    parser.add_argument("--dataset", type=str, default="esc", help="Dataset type (default: esc)")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Synthetic data directory")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    args = parser.parse_args()
    
    # Clean string input arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    args.data_dir = args.data_dir.strip() if args.data_dir else args.data_dir
    args.psi = args.psi.strip().lower() if args.psi else args.psi
    
    data_dir = Path(args.data_dir) / args.psi / args.dataset
    print(data_dir)
    compare_conversations(args.idx, args.dataset, data_dir)


if __name__ == "__main__":
    main()
