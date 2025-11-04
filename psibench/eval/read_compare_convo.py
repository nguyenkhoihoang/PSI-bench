"""Compare real and synthetic conversations side by side for a given index."""

import json
import os
from pathlib import Path
import pandas as pd
from tabulate import tabulate

from data_loader.esc import load_esc_data_with_indices, load_synthetic_data, load_esc_original_data


def load_synthetic_conversation(idx: int):
    """Load synthetic conversation for given index."""
    try:
        synthetic_data = load_synthetic_data('data/synthetic/esc') #replace where u save synthetic data
        if synthetic_data.empty:
            return None
            
        # Find the matching synthetic conversation
        file_path = Path('data/synthetic/esc') / f'session_{idx}.json'
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


def get_real_conversation(idx: int):
    """Get real conversation and situation for given index."""
    
    try:
        esc_df = load_esc_data_with_indices(original_indices = [idx])
        if esc_df.empty:
            print(f"[ERROR] No real conversation found at index {idx}")
            return None
            
        esc_original = load_esc_original_data()
        row = esc_df.iloc[0]  # Use first row since we only loaded one specific index
        id_source = row["id_source"]
        original_situation = esc_original[int(id_source)].get("situation", "")
        messages = row["messages"]
        
        return {
            'messages': messages,
            'situation': original_situation
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to get real conversation at index {idx}: {e}")
        return None


def compare_conversations(idx: int):
    """Compare real and synthetic conversations side by side."""
    real_data = get_real_conversation(idx)
    if not real_data:
        print("[ERROR] Could not load real conversation")
        return
        
    synthetic_data = load_synthetic_conversation(idx)
    if not synthetic_data:
        print("[ERROR] Could not load synthetic conversation")
        return
        
    real_messages = format_messages(real_data['messages'])
    synthetic_messages = format_messages(synthetic_data['messages'])
    
    # Create table data starting with situation
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
    args = parser.parse_args()
    
    compare_conversations(args.idx)


if __name__ == "__main__":
    main()
