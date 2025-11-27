"""Compare real and synthetic conversations side by side for a given index."""

import json
import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# --- 1. IMPORT EMOATLAS ---
from emoatlas import EmoScores

from psibench.data_loader.esc import load_esc_original_data
from psibench.data_loader.main_loader import load_eeyore_dataset

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def get_significant_emotions(text, analyzer):
    """
    Analyzes text and returns a dictionary of emotions with Z-score > 1.
    """
    if not text or not analyzer:
        return {}
    
    try:
        # Get all scores
        all_scores = analyzer.zscores(text)
        # Filter for only those > 1 (statistically significant)
        # Round to 2 decimals for cleaner display
        return {k: round(v, 2) for k, v in all_scores.items() if v > 1}
    except Exception as e:
        print(f"Error calculating emotions: {e}")
        return {}

def prepare_emotion_data(text_list, ea):
    """Helper: Calculates z-scores for each turn and returns a DataFrame."""
    timeline_data = []
    
    for i, text in enumerate(text_list):
        if not text or len(text.strip()) < 2:
            continue
        scores = ea.zscores(text)
        scores['turn_id'] = i + 1
        timeline_data.append(scores)
        
    return pd.DataFrame(timeline_data)

# ==========================================
# VISUALIZATION FUNCTIONS
# ==========================================

def visualize_comparison(real_msgs_list, syn_msgs_list, ea):
    """Generates charts comparing Real vs Synthetic patient emotions."""
    if not real_msgs_list or not syn_msgs_list:
        print("[VISUALIZATION SKIPPED] Not enough messages to visualize.")
        return

    # --- A. Global Profile (Bar Chart) ---
    real_full_text = " ".join(real_msgs_list)
    syn_full_text = " ".join(syn_msgs_list)
    
    real_global_scores = ea.zscores(real_full_text)
    syn_global_scores = ea.zscores(syn_full_text)

    emotions = list(real_global_scores.keys())
    df_global = pd.DataFrame({
        'Emotion': emotions,
        'Real': [real_global_scores[e] for e in emotions],
        'Synthetic': [syn_global_scores[e] for e in emotions]
    })

    # Plot 1: Global Profile
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    x = np.arange(len(emotions))
    width = 0.35

    ax1.bar(x - width/2, df_global['Real'], width, label='Real Patient', color='#1f77b4', alpha=0.8)
    ax1.bar(x + width/2, df_global['Synthetic'], width, label='Synthetic Patient', color='#ff7f0e', alpha=0.8)

    ax1.set_ylabel('Emotional Intensity (Z-Score)')
    ax1.set_title('Global Emotional Profile (Patient Role): Real vs. Synthetic')
    ax1.set_xticks(x)
    ax1.set_xticklabels(emotions)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    ax1.axhline(y=1, color='r', linestyle=':', label='Significance Threshold (>1)')
    plt.tight_layout()
    plt.show()

    # --- B. Fluctuation (Line Chart) ---
    df_real = prepare_emotion_data(real_msgs_list, ea)
    df_syn = prepare_emotion_data(syn_msgs_list, ea)
    
    # Check if we have data to plot
    if df_real.empty or df_syn.empty:
        print("[VISUALIZATION SKIPPED] Not enough valid text turns for timeline.")
        return

    # Define negative emotions to track distress
    neg_emotions = ['anger', 'disgust', 'fear', 'sadness']
    
    df_real['avg_negative'] = df_real[neg_emotions].mean(axis=1)
    df_syn['avg_negative'] = df_syn[neg_emotions].mean(axis=1)

    # Plot 2: Fluctuation
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(df_real['turn_id'], df_real['avg_negative'], marker='o', label='Real Patient (Negative)', color='#1f77b4')
    ax2.plot(df_syn['turn_id'], df_syn['avg_negative'], marker='s', label='Synthetic Patient (Negative)', color='#ff7f0e', linestyle='--')

    ax2.set_xlabel('Conversation Turn')
    ax2.set_ylabel('Negative Intensity (Avg Z-Score)')
    ax2.set_title('Fluctuation of Negative Emotions Throughout Conversation')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

# ==========================================
# UPDATED DATA LOADER FUNCTIONS
# ==========================================

def load_synthetic_conversation(idx: int, data_dir: str, analyzer=None, turn: bool=False):
    """
    Load synthetic conversation and calculate emotion scores for ASSISTANT role.
    UPDATED: Handles 'turn' folder structure.
    """
    print(f"Loading synthetic conversation (Index {idx})...")
    try:
        # --- PATH CHANGE HERE ---
        if turn:
            # Look inside the 'turn' folder
            file_path = "data/synthetic/eeyore/gpt-4.1-mini/esc/turn/" + f'session_{idx}.json'
        else:
            # Look in the main folder
            file_path = data_dir / f'session_{idx}.json'
        
        print(f"Reading: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            messages = data["messages"]
            
            # --- EXTRACT ASSISTANT TEXT & EMOTIONS ---
            assistant_msgs = [
                msg.get("content", "") 
                for msg in messages 
                if msg.get("role", "").lower() == "assistant"
            ]
            full_assistant_text = " ".join(assistant_msgs)
            
            emotions = get_significant_emotions(full_assistant_text, analyzer)
            # -----------------------------------------

            return {
                "messages": messages,
                "situation": data["profile"].get("situation of the client", ""),
                "emotions": emotions  # Return calculated scores
            }
    except FileNotFoundError:
        print(f"[ERROR] Synthetic file not found for index {idx} at path: {file_path}")
        return None

def format_messages(messages):
    """Format conversation messages with clear role separation."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "").upper()
        content = " ".join(msg.get("content", "").split())
        formatted.append((role, content))
    return formatted

def get_real_conversation(idx: int, dataset: str, analyzer=None):
    """
    Get real conversation and calculate emotion scores for ASSISTANT role.
    """
    try:
        eeyore_df = load_eeyore_dataset(dataset_type=dataset, indices=[idx])
        row = eeyore_df.iloc[0]
        id_source = row["id_source"]
        messages = row["messages"]
        
        situation = ""
        if dataset == "esc":
            esc_original = load_esc_original_data()
            if int(id_source) < len(esc_original):
                situation = esc_original[int(id_source)].get("situation", "")

        # --- EXTRACT ASSISTANT TEXT & EMOTIONS ---
        assistant_msgs = [
            msg.get("content", "") 
            for msg in messages 
            if msg.get("role", "").lower() == "assistant"
        ]
        full_assistant_text = " ".join(assistant_msgs)
        
        emotions = get_significant_emotions(full_assistant_text, analyzer)
        # -----------------------------------------

        return {
            'messages': messages,
            'situation': situation,
            'emotions': emotions  # Return calculated scores
        }
    except Exception as e:
        print(f"[ERROR] Failed to get real conversation at index {idx}: {e}")
        return None

def compare_conversations(idx: int, dataset: str, patient_sim: str, data_dir: str):
    """Compare real and synthetic conversations side by side."""
    
    # --- 1. INITIALIZE EMOATLAS (ONCE) ---
    print("Initializing EmoAtlas...")
    ea = EmoScores(language='english')

    # --- 2. LOAD DATA (Pass analyzer) ---
    real_data = get_real_conversation(idx, dataset, analyzer=ea)
    if not real_data:
        print("[ERROR] Could not load real conversation")
        return

    synthetic_data = load_synthetic_conversation(idx, data_dir, analyzer=ea)
    if not synthetic_data:
        print("[ERROR] Could not load synthetic conversation")
        return
    
    turn_synthetic_data = load_synthetic_conversation(idx, data_dir, analyzer=ea, turn=True)
    if not turn_synthetic_data:
        print("[ERROR] Could not load turn synthetic conversation")
        return
        
    # --- 3. PRINT EMOTION SCORES ---
    print("\n" + "="*50)
    print(f"EMOTION ANALYSIS (PATIENT ROLE) - Index {idx}")
    print("="*50)
    print(f"REAL Patient Emotions (>1.0):")
    print(json.dumps(real_data['emotions'], indent=2))
    print("-" * 30)
    print(f"SYNTHETIC Patient Emotions (>1.0):")
    print(json.dumps(synthetic_data['emotions'], indent=2))
    print("="*50 + "\n")

    # --- 4. GENERATE TABLE ---
    real_messages = format_messages(real_data['messages'])
    synthetic_messages = format_messages(synthetic_data['messages'])
    turn_synthetic_messages = format_messages(turn_synthetic_data['messages'])
    
    table_data = [["Situation", real_data['situation'], synthetic_data['situation'], turn_synthetic_data['situation']]]
    max_turns = max(len(real_messages), len(synthetic_messages))
    
    for i in range(max_turns):
        real_msg = real_messages[i] if i < len(real_messages) else ("", "")
        syn_msg = synthetic_messages[i] if i < len(synthetic_messages) else ("", "")
        turn_syn_msg = turn_synthetic_messages[i] if i < len(turn_synthetic_messages) else ("", "")

        real_formatted = f"{real_msg[0]}: {real_msg[1]}" if real_msg[0] else ""
        syn_formatted = f"{syn_msg[0]}: {syn_msg[1]}" if syn_msg[0] else ""
        turn_syn_formatted = f"{turn_syn_msg[0]}: {turn_syn_msg[1]}" if turn_syn_msg[0] else ""

        table_data.append([f"Message {i+1}", real_formatted, syn_formatted, turn_syn_formatted])

    comparison_table = tabulate(
        table_data,
        headers=["", "Real", "Synthetic", "Synthetic_turn"],
        tablefmt="grid",
        maxcolwidths=[10, 60, 60, 60],
        numalign="left",
    )

    output_dir = Path(f"output/{patient_sim}/{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"convo_{idx}.txt"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(comparison_table)
    print(f"\nComparison saved to: {output_path}")

    # ==========================================
    # 5. RUN VISUALIZATION
    # ==========================================
    
    # Extract Patient Messages (Role = ASSISTANT) for Visualization
    real_patient_msgs = [msg[1] for msg in real_messages if msg[0] == 'ASSISTANT']
    syn_patient_msgs = [msg[1] for msg in synthetic_messages if msg[0] == 'ASSISTANT']
    
    print("\nGenerating Charts...")
    # Pass the existing 'ea' instance to avoid reloading
    visualize_comparison(real_patient_msgs, syn_patient_msgs, ea)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare real and synthetic conversations')
    parser.add_argument('idx', type=int, help='Index of the conversation to compare')
    parser.add_argument("--dataset", type=str, default="esc", help="Dataset type (default: esc)")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Synthetic data directory")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    args = parser.parse_args()
    
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    args.data_dir = args.data_dir.strip() if args.data_dir else args.data_dir
    args.psi = args.psi.strip().lower() if args.psi else args.psi
    
    data_dir = Path(args.data_dir) / args.psi / args.dataset
    print(data_dir)
    compare_conversations(args.idx, args.dataset, args.psi, data_dir)


if __name__ == "__main__":
    main()