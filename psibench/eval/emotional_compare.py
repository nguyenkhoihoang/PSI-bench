"""Compare real and synthetic conversations side by side (Text & Emotions)."""

import json
import os
from pathlib import Path
import argparse as ap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from collections import defaultdict
from contextlib import contextmanager
import warnings

# Ensure emoatlas is installed
try:
    from emoatlas import EmoScores
except ImportError:
    print("[ERROR] EmoAtlas not found. Please run: pip install emoatlas")
    exit(1)

# Ensure psibench is in python path or installed
try:
    from psibench.data_loader.esc import load_esc_original_data
    from psibench.data_loader.main_loader import load_eeyore_dataset
except ImportError:
    print("[ERROR] psibench not found. Please ensure PsiBench packages are available.")
    exit(1)


# --- HELPER FUNCTIONS ---

def prepare_emotion_data(text_list, ea):
    """Calculates z-scores for each turn and returns a DataFrame."""
    timeline_data = []
    base_emotions = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]

    for i, text in enumerate(text_list):
        if not text or len(text.strip()) < 2:
            zeros = {k: 0.0 for k in base_emotions}
            zeros['turn_id'] = i + 1
            timeline_data.append(zeros)
            continue
            
        scores = ea.zscores(text)
        scores['turn_id'] = i + 1
        timeline_data.append(scores)
        
    return pd.DataFrame(timeline_data)

def _extract_patient_messages(messages):
    """Return patient-side messages (role==assistant)."""
    return [
        msg.get("content", "")
        for msg in messages
        if msg.get("role", "").lower() == "assistant"
    ]

def _resolve_synthetic_dir(base_dir: Path, psi: str, dataset: str) -> Path:
    """Resolve synthetic data directory path."""
    base_dir = Path(base_dir)
    if any(base_dir.glob("session_*.json")): return base_dir
    dataset_dir = base_dir / dataset
    if any(dataset_dir.glob("session_*.json")): return dataset_dir
    psi_dataset_dir = base_dir / psi / dataset
    return psi_dataset_dir

def _save_or_show(fig, output_dir: Path, filename: str, show_plots: bool):
    """Save figure to output_dir and optionally display."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig_path = output_dir / filename
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"[PLOT SAVED] {fig_path}")

@contextmanager
def warnings_suppressed():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        yield

def get_scores(text_list, ea):
    """
    Chunked Scoring to avoid Memory Error for large aggregated texts.
    """
    # Join all text first
    full_text = " ".join(text_list)
    if not full_text.strip(): 
        return {}
        
    # If text is safe size, just run it
    SAFE_LIMIT = 900000 # spaCy limit is 1,000,000
    if len(full_text) < SAFE_LIMIT:
        return ea.zscores(full_text)
        
    # If too large, split into chunks and average the results
    print(f"   [INFO] Text size {len(full_text)} exceeds limit. Processing in chunks...")
    chunks = []
    current_chunk = []
    current_len = 0
    
    # Split by words to avoid breaking tokens
    words = full_text.split(" ")
    for word in words:
        if current_len + len(word) + 1 > SAFE_LIMIT:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(word)
        current_len += len(word) + 1
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    # Calculate scores for each chunk
    results = []
    for chunk in chunks:
        if chunk.strip():
            results.append(ea.zscores(chunk))
            
    # Average the Z-scores across chunks
    if not results: return {}
    
    avg_scores = {}
    emotions = results[0].keys()
    for emotion in emotions:
        total = sum(r.get(emotion, 0.0) for r in results)
        avg_scores[emotion] = total / len(results)
        
    return avg_scores

# --- PLOTTING FUNCTIONS ---

def _plot_radar(emotions, values, label, color, output_dir: Path, filename: str, show_plots: bool):
    """
    Plot a 'Flower Petal' style radar chart.
    """
    if not emotions or not values: return

    # Preferred Plutchik order
    preferred = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    order = [e for e in preferred if e in emotions] + [e for e in emotions if e not in preferred]
    
    emo_to_val = {e: v for e, v in zip(emotions, values)}
    values_ordered = [emo_to_val.get(e, 0.0) for e in order]
    abs_vals = [abs(v) for v in values_ordered]

    num_vars = len(order)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    width = 2 * np.pi / num_vars * 0.85 

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    THRESHOLD_Z = 1.96

    # Grey Significance Disk
    ax.fill_between(np.linspace(0, 2 * np.pi, 100), 0, THRESHOLD_Z, color="#E0E0E0", alpha=0.6, zorder=0)
    ax.plot(np.linspace(0, 2 * np.pi, 100), [THRESHOLD_Z] * 100, color="#999999", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(np.pi/8, THRESHOLD_Z + 0.2, f"|z|={THRESHOLD_Z}", color="#888888", fontsize=9)

    # Petals
    for ang, emo, v_abs, v_raw in zip(angles, order, abs_vals, values_ordered):
        significant = v_abs >= THRESHOLD_Z
        col = color if significant else "#B0B0B0"
        ax.bar([ang], [v_abs], width=width, color=col, alpha=0.75, linewidth=1.5, zorder=3)
        
        # Labels
        label_r = max(v_abs + 0.4, THRESHOLD_Z + 0.5) if significant else v_abs + 0.3
        font_weight = "bold" if significant else "normal"
        ax.text(ang, label_r, f"{v_raw:+.2f}", ha="center", va="center", fontsize=10, 
                color=col, fontweight=font_weight, zorder=4)

    ax.set_xticks(angles)
    ax.set_xticklabels(order, fontsize=11, fontweight="bold")
    ax.tick_params(axis='x', pad=15)
    ax.set_yticklabels([])
    
    r_max = max(abs_vals) if abs_vals else 0.0
    r_lim = max(r_max * 1.15, THRESHOLD_Z * 1.2)
    ax.set_ylim(0, r_lim)
    ax.grid(False)
    ax.spines['polar'].set_visible(False)
    
    # ax.set_title(f"{label}\n(Petal=|Z|; Grey Area=Insignificant)", pad=30, fontsize=13, fontweight="bold")
    _save_or_show(fig, output_dir, filename, show_plots)


def plot_global_intensity_comparison(real_scores_dict, syn_scores_dict, output_dir, show_plots):
    emotions = sorted(list(real_scores_dict.keys()))
    real_means = [np.mean(real_scores_dict[e]) if real_scores_dict[e] else 0.0 for e in emotions]
    syn_means = [np.mean(syn_scores_dict[e]) if syn_scores_dict[e] else 0.0 for e in emotions]
    
    x = np.arange(len(emotions))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width/2, real_means, width, label='Real', color='#1f77b4', alpha=0.9)
    ax.bar(x + width/2, syn_means, width, label='Synthetic', color='#ff7f0e', alpha=0.9)

    ax.set_ylabel('Average Z-Score Intensity')
    # ax.set_title('Global Emotion Intensity Comparison (All Sessions)', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(emotions, rotation=0, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.axhline(y=1.96, color='gray', linestyle=':', alpha=0.7, label='Sig. Threshold (1.96)')

    _save_or_show(fig, output_dir, "global_intensity_bar_comparison.png", show_plots)


def plot_eight_emotion_timelines(real_turns_dict, syn_turns_dict, output_dir, show_plots):
    """
    Plots the fluctuation of 8 emotions over turns.
    LIMIT: Shows only the first 15 turns.
    """
    emotions = sorted(list(real_turns_dict.keys()))
    
    # --- HARD LIMIT: SHOW ONLY FIRST 15 TURNS ---
    MAX_SHOW_TURNS = 15
    
    for emotion in emotions:
        real_vals = real_turns_dict[emotion]
        syn_vals = syn_turns_dict[emotion]
        
        if not real_vals or not syn_vals: continue
        
        # Calculate max turns present in data just for matrix sizing
        max_turns_data = max(max(len(x) for x in real_vals), max(len(x) for x in syn_vals))
        if max_turns_data == 0: continue

        # Align data into matrices
        real_arr = np.full((len(real_vals), max_turns_data), np.nan)
        syn_arr = np.full((len(syn_vals), max_turns_data), np.nan)
        
        for i, row in enumerate(real_vals): real_arr[i, :len(row)] = row
        for i, row in enumerate(syn_vals): syn_arr[i, :len(row)] = row
            
        with warnings_suppressed():
            real_mean = np.nanmean(real_arr, axis=0)
            syn_mean = np.nanmean(syn_arr, axis=0)
        
        # --- CUT OFF DATA AFTER MAX_SHOW_TURNS ---
        limit = min(MAX_SHOW_TURNS, len(real_mean))
        real_mean = real_mean[:limit]
        syn_mean = syn_mean[:limit]
        turns = range(1, limit + 1)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(turns, real_mean, marker='o', markersize=4, linewidth=2, label='Real (Avg)', color='#1f77b4')
        ax.plot(turns, syn_mean, marker='s', markersize=4, linewidth=2, linestyle='--', label='Synthetic (Avg)', color='#ff7f0e')
        
        # ax.set_title(f"Fluctuation of '{emotion.upper()}' (First {limit} Turns)", fontweight='bold')
        ax.set_xlabel("Conversation Turn")
        ax.set_ylabel("Z-Score Intensity")
        
        # Set integer ticks for clarity
        ax.set_xticks(np.arange(1, limit + 1, step=1))
        
        ax.axhline(y=1.96, color='gray', linestyle=':', alpha=0.5, label='Significance (1.96)')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.4)
        
        _save_or_show(fig, output_dir, f"fluctuation_{emotion}_first15.png", show_plots)

# --- LOADING & PROCESSING ---

def load_synthetic_conversation(idx: int, data_dir: Path):
    try:
        file_path = data_dir / f"session_{idx}.json"
        if not file_path.exists(): file_path = data_dir / "turn" / f"session_{idx}.json"
        if not file_path.exists(): return None

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f).get("messages", [])
    except Exception: return None

def compare_conversations(idx: int, dataset: str, patient_sim: str, data_dir: Path, show_plots: bool = False):
    """Single session comparison (kept for backward compatibility, logic unchanged)."""
    print(f"Processing Single Index: {idx}")
    syn_msgs = load_synthetic_conversation(idx, data_dir)
    if not syn_msgs:
        print(f"Synthetic data not found for index {idx}")
        return
    try:
        real_df = load_eeyore_dataset(dataset_type=dataset, indices=[idx])
        if real_df.empty: return
        real_msgs = real_df.iloc[0].get("messages", [])
    except Exception as e:
        print(f"Error loading real data: {e}")
        return

    print(f"\nReal Turns: {len(real_msgs)} | Synthetic Turns: {len(syn_msgs)}")
    print("Use --all-sessions to generate the detailed plots.")


def compare_all_sessions(dataset: str, patient_sim: str, data_dir: Path, show_plots: bool = False, max_sessions: int | None = None):
    print("\n" + "="*60)
    print(f"AGGREGATE ANALYSIS: {data_dir.name}")
    print("="*60)
    
    print("Initializing EmoAtlas...")
    ea = EmoScores(language='english')
    
    # Find Files
    syn_files = sorted(data_dir.glob("session_*.json"))
    if not syn_files:
        print(f"[ERROR] No synthetic session_*.json files found in {data_dir}")
        return

    syn_indices_map = {}
    for f in syn_files:
        try: syn_indices_map[int(f.stem.split("_")[1])] = f
        except: continue
    
    # Load Real Data
    syn_indices = sorted(syn_indices_map.keys())
    print("[INFO] Loading real datasets...")
    real_df = load_eeyore_dataset(dataset_type=dataset, indices=syn_indices)
    intersect_indices = sorted(list(set(real_df.index) & set(syn_indices)))
    
    if max_sessions and max_sessions < len(intersect_indices):
        intersect_indices = intersect_indices[:max_sessions]
    
    print(f"[INFO] Processing {len(intersect_indices)} overlapping sessions.")

    # Data Collection
    real_global_agg = defaultdict(list) 
    syn_global_agg = defaultdict(list)
    real_turn_agg = defaultdict(list)
    syn_turn_agg = defaultdict(list)

    # --- BUCKETS FOR INTERVAL FLOWERS (3-Turn Chunks) ---
    MAX_FLOWER_TURNS = 15
    CHUNK_SIZE = 3
    
    # Structure: buckets[chunk_index] = [all_text_from_that_chunk_across_all_sessions]
    real_buckets = defaultdict(list)
    syn_buckets = defaultdict(list)

    processed_count = 0
    
    for i, idx in enumerate(intersect_indices):
        if (i+1) % 10 == 0: print(f"Processing {i+1}/{len(intersect_indices)}...")
        
        # Load Texts
        with open(syn_indices_map[idx], 'r', encoding='utf-8') as f:
            syn_text = _extract_patient_messages(json.load(f).get("messages", [])) 
        real_text = _extract_patient_messages(real_df.loc[idx].get("messages", []))
        
        if not syn_text or not real_text: continue

        # --- FILL INTERVAL BUCKETS ---
        def fill_buckets(msgs, bucket_dict):
            for turn_idx, msg in enumerate(msgs):
                # Stop if we pass turn 15
                if turn_idx >= MAX_FLOWER_TURNS: break 
                
                # Turn 0,1,2 -> chunk 0; Turn 3,4,5 -> chunk 1
                chunk_id = turn_idx // CHUNK_SIZE 
                bucket_dict[chunk_id].append(msg)

        fill_buckets(real_text, real_buckets)
        fill_buckets(syn_text, syn_buckets)

        # --- EMOTION SCORES (For Line Charts) ---
        df_real = prepare_emotion_data(real_text, ea)
        df_syn = prepare_emotion_data(syn_text, ea)
        
        # Truncate to min length for simple turn-alignment
        min_len = min(len(df_real), len(df_syn))
        df_real_trunc = df_real.iloc[:min_len]
        df_syn_trunc = df_syn.iloc[:min_len]
        
        emotions = [c for c in df_real.columns if c != 'turn_id']
        
        for emo in emotions:
            real_turn_agg[emo].append(df_real_trunc[emo].tolist())
            syn_turn_agg[emo].append(df_syn_trunc[emo].tolist())
            real_global_agg[emo].append(df_real[emo].mean())
            syn_global_agg[emo].append(df_syn[emo].mean())
            
        processed_count += 1

    # Generate Plots
    output_base = Path(f"output/{patient_sim}/{dataset}")
    output_dir = output_base / "aggregate_analysis_15turns"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[OUTPUT] Saving plots to: {output_dir}")
    
    # Plot A: Interval Flowers (Every 3 Turns)
    print("\nGenerating Interval Flowers (3-turn chunks)...")
    emotions_list = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    colors = {"Real": "#1f77b4", "Synthetic": "#ff7f0e"}
    
    num_chunks = MAX_FLOWER_TURNS // CHUNK_SIZE # 5 chunks
    
    for chunk_id in range(num_chunks):
        start_t = chunk_id * CHUNK_SIZE + 1
        end_t = (chunk_id + 1) * CHUNK_SIZE
        label_suffix = f"Turns_{start_t}-{end_t}"
        
        # Real
        real_scores = get_scores(real_buckets[chunk_id], ea)
        if real_scores:
            vals = [real_scores.get(e, 0.0) for e in emotions_list]
            _plot_radar(emotions_list, vals, f"Real Patient ({label_suffix})", colors["Real"], 
                        output_dir, f"flower_real_{label_suffix}.png", show_plots)
            
        # Synthetic
        syn_scores = get_scores(syn_buckets[chunk_id], ea)
        if syn_scores:
            vals = [syn_scores.get(e, 0.0) for e in emotions_list]
            _plot_radar(emotions_list, vals, f"Synthetic Patient ({label_suffix})", colors["Synthetic"], 
                        output_dir, f"flower_syn_{label_suffix}.png", show_plots)

    # Plot B: Global Bar Comparison
    plot_global_intensity_comparison(real_global_agg, syn_global_agg, output_dir, show_plots)
    
    # Plot C: 8 Separate Fluctuation Charts (Capped at 15 Turns)
    plot_eight_emotion_timelines(real_turn_agg, syn_turn_agg, output_dir, show_plots)

    print("[DONE] Analysis Complete.")

def main():
    parser = ap.ArgumentParser(description='Compare real and synthetic patient conversations (Text & Emotions).')
    parser.add_argument('idx', type=int, nargs='?', default=None, help='Index of conversation.')
    parser.add_argument("--dataset", type=str, default="esc", help="Dataset name.")
    parser.add_argument("--data-dir", type=str, default="/work/hdd/bfjp/data/synthetic/test/eeyore/hosted_vllm_openai_gpt-oss-120b/")
    parser.add_argument("--psi", type=str, default="eeyore")
    parser.add_argument("--show-plots", action="store_true")
    parser.add_argument("--all-sessions", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    args = parser.parse_args()
    
    if args.idx is None and not args.all_sessions:
        parser.error("You must provide an 'idx' OR use the '--all-sessions' flag.")

    dataset_name = args.dataset.strip().lower()
    psi_name = args.psi.strip().lower()
    data_dir_raw = Path(args.data_dir.strip())
    
    data_dir_final = _resolve_synthetic_dir(data_dir_raw, psi_name, dataset_name)
    if not data_dir_final.exists():
        print(f"[ERROR] Path not found: {data_dir_final}")
        exit(1)
        
    print(f"[INFO] Working with synthetic data directory: {data_dir_final}")

    if args.all_sessions:
        compare_all_sessions(dataset_name, psi_name, data_dir_final, 
                             show_plots=args.show_plots, max_sessions=args.max_sessions)
    else:
        compare_conversations(args.idx, dataset_name, psi_name, data_dir_final, 
                              show_plots=args.show_plots)

if __name__ == "__main__":
    main()