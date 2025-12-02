"""Compare real and synthetic conversations side by side for a given index."""

import json
import os
from pathlib import Path
import argparse as ap  # Check: Renamed to avoid conflict if main calls it

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Ensure emoatlas is installed: pip install emoatlas
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


def get_significant_emotions(text, analyzer):
    """
    Analyzes text and returns a dictionary of emotions with Z-score > 1.
    This function is kept for compatibility with potential other scripts, 
    but the visualization now uses full scores.
    """
    if not text or not analyzer:
        return {}
    
    try:
        # Note: For visualization, we usually want all scores, not just >1.
        # But for the text report listing, keeping >1 filter is fine.
        all_scores = analyzer.zscores(text)
        significant_scores = {k: v for k, v in all_scores.items() if abs(v) > 1.0}
        return significant_scores
    except Exception as e:
        print(f"Error calculating emotions: {e}")
        return {}

def prepare_emotion_data(text_list, ea):
    """Calculates z-scores for each turn and returns a DataFrame."""
    timeline_data = []
    
    for i, text in enumerate(text_list):
        if not text or len(text.strip()) < 2:
            continue
        scores = ea.zscores(text)
        scores['turn_id'] = i + 1
        timeline_data.append(scores)
        
    return pd.DataFrame(timeline_data)


def _add_turn_metrics(df: pd.DataFrame):
    """Augment per-turn z-score DataFrame with variance and diversity."""
    if df.empty:
        return df
    df = df.copy()
    emotion_cols = [c for c in df.columns if c not in ["turn_id"]]

    # Intensity variance across emotions for each turn
    df["intensity_var"] = df[emotion_cols].var(axis=1, ddof=0)

    # Diversity: entropy of absolute scores distribution + count of strong emotions
    abs_scores = df[emotion_cols].abs()
    # Avoid division by zero
    sum_abs = abs_scores.sum(axis=1).replace(0, np.nan)
    prob = abs_scores.div(sum_abs, axis=0).fillna(0)
    
    eps = 1e-9
    # Calculate entropy (handling log(0) with eps)
    entropy = -(prob * np.log(prob + eps)).sum(axis=1)
    # If the sum of abs scores was 0, entropy should be 0
    df["diversity_entropy"] = entropy.where(sum_abs.notna(), 0.0)
    
    df["diversity_count_gt1"] = (abs_scores > 1.0).sum(axis=1)
    return df


def _diversity_from_scores(scores: dict):
    """Compute diversity metrics from a dict of emotion scores."""
    if not scores:
        return 0.0, 0
    vals = np.abs(np.array(list(scores.values()), dtype=float))
    total = vals.sum()
    if total == 0:
        return 0.0, int((vals > 1.0).sum())
    prob = vals / total
    eps = 1e-9
    entropy = float(-(prob * np.log(prob + eps)).sum())
    strong_count = int((vals > 1.0).sum())
    return entropy, strong_count


def _save_or_show(fig, output_dir: Path, filename: str, show_plots: bool):
    """Save figure to output_dir and optionally display."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig_path = output_dir / filename
    # Use a slightly higher DPI for better looking petal plots
    fig.savefig(fig_path, dpi=250, bbox_inches='tight')
    if show_plots:
        plt.show()
    plt.close(fig)
    print(f"[PLOT SAVED] {fig_path}")

# ==============================================================================
# MODIFIED FUNCTION: Flower Petal Style Radar Chart
# ==============================================================================
def _plot_radar(emotions, values, label, color, output_dir: Path, filename: str, show_plots: bool):
    """
    Plot a 'Flower Petal' style radar chart.
    Adds a central grey disk indicating the statistical significance threshold (|z|<1.96).
    Petals extending beyond this disk are considered significantly over-expressed.
    """
    if not emotions or not values:
        print(f"[WARN] Skipping radar plot; empty data for {label}")
        return

    # Preferred Plutchik order for visual consistency
    preferred = ["joy", "trust", "fear", "surprise", "sadness", "disgust", "anger", "anticipation"]
    order = [e for e in preferred if e in emotions] + [e for e in emotions if e not in preferred]
    
    # Map emotions to values
    emo_to_val = {e: v for e, v in zip(emotions, values)}
    values_ordered = [emo_to_val.get(e, 0.0) for e in order]
    # Use absolute values for the length of the petal (radius)
    abs_vals = [abs(v) for v in values_ordered]

    num_vars = len(order)
    # Calculate angles for each petal
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
    # Width of petals (leaving some gap between them)
    width = 2 * np.pi / num_vars * 0.85 

    # Standard Plutchik color palette
    cmap = {
        "joy": "#F7D13D",        # Yellow
        "trust": "#6BAF5E",      # Light Green
        "fear": "#2C9C6E",       # Green
        "surprise": "#6CC6E8",   # Light Blue
        "sadness": "#4A90E2",    # Blue
        "disgust": "#7B5AB0",    # Purple
        "anger": "#E4572E",      # Red
        "anticipation": "#F29E4C", # Orange
    }

    # --- Setup Plot ---
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Define significance threshold (standard Z-score threshold)
    THRESHOLD_Z = 1.96

    # --- Draw Central Grey Disk (Significance Threshold) ---
    # 1. Fill the central area
    ax.fill_between(
        np.linspace(0, 2 * np.pi, 100),
        0,
        THRESHOLD_Z,
        color="#E0E0E0",  # Light grey
        alpha=0.6,
        zorder=0,         # Draw in background
    )
    # 2. Add the boundary line for the disk
    ax.plot(
        np.linspace(0, 2 * np.pi, 100),
        [THRESHOLD_Z] * 100,
        color="#999999",  # Darker grey edge
        linestyle="--",
        linewidth=1.2,
        zorder=1
    )
    # Add label for the threshold circle
    ax.text(np.pi/8, THRESHOLD_Z + 0.2, f"|z|={THRESHOLD_Z}", color="#888888", fontsize=9)

    # --- Draw Petals (Bars) ---
    for ang, emo, v_abs, v_raw in zip(angles, order, abs_vals, values_ordered):
        base_col = cmap.get(emo, color)  # fallback to provided color if emotion not in map
        significant = v_abs >= THRESHOLD_Z
        col = base_col if significant else "#B0B0B0"

        # Draw the petal
        ax.bar(
            [ang],
            [v_abs],
            width=width,
            color=col,
            edgecolor=col,
            alpha=0.75,
            linewidth=1.5,
            zorder=3,
        )

        # Annotate values at petal tips
        label_r = max(v_abs + 0.4, THRESHOLD_Z + 0.5) if significant else v_abs + 0.3
        text_color = col
        font_weight = "bold" if significant else "normal"

        ax.text(
            ang,
            label_r,
            f"{v_raw:+.2f}",
            ha="center",
            va="center",
            fontsize=10,
            color=text_color,
            fontweight=font_weight,
            zorder=4,
        )

    # --- Styling ---
    # Set X-axis labels (Emotion names)
    ax.set_xticks(angles)
    ax.set_xticklabels(order, fontsize=11, fontweight="bold")
    # Move radial labels away from the center to avoid overlapping data
    ax.tick_params(axis='x', pad=15)

    # Hide Y-axis (radial) value labels
    ax.set_yticklabels([])
    
    # Determine plot limits dynamically
    r_max = max(abs_vals) if abs_vals else 0.0
    # Ensure limit is at least slightly bigger than threshold disk
    r_lim = max(r_max * 1.15, THRESHOLD_Z * 1.2)
    ax.set_ylim(0, r_lim)

    # Hide default circular grid lines to emphasize our custom threshold disk
    ax.grid(False)
    # Optional: Add faint spoke lines
    ax.spines['polar'].set_visible(False) # Hide outer circle frame
    for ang in angles:
         ax.plot([ang, ang], [0, r_lim], color='#DDDDDD', lw=1, zorder=0, alpha=0.5)

    # Title
    ax.set_title(
        f"Global Emotion Intensity Profile ({label})\n(Petal length = |Z-score|; Grey Disk = Insignificant Range |z|<1.96)", 
        pad=30, fontsize=13, fontweight="bold"
    )

    _save_or_show(fig, output_dir, filename, show_plots)
# ==============================================================================


def visualize_comparison(real_msgs_list, syn_msgs_list, ea, output_dir: Path, show_plots: bool = False):
    """Generate charts focusing on intensity variance and emotion diversity (patient only)."""
    if not real_msgs_list or not syn_msgs_list:
        print("[VISUALIZATION SKIPPED] Not enough messages to visualize.")
        return

    df_real = prepare_emotion_data(real_msgs_list, ea)
    df_syn = prepare_emotion_data(syn_msgs_list, ea)
    
    if df_real.empty or df_syn.empty:
        print("[VISUALIZATION SKIPPED] Not enough valid text turns for timeline.")
        return

    df_real = _add_turn_metrics(df_real)
    df_syn = _add_turn_metrics(df_syn)

    # Align to shortest length for turn-by-turn comparison
    min_turns = min(len(df_real), len(df_syn))
    if min_turns == 0:
        print("[VISUALIZATION SKIPPED] No overlapping turns.")
        return
    df_real_turns = df_real.head(min_turns).copy()
    df_syn_turns = df_syn.head(min_turns).copy()
    # Reset turn_id for aligned plots
    df_real_turns["turn_id"] = range(1, min_turns + 1)
    df_syn_turns["turn_id"] = range(1, min_turns + 1)

    # --- 1) Emotional Intensity Variance (Line Plot) ---
    fig_var_line, ax_var_line = plt.subplots(figsize=(12, 6))
    ax_var_line.plot(df_real_turns["turn_id"], df_real_turns["intensity_var"], marker='o', label='Real', color='#1f77b4', linewidth=2)
    ax_var_line.plot(df_syn_turns["turn_id"], df_syn_turns["intensity_var"], marker='s', label='Synthetic', color='#ff7f0e', linestyle='--', linewidth=2)
    ax_var_line.set_xlabel('Conversation Turn', fontsize=12)
    ax_var_line.set_ylabel('Variance of Emotion Z-Scores', fontsize=12)
    ax_var_line.set_title('Emotional Intensity Variance per Turn (Patient Role)', fontsize=14, fontweight='bold')
    ax_var_line.grid(True, linestyle='--', alpha=0.6)
    ax_var_line.legend(fontsize=11)
    _save_or_show(fig_var_line, output_dir, "intensity_variance_turns.png", show_plots)

    # --- 2) Global Emotion Profile (Flower Petal Radar Plots) ---
    # Calculate scores for the entire concatenated text per role
    real_full_text = " ".join(real_msgs_list)
    syn_full_text = " ".join(syn_msgs_list)
    
    # Use try-except block in case text is empty or analysis fails
    try:
        real_full_scores = ea.zscores(real_full_text) if real_full_text.strip() else {}
        syn_full_scores = ea.zscores(syn_full_text) if syn_full_text.strip() else {}
    except Exception as e:
        print(f"[WARN] Global score calculation failed: {e}")
        real_full_scores = {}
        syn_full_scores = {}

    # Get union of all emotions present
    emotions = sorted(set(real_full_scores.keys()) | set(syn_full_scores.keys()))
    
    real_vals = [real_full_scores.get(e, 0.0) for e in emotions]
    syn_vals = [syn_full_scores.get(e, 0.0) for e in emotions]

    # Generate separate radar plots
    _plot_radar(emotions, real_vals, "Real Patient", '#1f77b4', output_dir, "emotion_profile_global_real.png", show_plots)
    _plot_radar(emotions, syn_vals, "Synthetic Patient", '#ff7f0e', output_dir, "emotion_profile_global_synthetic.png", show_plots)

    # --- 3) Emotion Diversity (Line Plot) ---
    fig_div_line, ax_div_line = plt.subplots(figsize=(12, 6))
    ax_div_line.plot(df_real_turns["turn_id"], df_real_turns["diversity_entropy"], marker='o', label='Real entropy', color='#1f77b4', linewidth=2)
    ax_div_line.plot(df_syn_turns["turn_id"], df_syn_turns["diversity_entropy"], marker='s', label='Synthetic entropy', color='#ff7f0e', linestyle='--', linewidth=2)
    ax_div_line.set_xlabel('Conversation Turn', fontsize=12)
    ax_div_line.set_ylabel('Entropy (Higher = More Diverse/Complex)', fontsize=12)
    ax_div_line.set_title('Emotion Diversity per Turn (Patient Role)', fontsize=14, fontweight='bold')
    ax_div_line.grid(True, linestyle='--', alpha=0.6)
    ax_div_line.legend(fontsize=11)
    _save_or_show(fig_div_line, output_dir, "emotion_diversity_turns.png", show_plots)

    # --- 4) Global Emotion Diversity (Bar Chart) ---
    real_entropy, real_count = _diversity_from_scores(real_full_scores)
    syn_entropy, syn_count = _diversity_from_scores(syn_full_scores)
    
    fig_div_bar, ax_div_bar = plt.subplots(figsize=(8, 6))
    bars = ax_div_bar.bar(['Real', 'Synthetic'], [real_entropy, syn_entropy], color=['#1f77b4', '#ff7f0e'], alpha=0.85, width=0.6)
    ax_div_bar.set_ylabel('Entropy (Higher = More Diverse)', fontsize=12)
    ax_div_bar.set_title('Global Emotion Diversity (Patient, Full Session)', fontsize=14, fontweight='bold')
    
    # Add text annotations onto the bars
    for i, (rect, val, count) in enumerate(zip(bars, [real_entropy, syn_entropy], [real_count, syn_count])):
        height = rect.get_height()
        ax_div_bar.text(rect.get_x() + rect.get_width()/2., height + 0.05,
                        f"Entropy: {val:.2f}\nSig. Emotions (>1.0): {count}",
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

    ax_div_bar.set_ylim(0, max(real_entropy, syn_entropy) * 1.3 if max(real_entropy, syn_entropy) > 0 else 1)
    ax_div_bar.grid(axis='y', linestyle='--', alpha=0.6)
    _save_or_show(fig_div_bar, output_dir, "emotion_diversity_global.png", show_plots)


def _resolve_synthetic_dir(base_dir: Path, psi: str, dataset: str) -> Path:
    """Resolve synthetic data directory path."""
    base_dir = Path(base_dir)
    # Check direct path
    if any(base_dir.glob("session_*.json")):
        return base_dir
    # Check dataset subdirectory
    dataset_dir = base_dir / dataset
    if any(dataset_dir.glob("session_*.json")):
        return dataset_dir
    # Check psi/dataset subdirectory
    psi_dataset_dir = base_dir / psi / dataset
    return psi_dataset_dir


def load_synthetic_conversation(idx: int, data_dir: Path, analyzer=None, turn: bool = False):
    """Load synthetic conversation JSON."""
    print(f"Loading synthetic conversation (Index {idx})...")
    try:
        file_path = (data_dir / "turn" / f"session_{idx}.json") if turn else (data_dir / f"session_{idx}.json")
        print(f"Reading: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            messages = data.get("messages", [])
            
            # Extract assistant (patient) messages for analysis check
            assistant_msgs = [
                msg.get("content", "") 
                for msg in messages 
                if msg.get("role", "").lower() == "assistant"
            ]
            full_assistant_text = " ".join(assistant_msgs)
            
            # Get significant emotions for report listing (using the original function for compatibility)
            emotions_for_report = get_significant_emotions(full_assistant_text, analyzer)

            return {
                "messages": messages,
                "situation": data.get("profile", {}).get("situation of the client", "N/A"),
                "emotions_report": emotions_for_report
            }
    except FileNotFoundError:
        print(f"[ERROR] Synthetic file not found for index {idx} at path: {file_path}")
        return None
    except json.JSONDecodeError:
         print(f"[ERROR] Invalid JSON format in file: {file_path}")
         return None


def format_messages(messages):
    """Format conversation messages for table view."""
    formatted = []
    for msg in messages:
        role = msg.get("role", "UNKNOWN").upper()
        # Clean up whitespace
        content = " ".join(msg.get("content", "").split())
        formatted.append((role, content))
    return formatted


def _extract_patient_messages(messages):
    """Return patient-side messages (role==assistant)."""
    return [
        msg.get("content", "")
        for msg in messages
        if msg.get("role", "").lower() == "assistant"
    ]

def get_real_conversation(idx: int, dataset: str, analyzer=None):
    """Load real conversation data using psibench."""
    try:
        # Load specified index
        eeyore_df = load_eeyore_dataset(dataset_type=dataset, indices=[idx])
        if eeyore_df.empty:
             print(f"[ERROR] Index {idx} not found in {dataset} dataset.")
             return None
             
        row = eeyore_df.iloc[0]
        id_source = row.get("id_source")
        messages = row.get("messages", [])
        
        # Try to fetch situation data if available (ESC dataset specific)
        situation = "N/A"
        if dataset == "esc" and id_source is not None:
            try:
                esc_original = load_esc_original_data()
                id_src_int = int(id_source)
                if 0 <= id_src_int < len(esc_original):
                    situation = esc_original[id_src_int].get("situation", "N/A")
            except Exception as e_sit:
                print(f"[WARN] Could not load ESC situation data: {e_sit}")

        # Extract assistant messages for report check
        assistant_msgs = [
            msg.get("content", "") 
            for msg in messages 
            if msg.get("role", "").lower() == "assistant"
        ]
        full_assistant_text = " ".join(assistant_msgs)
        
        # Get significant emotions for report listing
        emotions_for_report = get_significant_emotions(full_assistant_text, analyzer)

        return {
            'messages': messages,
            'situation': situation,
            'emotions_report': emotions_for_report
        }
    except Exception as e:
        print(f"[ERROR] Failed to get real conversation at index {idx}: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_conversations(idx: int, dataset: str, patient_sim: str, data_dir: Path, show_plots: bool = False):
    """Process and compare a single conversation pair."""
    
    print("\n" + "="*60)
    print(f"PROCESSING CONVERSATION INDEX: {idx}")
    print("="*60)

    print("Initializing EmoAtlas Analyzer...")
    try:
        ea = EmoScores(language='english')
    except Exception as e:
        print(f"[FATAL] Failed to initialize EmoScores: {e}")
        return

    # 1. Load Data
    real_data = get_real_conversation(idx, dataset, analyzer=ea)
    if not real_data: return

    synthetic_data = load_synthetic_conversation(idx, data_dir, analyzer=ea)
    if not synthetic_data: return
        
    # 2. Print Text Report
    print("\n" + "-"*30)
    print(f"EMOTION REPORT (Patient Role) - Index {idx}")
    print("-"*30)
    print(f"REAL Patient Significant Emotions (>1.0):")
    print(json.dumps(real_data['emotions_report'], indent=2) if real_data['emotions_report'] else "  None detected")
    print("-" * 20)
    print(f"SYNTHETIC Patient Significant Emotions (>1.0):")
    print(json.dumps(synthetic_data['emotions_report'], indent=2) if synthetic_data['emotions_report'] else "  None detected")
    print("-" * 30 + "\n")

    # 3. Generate Comparison Table
    real_messages_fmt = format_messages(real_data['messages'])
    synthetic_messages_fmt = format_messages(synthetic_data['messages'])
    
    table_data = [["Situation Header", f"REAL: {real_data['situation']}", f"SYNTHETIC: {synthetic_data['situation']}"]]
    table_data.append(["-"*10, "-"*80, "-"*80]) # Separator

    max_turns = max(len(real_messages_fmt), len(synthetic_messages_fmt))
    
    for i in range(max_turns):
        real_msg = real_messages_fmt[i] if i < len(real_messages_fmt) else ("", "")
        syn_msg = synthetic_messages_fmt[i] if i < len(synthetic_messages_fmt) else ("", "")

        real_txt = f"[{real_msg[0]}] {real_msg[1]}" if real_msg[0] else ""
        syn_txt = f"[{syn_msg[0]}] {syn_msg[1]}" if syn_msg[0] else ""

        table_data.append([f"Turn {i+1}", real_txt, syn_txt])

    comparison_table = tabulate(
        table_data,
        headers=["#", "Real Conversation", "Synthetic Conversation"],
        tablefmt="grid",
        maxcolwidths=[12, 90, 90],
        numalign="center",
    )

    # 4. Save Table
    output_base = Path(f"output/{patient_sim}/{dataset}")
    output_dir = output_base / f"idx_{idx}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"conversation_comparison_{idx}.txt"
    with open(output_path, "w", encoding='utf-8') as f:
        f.write(comparison_table)
    print(f"[OUTPUT] Comparison table saved to: {output_path}")

    # 5. Generate Visualizations
    print("\nGenerating Visualizations...")
    real_patient_msgs = [msg[1] for msg in real_messages_fmt if msg[0] == 'ASSISTANT']
    syn_patient_msgs = [msg[1] for msg in synthetic_messages_fmt if msg[0] == 'ASSISTANT']
    
    visualize_comparison(real_patient_msgs, syn_patient_msgs, ea, output_dir=output_dir, show_plots=show_plots)
    print(f"[DONE] Finished processing index {idx}.")


def _aggregate_turn_metrics(df_real: pd.DataFrame, df_syn: pd.DataFrame, buckets_real: list, buckets_syn: list, key: str):
    """Append per-turn metrics into buckets aligned to min length."""
    min_len = min(len(df_real), len(df_syn))
    for t in range(min_len):
        while len(buckets_real) <= t:
            buckets_real.append([])
            buckets_syn.append([])
        buckets_real[t].append(df_real.iloc[t][key])
        buckets_syn[t].append(df_syn.iloc[t][key])


def compare_all_sessions(dataset: str, patient_sim: str, data_dir: Path, show_plots: bool = False, max_sessions: int | None = None):
    """
    Aggregate comparison across all available conversations in synthetic directory.
    Produces averaged per-turn lines and averaged global flower petal plots.
    """
    print("\n" + "="*60)
    print(f"AGGREGATE ANALYSIS: All Sessions in {data_dir.name}")
    print("="*60)
    print("Initializing EmoAtlas (all sessions)...")
    ea = EmoScores(language='english')

    # Find synthetic files
    session_files = sorted(data_dir.glob("session_*.json"))
    if not session_files:
        print(f"[ERROR] No synthetic session_*.json files found in {data_dir}")
        return

    syn_indices_map = {}
    for f in session_files:
        try:
            # Assumes format session_123.json
            idx = int(f.stem.split("_")[1])
            syn_indices_map[idx] = f
        except (IndexError, ValueError):
            print(f"[WARN] Skipping file with unexpected name format: {f.name}")
            continue

    if not syn_indices_map:
        print("[ERROR] No valid session indices parsed from filenames.")
        return
    
    syn_indices = sorted(syn_indices_map.keys())
    print(f"[INFO] Found {len(syn_indices)} synthetic sessions.")

    # Load corresponding real data
    print("[INFO] Loading corresponding real datasets...")
    # Load only intersecting indices to avoid missing pairs and save memory
    real_df = load_eeyore_dataset(dataset_type=dataset, indices=syn_indices)
    if real_df.empty:
        print(f"[ERROR] Failed to load any real conversations for dataset '{dataset}' with the specified indices.")
        return

    real_indices = set(real_df.index.tolist())
    syn_indices_set = set(syn_indices)
    intersect_indices = sorted(list(real_indices & syn_indices_set))
    
    if not intersect_indices:
        print("[ERROR] No overlapping indices found between real and synthetic data.")
        return

    print(f"[INFO] Found {len(intersect_indices)} overlapping sessions for comparison.")
    
    # Apply max_sessions limit if provided
    if max_sessions and max_sessions < len(intersect_indices):
        print(f"[INFO] Limiting analysis to first {max_sessions} sessions as requested.")
        intersect_indices = intersect_indices[:max_sessions]

    # --- Aggregation Containers ---
    # Buckets for per-turn averaging: buckets[turn_index] = [val_session1, val_session2, ...]
    real_var_turns, syn_var_turns = [], []
    real_div_turns, syn_div_turns = [], []

    # Global emotion aggregation
    from collections import defaultdict
    real_global_scores_acc = defaultdict(list)
    syn_global_scores_acc = defaultdict(list)
    real_global_div_acc = []
    syn_global_div_acc = []

    processed_count = 0
    print("\nStarting aggregate processing...")
    for i, idx in enumerate(intersect_indices):
        if (i+1) % 10 == 0 or (i+1) == len(intersect_indices):
            print(f"Processing session {i+1}/{len(intersect_indices)} (Index {idx})...")

        # 1. Get Text Data
        # Synthetic
        syn_file_path = syn_indices_map[idx]
        try:
            with open(syn_file_path, 'r', encoding='utf-8') as f:
                syn_json = json.load(f)
                patient_msgs_syn = _extract_patient_messages(syn_json.get("messages", []))
        except Exception as e:
             print(f"[WARN] Failed to read synthetic index {idx}: {e}")
             continue
             
        # Real
        real_row = real_df.loc[idx]
        patient_msgs_real = _extract_patient_messages(real_row.get("messages", []))

        if not patient_msgs_real or not patient_msgs_syn:
             # Skip if one side has no patient messages
            continue

        # 2. Process Emotions per Turn
        df_real = prepare_emotion_data(patient_msgs_real, ea)
        df_syn = prepare_emotion_data(patient_msgs_syn, ea)
        if df_real.empty or df_syn.empty:
            continue
            
        df_real = _add_turn_metrics(df_real)
        df_syn = _add_turn_metrics(df_syn)

        # Aggregate turn metrics
        _aggregate_turn_metrics(df_real, df_syn, real_var_turns, syn_var_turns, "intensity_var")
        _aggregate_turn_metrics(df_real, df_syn, real_div_turns, syn_div_turns, "diversity_entropy")

        # 3. Process Global Emotions
        # Concatenate all patient text for global score
        real_full_text = " ".join(patient_msgs_real)
        syn_full_text = " ".join(patient_msgs_syn)
        
        try:
            real_scores = ea.zscores(real_full_text) if real_full_text.strip() else {}
            syn_scores = ea.zscores(syn_full_text) if syn_full_text.strip() else {}
        except:
            real_scores = {}
            syn_scores = {}

        for e, v in real_scores.items(): real_global_scores_acc[e].append(v)
        for e, v in syn_scores.items(): syn_global_scores_acc[e].append(v)

        # Global diversity metrics per session
        r_ent, r_cnt = _diversity_from_scores(real_scores)
        s_ent, s_cnt = _diversity_from_scores(syn_scores)
        real_global_div_acc.append((r_ent, r_cnt))
        syn_global_div_acc.append((s_ent, s_cnt))

        processed_count += 1

    if processed_count == 0:
        print("[ERROR] No sessions were successfully processed for aggregation.")
        return

    print(f"\nAggregation complete. Successfully processed {processed_count} pairs.")
    
    output_base = Path(f"output/{patient_sim}/{dataset}")
    output_dir = output_base / "aggregate_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTPUT] Saving aggregate plots to: {output_dir}")

    # --- Generate Averaged Plots ---

    # 1. Per-turn Averaged Line Plots
    if real_var_turns and syn_var_turns:
        # Calculate means ignoring NaNs if any (though prepare_data shouldn't produce them)
        real_var_mean = [np.nanmean(v) for v in real_var_turns]
        syn_var_mean = [np.nanmean(v) for v in syn_var_turns]
        turns_axis = list(range(1, len(real_var_mean) + 1))

        fig_var_line, ax_var_line = plt.subplots(figsize=(12, 6))
        ax_var_line.plot(turns_axis, real_var_mean, marker='o', label='Real (Avg)', color='#1f77b4', linewidth=2)
        ax_var_line.plot(turns_axis, syn_var_mean, marker='s', label='Synthetic (Avg)', color='#ff7f0e', linestyle='--', linewidth=2)
        ax_var_line.set_xlabel('Conversation Turn', fontsize=12)
        ax_var_line.set_ylabel('Average Variance of Emotion Z-Scores', fontsize=12)
        ax_var_line.set_title(f'Average Emotional Intensity Variance per Turn (N={processed_count} sessions)', fontsize=14, fontweight='bold')
        ax_var_line.grid(True, linestyle='--', alpha=0.6)
        ax_var_line.legend(fontsize=11)
        _save_or_show(fig_var_line, output_dir, "avg_intensity_variance_turns.png", show_plots)

    if real_div_turns and syn_div_turns:
        real_div_mean = [np.nanmean(v) for v in real_div_turns]
        syn_div_mean = [np.nanmean(v) for v in syn_div_turns]
        turns_axis_div = list(range(1, len(real_div_mean) + 1))
        
        fig_div_line, ax_div_line = plt.subplots(figsize=(12, 6))
        ax_div_line.plot(turns_axis_div, real_div_mean, marker='o', label='Real Entropy (Avg)', color='#1f77b4', linewidth=2)
        ax_div_line.plot(turns_axis_div, syn_div_mean, marker='s', label='Synthetic Entropy (Avg)', color='#ff7f0e', linestyle='--', linewidth=2)
        ax_div_line.set_xlabel('Conversation Turn', fontsize=12)
        ax_div_line.set_ylabel('Average Entropy', fontsize=12)
        ax_div_line.set_title(f'Average Emotion Diversity per Turn (N={processed_count} sessions)', fontsize=14, fontweight='bold')
        ax_div_line.grid(True, linestyle='--', alpha=0.6)
        ax_div_line.legend(fontsize=11)
        _save_or_show(fig_div_line, output_dir, "avg_emotion_diversity_turns.png", show_plots)

    # 2. Global Averaged Flower Petal Plots
    emotions_all = sorted(set(real_global_scores_acc.keys()) | set(syn_global_scores_acc.keys()))
    if not emotions_all:
        print("[WARN] No global emotion scores available for radar plots.")
    else:
        # Calculate average Z-score for each emotion across all sessions
        # Use np.nanmean to handle cases where an emotion might be missing in some sessions
        real_mean_scores = [np.nanmean(real_global_scores_acc[e]) if real_global_scores_acc[e] else 0.0 for e in emotions_all]
        syn_mean_scores = [np.nanmean(syn_global_scores_acc[e]) if syn_global_scores_acc[e] else 0.0 for e in emotions_all]

        _plot_radar(emotions_all, real_mean_scores, f"Real Patients (Avg of {processed_count})", '#1f77b4', output_dir, "avg_emotion_profile_real.png", show_plots)
        _plot_radar(emotions_all, syn_mean_scores, f"Synthetic Patients (Avg of {processed_count})", '#ff7f0e', output_dir, "avg_emotion_profile_synthetic.png", show_plots)

    # 3. Global Averaged Diversity Bar Chart
    if real_global_div_acc and syn_global_div_acc:
        real_entropy_vals = [v[0] for v in real_global_div_acc]
        syn_entropy_vals = [v[0] for v in syn_global_div_acc]

        real_entropy_mean = np.mean(real_entropy_vals)
        syn_entropy_mean = np.mean(syn_entropy_vals)
        real_entropy_min = np.min(real_entropy_vals)
        syn_entropy_min = np.min(syn_entropy_vals)
        real_entropy_max = np.max(real_entropy_vals)
        syn_entropy_max = np.max(syn_entropy_vals)

        fig_div_bar, ax_div_bar = plt.subplots(figsize=(8, 6))
        x_pos = np.array([0, 1])
        means = np.array([real_entropy_mean, syn_entropy_mean])
        yerr_lower = means - np.array([real_entropy_min, syn_entropy_min])
        yerr_upper = np.array([real_entropy_max, syn_entropy_max]) - means

        ax_div_bar.errorbar(
            [x_pos[0]],
            [means[0]],
            yerr=[[yerr_lower[0]], [yerr_upper[0]]],
            fmt='o',
            capsize=12,
            elinewidth=2,
            markersize=8,
            color='#1f77b4',
            ecolor='#1f77b4',
            label='Real (mean, min-max)',
        )
        ax_div_bar.errorbar(
            [x_pos[1]],
            [means[1]],
            yerr=[[yerr_lower[1]], [yerr_upper[1]]],
            fmt='o',
            capsize=12,
            elinewidth=2,
            markersize=8,
            color='#ff7f0e',
            ecolor='#ff7f0e',
            label='Synthetic (mean, min-max)',
        )

        ax_div_bar.set_xticks(x_pos)
        ax_div_bar.set_xticklabels(['Real', 'Synthetic'], fontsize=11, fontweight='bold')
        ax_div_bar.set_ylabel('Entropy', fontsize=12)
        ax_div_bar.set_title(f'Global Emotion Diversity (mean with min/max, N={processed_count})', fontsize=14, fontweight='bold')

        ymax = max(real_entropy_max, syn_entropy_max, real_entropy_mean, syn_entropy_mean)
        ymin = min(real_entropy_min, syn_entropy_min, 0)
        ax_div_bar.set_ylim(ymin * 1.1 if ymin < 0 else 0, ymax * 1.2 if ymax > 0 else 1)
        ax_div_bar.grid(axis='y', linestyle='--', alpha=0.6)
        ax_div_bar.legend(frameon=False)
        _save_or_show(fig_div_bar, output_dir, "avg_emotion_diversity_global.png", show_plots)

    print(f"\n[DONE] Aggregate analysis finished. Check output directory: {output_dir}")


def main():
    # Use 'ap' alias to avoid conflict if main calls it later
    parser = ap.ArgumentParser(description='Compare real and synthetic patient conversations (Text & Emotions).')
    
    # Arguments
    parser.add_argument('idx', type=int, nargs='?', default=None, 
                        help='Index of the specific conversation to compare (Required unless --all-sessions is used).')
    
    parser.add_argument("--dataset", type=str, default="esc", 
                        help="Dataset name (e.g., 'esc', 'daily_dialog'). Default: 'esc'")
    
    parser.add_argument("--data-dir", type=str, 
                        default="/work/hdd/bfjp/data/synthetic/test/eeyore/hosted_vllm_openai_gpt-oss-120b/",
                        help="Base directory containing synthetic 'session_*.json' files.")
    
    parser.add_argument("--psi", type=str, default="eeyore", 
                        help="Name of the patient simulation model (for output folder naming). Default: 'eeyore'")
    
    parser.add_argument("--show-plots", action="store_true", 
                        help="If set, display plots interactively in addition to saving them to disk.")
    
    parser.add_argument("--all-sessions", action="store_true", 
                        help="Run aggregate analysis on all paired synthetic/real sessions found in the data directory.")
    
    parser.add_argument("--max-sessions", type=int, default=None, 
                        help="For --all-sessions mode: Limit the number of sessions processed (useful for testing).")
    
    args = parser.parse_args()
    
    # Check inputs
    if args.idx is None and not args.all_sessions:
        parser.error("You must provide an 'idx' OR use the '--all-sessions' flag.")

    # Clean inputs
    dataset_name = args.dataset.strip().lower()
    psi_name = args.psi.strip().lower()
    data_dir_raw = Path(args.data_dir.strip())
    
    # Resolve synthetic directory path
    data_dir_final = _resolve_synthetic_dir(data_dir_raw, psi_name, dataset_name)
    if not data_dir_final.exists():
        print(f"[ERROR] Resolved synthetic data directory does not exist: {data_dir_final}")
        exit(1)
        
    print(f"[INFO] Working with synthetic data directory: {data_dir_final}")

    # Run requested mode
    if args.all_sessions:
        compare_all_sessions(dataset_name, psi_name, data_dir_final, 
                             show_plots=args.show_plots, max_sessions=args.max_sessions)
    else:
        compare_conversations(args.idx, dataset_name, psi_name, data_dir_final, 
                              show_plots=args.show_plots)


if __name__ == "__main__":
    main()
