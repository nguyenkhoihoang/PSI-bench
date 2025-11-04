"""Data loading utilities for emotional support conversation datasets."""

import json
import os
from pathlib import Path
import pandas as pd
from datasets import load_dataset


def load_esc_data():
    """Load ESC samples from the Eeyore dataset."""
    data = load_dataset("liusiyang/eeyore_profile", split='train', token=os.getenv("HF_TOKEN"))
    df = data.to_pandas()
    return df[df['source'] == 'ESC']


def load_esc_data_with_indices(original_indices: list = None):
    """Load ESC samples with specific original dataset indices."""
    data = load_dataset("liusiyang/eeyore_profile", split='train', token=os.getenv("HF_TOKEN"))
    df = data.to_pandas()
    
    if original_indices:
        # Directly get rows with matching indices
        matched_df = df.loc[original_indices].copy()
        esc_df = matched_df[matched_df['source'] == 'ESC'].copy()
        print(f"[INFO] Found {len(esc_df)} matching ESC samples out of {len(original_indices)} requested")
    else:
        # Just filter to ESC samples if no indices specified
        esc_df = df[df['source'] == 'ESC'].copy()
    
    return esc_df


def load_esc_original_data():
    """Load original ESConv data to get situations by id_source."""
    try:
        # Get project root directory (2 levels up from this file)
        base_dir = Path(__file__).parent.parent.parent
        esc_path = base_dir / 'data' / 'real' / 'ESConv.json'
        with open(esc_path, 'r') as f:
            data = json.load(f,)
            print(f"[INFO] Loaded original ESConv data from {esc_path} with {len(data)} samples")
            return data
    except FileNotFoundError:
        print("[ERROR] ESConv.json not found at", esc_path)
        print("[ERROR] Please run: wget https://raw.githubusercontent.com/thu-coai/Emotional-Support-Conversation/main/ESConv.json")
        return {}


def load_synthetic_data(data_dir: str):
    """Load synthetic session data from directory."""
    data_dir = Path(data_dir)
    sessions = []
    
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return pd.DataFrame()
    
    # Load all session JSON files
    for session_file in sorted(data_dir.glob('session_*.json')):
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
                sessions.append({
                    'messages': session_data.get('messages', []),
                    'profile': session_data.get('profile', {}),
                    'source': 'SYNTHETIC'
                })
        except Exception as e:
            print(f"[WARNING] Failed to load {session_file}: {e}")
            continue
    
    if not sessions:
        print(f"[WARNING] No session files found in {data_dir}")
        return pd.DataFrame()
    
    return pd.DataFrame(sessions)


def get_synthetic_indices(data_dir: str):
    """Get the indices from synthetic session files."""
    data_dir = Path(data_dir)
    indices = []
    
    for session_file in sorted(data_dir.glob('session_*.json')):
        # Extract index from session_XXX.json
        idx = int(session_file.stem.split('_')[1])
        indices.append(idx)
    
    return sorted(indices)