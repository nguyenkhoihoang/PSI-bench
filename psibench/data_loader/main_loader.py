import os
from typing import Literal, Optional
from datasets import load_dataset
from psibench.data_loader.utils import merge_consecutive_messages
import pandas as pd
from pathlib import Path
import json

# Load ESC, HOPE, AnnoMI datasets from Eeyore on HF
def load_eeyore_from_hf(dataset_type: Literal["ESC","HOPE", "AnnoMI"], indices: list = None):
    """Load specific datasets from the Eeyore dataset."""
    data = load_dataset("liusiyang/eeyore_profile", split='train', token=os.getenv("HF_TOKEN"))
    df = data.to_pandas()
    df['messages'] = df['messages'].apply(merge_consecutive_messages)
    if indices:
        # Directly get rows with matching indices
        matched_df = df.loc[indices].copy()
        return matched_df[matched_df['source'] == dataset_type].copy()
    
    return df[df['source'] == dataset_type]

def load_all_real(indices: list = None):
    """Load all real datasets from the Eeyore dataset (ESC, HOPE, AnnoMI).
    
    Args:
        indices: Optional list of specific indices to load
        
    Returns:
        DataFrame containing all real conversation data
    """
    data = load_dataset("liusiyang/eeyore_profile", split='train', token=os.getenv("HF_TOKEN"))
    df = data.to_pandas()
    df['messages'] = df['messages'].apply(merge_consecutive_messages)
    
    if indices:
        # Directly get rows with matching indices
        matched_df = df.loc[indices].copy()
        return matched_df[matched_df['source'].isin(["ESC", "HOPE", "AnnoMI"])].copy()
    
    return df[df['source'].isin(["ESC", "HOPE", "AnnoMI"])]

def load_eeyore_dataset(dataset_type: str, indices: list = None):
    """Load dataset based on type and optional indices.
    
    Args:
        dataset_type: Type of dataset to load (e.g., "esc" or "all")
        indices: Optional list of specific indices to load
        
    Returns:
        DataFrame containing the loaded dataset
        
    Raises:
        ValueError: If dataset_type is not supported

    """
    match dataset_type:
        case "esc":
            return load_eeyore_from_hf(dataset_type="ESC", indices=indices)
        case "hope":
            return load_eeyore_from_hf(dataset_type="HOPE", indices=indices)
        case "annomi":
            return load_eeyore_from_hf(dataset_type="AnnoMI", indices=indices)
        case "all":
            return load_all_real(indices=indices)
        case _:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")


def load_synthetic_data_to_df(data_dir: str):
    """Load synthetic session data from directory 
        Return into a DataFrame."""
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
                sessions.append(session_data)
        except Exception as e:
            print(f"[WARNING] Failed to load {session_file}: {e}")
            continue
    
    if not sessions:
        print(f"[WARNING] No session files found in {data_dir}")
        return pd.DataFrame()
    
    return pd.DataFrame(sessions)


def _safe_json_load(value):
    """Parse JSON strings back to Python objects when possible."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return value
    return value


def load_synthetic_hf_to_df(psi: str, backend_llm: str, split: str = "train", token: Optional[str] = None):
    """Load PSI-bench synthetic data from HuggingFace and filter by simulator and backend.

    Args:
        psi: Name of the PSI simulator (e.g., "patientpsi" or "roleplaydoh").
        backend_llm: Backend LLM identifier (e.g., "hosted_vllm_openai_gpt-oss-120b").
        split: HF split to load (default: "train").
        token: Optional HF token; falls back to HF_TOKEN env var.

    Returns:
        Pandas DataFrame filtered to the requested psi and backend_llm.
    """

    hf_token = token or os.getenv("HF_TOKEN")
    dataset = load_dataset("hknguyen20/psibench-synthetic", split=split, token=hf_token)
    df = dataset.to_pandas()

    # Filter by psi and backend (case-insensitive)
    filtered = df[
        df["psi"].str.lower() == psi.lower()
    ]
    filtered = filtered[
        filtered["backend_llm"].str.lower() == backend_llm.lower()
    ].copy()

    if filtered.empty:
        print(f"[WARNING] No rows found for psi='{psi}' and backend_llm='{backend_llm}'")
        return filtered

    # Convert JSON-like strings back to Python objects for convenience
    for col in ("messages", "profile", "ccd"):
        if col in filtered.columns:
            filtered[col] = filtered[col].apply(_safe_json_load)

    return filtered

