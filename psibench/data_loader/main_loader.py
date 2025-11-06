import os
from typing import Literal
from datasets import load_dataset
from data_loader.esc import load_esc_data, load_esc_data_with_indices

def load_eeyore_from_hf(dataset_type: Literal["ESC","HOPE", "AnnoMI"]):
    """Load ESC samples from the Eeyore dataset."""
    data = load_dataset("liusiyang/eeyore_profile", split='train', token=os.getenv("HF_TOKEN"))
    df = data.to_pandas()
    return df[df['source'] == dataset_type]

def load_eeyore_dataset(dataset_type: str, indices: list = None):
    """Load dataset based on type and optional indices.
    
    Args:
        dataset_type: Type of dataset to load (e.g., "esc")
        indices: Optional list of specific indices to load
        
    Returns:
        DataFrame containing the loaded dataset
        
    Raises:
        ValueError: If dataset_type is not supported

    """
    match dataset_type:
        case "esc":
            if indices is not None:
                return load_esc_data_with_indices(original_indices=indices)
            return load_esc_data()
        case "hope":
            return load_eeyore_from_hf(dataset_type="HOPE")
        case "annomi":
            return load_eeyore_from_hf(dataset_type="AnnoMI")
        case _:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")