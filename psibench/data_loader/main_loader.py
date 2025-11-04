from data_loader.esc import load_esc_data, load_esc_data_with_indices

def load_dataset(dataset_type: str, indices: list = None):
    """Load dataset based on type and optional indices.
    
    Args:
        dataset_type: Type of dataset to load (e.g., "esc")
        indices: Optional list of specific indices to load
        
    Returns:
        DataFrame containing the loaded dataset
        
    Raises:
        ValueError: If dataset_type is not supported
    """
    if dataset_type == "esc":
        if indices is not None:
            return load_esc_data_with_indices(original_indices=indices)
        return load_esc_data()
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")