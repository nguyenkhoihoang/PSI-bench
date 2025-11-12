from pathlib import Path

# Load locally saved synthetic data from local path
def get_synthetic_indices(data_dir: str):
    """Get the indices from synthetic session files."""
    data_dir = Path(data_dir)
    indices = []
    
    for session_file in sorted(data_dir.glob('session_*.json')):
        # Extract index from session_XXX.json
        idx = int(session_file.stem.split('_')[1])
        indices.append(idx)
    
    return sorted(indices)

def merge_consecutive_messages(real_messages:list):
    """Merge consecutive messages with the same role."""

    merged = []
    current = real_messages[0]
    
    for msg in real_messages[1:]:
        if msg['role'] == current['role']:
            current['content'] += '\n' + msg['content']
        else:
            merged.append(current)
            current = msg
    
    merged.append(current)
    return merged
