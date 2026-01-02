from collections import defaultdict
from typing import List, Dict

def extract_patient_messages_by_turn(conversations: List[Dict], max_turns: int = None) -> Dict[int, List[str]]:
    """Extract patient messages organized by turn index.
    
    Args:
        conversations: List of conversation dictionaries
        max_turns: Maximum turn to analyze (None = analyze all)
        
    Returns:
        Dictionary mapping turn_index -> list of patient messages at that turn
    """
    messages_by_turn = defaultdict(list)
    
    for conv in conversations:
        messages = conv.get('messages', [])
        patient_turn_idx = 0
        
        for msg in messages:
            # Patient messages have role 'assistant'
            if msg.get('role') == 'assistant':
                content = msg.get('content', '').strip()
                if content:  # Only count non-empty messages
                    if max_turns is None or patient_turn_idx < max_turns:
                        messages_by_turn[patient_turn_idx].append(content)
                    patient_turn_idx += 1
    
    return messages_by_turn
