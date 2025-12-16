"""
Test script for single-turn PTC classification with limited history.

This script demonstrates the new classify_turns_batch method that classifies
individual patient messages based on a limited history window.
"""

import yaml
from pathlib import Path
from psibench.eval.ptc.ptc_classification import PTCClassifier
from psibench.data_loader.main_loader import load_eeyore_dataset
def test_single_turn_classification():
    """Test the new single-turn classification method."""
    
    # Load configuration
    config_path = Path("configs/default.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize classifier
    judge = PTCClassifier(config, debug=True)
    
    # Load a single conversation from ESC dataset
    print("Loading test conversation from ESC dataset...")
    df = load_eeyore_dataset(dataset_type="esc", indices=[3])
    
    # Get the first conversation
    messages = df.loc[3]["messages"]
    print(messages)
    print(type(messages))
    print(f"\nTest conversation has {len(messages)} messages")
    
    # Count patient turns
    patient_turns = sum(1 for msg in messages if msg["role"] in ["assistant", "patient"])
    print(f"Patient turns: {patient_turns}")
    
    # Test the batch classification
    print("\n" + "="*60)
    print("Testing classify_conversation_batch_by_conversation()...")
    print("="*60)
        
    # Test with different history windows
    for num_messages in [2, 4, 6]:
        print(f"\n--- Testing with num_messages={num_messages} ---")
        results = judge.classify_turns_batch([messages], num_messages=num_messages)
        
        print(f"\nResults for conversation 0:")
        for turn in results[0]:
            print(f"  Turn {turn['turn_index']}: {turn['classification']} - {turn['content']}")
    
    print("\n" + "=" * 70)
    print("Test completed!")
    
    # Compare with full conversation classification
    print("\n--- Comparing with full conversation classification ---")
    full_results = judge.classify_conversations_batch([messages])
    print(f"\nFull conversation classification:")
    for turn in full_results[0]:
        print(f"  {turn['classification']}: {turn.get('content', 'N/A')}")

if __name__ == "__main__":
    test_single_turn_classification()
