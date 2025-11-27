"""Quick test script for classify_conversation_batch_by_conversation method."""

import yaml
from psibench.eval.ptc_classification import PTCClassifier
from psibench.data_loader.main_loader import load_eeyore_dataset

def test_batch_classification():
    """Test the batch classification method on a single conversation."""
    
    # Load config
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize classifier
    print("Initializing PTCClassifier...")
    judge = PTCClassifier(config)
    
    # Load a single conversation from ESC dataset
    print("Loading test conversation from ESC dataset...")
    df = load_eeyore_dataset(dataset_type="esc", indices=[3])
    
    # Get the first conversation
    messages = df.loc[3]["messages"]
    print(f"\nTest conversation has {len(messages)} messages")
    
    # Count patient turns
    patient_turns = sum(1 for msg in messages if msg["role"] in ["assistant", "patient"])
    print(f"Patient turns: {patient_turns}")
    
    # Test the batch classification
    print("\n" + "="*60)
    print("Testing classify_conversation_batch_by_conversation()...")
    print("="*60)
    
    try:
        classifications = judge.classify_conversation_batch_by_conversation(messages)
        print(classifications)
        import json
        output = {"messages": messages, "classifications": classifications}
        json.dump(output, open("ptc_batch_output.json", "w"), indent=2)
        print(len(messages)==len(classifications))
        return True
    except Exception as e:
        print(f"Error during batch classification: {e}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("PTC Batch Classification Test")
    print("="*60)
    
    success = test_batch_classification()
    
    print("\n" + "="*60)
    if success:
        print("✓ Test completed successfully!")
    else:
        print("✗ Test failed!")
    print("="*60)
