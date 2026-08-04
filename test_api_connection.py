"""Simple test to verify API connectivity for therapist and patient agents."""

import argparse
import yaml
from psibench.agents.therapist import TherapistAgent
from psibench.agents.patient import PatientAgent

def test_therapist_api(config_path: str):
    """Test therapist API call."""
    print("\n" + "="*60)
    print("Testing Therapist API...")
    
    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Initialize therapist
        therapist = TherapistAgent(config=config)
        
        # Simple test call
        response = therapist.respond(
            conversation_history=[],
            patient_message=None  # Starting the session
        )
        
        print("✓ Therapist API call SUCCESSFUL")
        print(f"Response preview: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ Therapist API call FAILED")
        print(f"Error: {e}")
        return False


def test_patient_api(config_path: str):
    """Test patient API call."""
    print("\n" + "="*60)
    print("Testing Patient API...")    
    try:
        # Load config
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Simple patient profile
        patient_profile = {
                "gender": "Female",
                "age": "25~44",
                "marital status": "Married",
                "occupation": "Unemployed",
                "situation of the client": "The client is feeling overwhelmed and guilty due to overspending on a sale, has no job, and is struggling to pay off credit card debt. Her husband pays all the bills, and she feels pressure to resolve the financial issue.",
                "resistance toward the support": "Medium",
                "symptom severity": {
                "feelings of sadness, tearfulness, emptiness, or hopelessness": "Moderate",
                "angry outbursts, irritability, or frustration, even over small matters": "Mild",
                "tiredness and lack of energy, so even small tasks take extra effort": "Mild",
                "anxiety, agitation, or restlessness": "Moderate",
                "feelings of worthlessness or guilt, fixating on past failures or self-blame": "Moderate",
                "increased engagement in high-risk activities": "Mild",
                "greater impulsivity": "Moderate",
                "inability to meet the responsibilities of work and family or ignoring other important roles": "Moderate"
                },
                "cognition distortion exhibition": {
                "personalization": "Exhibited",
                "catastrophic thinking": "Exhibited"
                },
                "depression severity": "Moderate Depression",
                "suicidal ideation severity": "No suicidal ideation",
                "homicidal ideation severity": "No homicidal ideation",
                "counseling history": "N/A"
            }

        patient = PatientAgent(patient_profile=patient_profile, config=config)
        
        # Simple test call
        response = patient.respond(
            conversation_history=[],
            therapist_message="Hello, how are you feeling today?"
        )
        
        print("✓ Patient API call SUCCESSFUL")
        print(f"Response preview: {response[:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ Patient API call FAILED")
        print(f"Error: {e}")
        return False


def main():
    """Run all API tests."""
    parser = argparse.ArgumentParser(description="Test API connectivity for PSI-bench agents")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vllm.yaml",
        help="Path to YAML config file (default: configs/vllm.yaml)",
    )
    args = parser.parse_args()
    
    results = []
    
    # Test therapist
    results.append(("Therapist", test_therapist_api(args.config)))
    
    # Test patient
    results.append(("Patient", test_patient_api(args.config)))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name}: {status}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n✓ All API tests passed!")
    else:
        print("\n✗ Some API tests failed. Check errors above.")
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
