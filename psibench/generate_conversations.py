"""Script to generate simulated counseling conversations."""

import json
import yaml
import argparse
import os
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()


from data_loader.main_loader import load_eeyore_dataset
from agents.patient import PatientAgent
from agents.therapist import TherapistAgent
from models.eeyore import prepare_prompt_from_profile

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate simulated counseling conversations")
    parser.add_argument("--dataset", type=str, default="esc", help="Dataset type (default: esc)")
    parser.add_argument("--max-turns", type=int, help="Override max turns from config")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    parser.add_argument("--N", type=int, default=5, 
                       help="Number of conversations to generate (default: all available samples)")
    
    args = parser.parse_args()
    
    # Clean string input arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    args.output_dir = args.output_dir.strip() if args.output_dir else args.output_dir
    args.psi = args.psi.strip().lower() if args.psi else args.psi
    
    return args


def run_session(profile: Dict[str, Any], config: Dict[str, Any], psi: str = "eeyore") -> Dict[str, Any]:
    """Run a simulated counseling session."""
    
    # Initialize agents
    patient = PatientAgent(patient_profile=profile, config=config)
    therapist = TherapistAgent(config=config)
    
    messages = []
    max_turns = config["session"]["max_turns"]
    
    try:
        # Start conversation with therapist
        therapist_msg = therapist.respond(messages)
        messages.append({"role": "user", "content": therapist_msg})
        
        # Main conversation loop
        # User: therapist, Assistant: patient
        for _ in range(max_turns):
            patient_msg = patient.respond(messages[:-1], messages[-1]["content"])
            messages.append({"role": "assistant", "content": patient_msg})
            
            therapist_msg = therapist.respond(messages[:-1], messages[-1]["content"])
            messages.append({"role": "user", "content": therapist_msg})
            
    except Exception as e:
        print(f"Session ended early due to error: {e}")
    
    return {
        "messages": messages,
        "profile": profile
    }

def save_session_results(session_data: Dict[str, Any], output_dir: Path, session_id: int):
    """Save session results to JSON file."""
    output_path = output_dir / f"session_{session_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)

def main():
    args = parse_args()
    
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    if args.max_turns:
        config["session"]["max_turns"] = args.max_turns

    output_dir = Path(args.output_dir) / args.psi / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_eeyore_dataset(args.dataset)
    
    # Limit number of conversations if specified
    if args.N is not None:
        df = df.head(args.N)
    
    print(f"Generating {len(df)} conversations from {args.dataset} dataset")
    
    # Generate conversations
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            profile = json.loads(row["profile"])          
            if args.psi == "eeyore":
                system_prompt, _, _ = prepare_prompt_from_profile(profile)
                profile["eeyore_system_prompt"] = system_prompt
            
            # Run session
            final_state = run_session(profile, config=config, psi=args.psi)
            
            # Save results
            save_session_results(final_state, output_dir, idx)
            
        except Exception as e:
            print(f"Error in session {idx}: {e}")
            continue
    
    print(f"\nFinished! Session transcripts saved to {output_dir}/")

if __name__ == "__main__":
    main()
