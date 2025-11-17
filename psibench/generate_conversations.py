"""Script to generate simulated counseling conversations."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()


# from data_loader.main_loader import load_eeyore_dataset
# from agents.patient import PatientAgent
# from agents.therapist import TherapistAgent
# from models.eeyore import prepare_prompt_from_profile
from psibench.data_loader.main_loader import load_eeyore_dataset
from psibench.agents.patient import PatientAgent
from psibench.agents.therapist import TherapistAgent
from psibench.models.eeyore import prepare_prompt_from_profile
from models.patient_psi import generate_chain
from models.generation_template import GenerationModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate simulated counseling conversations"
    )
    parser.add_argument(
        "--dataset", type=str, default="esc", help="Dataset type (default: esc)"
    )
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


async def run_session(
    profile: Dict[str, Any], real_messages: list, config: Dict[str, Any], psi: str = "eeyore"
) -> Dict[str, Any]:
    """Run a simulated counseling session matching the length of real messages.
    
    Args:
        profile: Patient profile
        real_messages: Real conversation messages to match length
        config: Configuration dictionary
        psi: Patient simulator type
        
    Returns:
        Dictionary with generated messages and profile
    """
    print("Running Session")
    # Initialize agents
    patient = PatientAgent(patient_profile=profile, config=config)
    therapist = TherapistAgent(config=config)


    messages = []
    
    # Calculate number of patient turns from real messages
    num_patient_turns = sum(1 for msg in real_messages if msg.get("role") in ["assistant", "patient"])
    print(f"Matching conversation length: {num_patient_turns} patient turns")

    if real_messages and real_messages[0]["role"] in ["assistant", "patient"]:
        messages.append({"role": "assistant", "content": ""})
        num_patient_turns -= 1  # Reduce count since we're adding initial assistant message
    try:
        # Start conversation with therapist
        therapist_msg = therapist.respond(messages)
        messages.append({"role": "user", "content": therapist_msg})

        # Main conversation loop - generate same number of turns as real conversation
        # User: therapist, Assistant: patient
        for _ in range(num_patient_turns):
            patient_msg = await patient.respond(messages[:-1], messages[-1]["content"])
            messages.append({"role": "assistant", "content": patient_msg})

            therapist_msg = therapist.respond(messages[:-1], messages[-1]["content"])
            messages.append({"role": "user", "content": therapist_msg})

    except Exception as e:
        print(f"Session ended early due to error: {e}")

    return {"messages": messages, "profile": profile}


def save_session_results(
    session_data: Dict[str, Any], output_dir: Path, session_id: int
):
    """Save session results to JSON file."""
    output_path = output_dir / f"session_{session_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)


async def main():
    args = parse_args()

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    # if args.max_turns:
    #     config["session"]["max_turns"] = args.max_turns

    config["patient"]["simulator"] = args.psi

    output_dir = Path(args.output_dir) / args.psi / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_eeyore_dataset(args.dataset)

    if args.N is not None:
        df = df.head(args.N)

    print(f"Generating {len(df)} conversations from {args.dataset} dataset")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            profile = json.loads(row["profile"])
            real_messages = row["messages"]
            if args.psi == "eeyore":
                system_prompt, _, _ = prepare_prompt_from_profile(profile)
                profile["eeyore_system_prompt"] = system_prompt

            elif args.psi == "patientpsi":
                print("Started doing thi")
                # generate_chain returns the patientpsi prompt which includes system prompt for patient Agent
                system_prompt = generate_chain(real_messages)
                print("system_prompt = generate_chain(messages)")
                print(system_prompt)
                profile = {"system_prompt": system_prompt}

            # Run session
            final_state = await run_session(profile, real_messages, config=config, psi=args.psi)

            # Save results
            save_session_results(final_state, output_dir, idx)

        except Exception as e:
            print(f"Error in session {idx}: {e}")
            continue

    print(f"\nFinished! Session transcripts saved to {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())
