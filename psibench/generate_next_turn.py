# TODO: Real history + Next Turn
import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from data_loader.main_loader import load_eeyore_dataset
from agents.patient import PatientAgent
from models.eeyore import prepare_prompt_from_profile
from models.patient_psi import generate_chain
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generated simulated next turn patient response"
    )

    parser.add_argument("--dataset", type=str, default="esc", help="Dataset type (default: esc)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--turn_idx", type=int, default=0, help="Start from which therapist turn")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    parser.add_argument("--N", type=int, default=5, help="Number of conversations to generate")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file (default: configs/default.yaml)")
    args = parser.parse_args()

    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    args.output_dir = args.output_dir.strip() if args.output_dir else args.output_dir
    args.psi = args.psi.strip().lower() if args.psi else args.psi

    return args

async def run_session(
    profile: Dict[str, Any], real_messages: list, config: Dict[str, Any], start_turn: int = 1
):
    patient = PatientAgent(patient_profile=profile, config=config)
    
    messages = []
    if real_messages[0]["role"] == "assistant":
        messages.append({"role": "assistant", "content": ""})
    # return_messages.append({"role": "patient", "content": ""})
    try:
        for i in range(1, len(real_messages), 2):
            if i < start_turn * 2:
                messages.append({"role": "user", "content": ""})
                messages.append({"role": "assistant", "content": ""})
            else:
                messages.append({"role": "user", "content": ""})
                patient_msg = await patient.respond(real_messages[:i], real_messages[i])
                messages.append({"role": "assistant", "content": patient_msg})
    except Exception as e:
        print(f"Session ended early due to error: {e}")
    
    return {"messages": messages, "profile": profile}

def save_session_results(
    session_data: Dict[str, Any], output_dir: Path, session_id: int
):
    """Save session results to JSON file."""
    output_path = output_dir / f"session_{session_id}_turn.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)


async def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["patient"]["simulator"] = args.psi
    output_dir = Path(args.output_dir) / args.psi / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_eeyore_dataset(args.dataset)
    
    # Limit number of conversations if specified
    if args.N is not None:
        df = df.head(args.N)

    print(f"Generating {len(df)} conversations from {args.dataset} dataset for PSI: {args.psi}")

    # Generate conversations
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            profile = json.loads(row["profile"])
            real_messages = row["messages"]
            if args.psi == "eeyore":
                system_prompt, _, _ = prepare_prompt_from_profile(profile)
                profile["system_prompt"] = system_prompt
            elif args.psi == "patientpsi":
                system_prompt = generate_chain(real_messages, config)
                profile["system_prompt"] = system_prompt
            # Run session
            final_state = await run_session(profile, real_messages, config=config, start_turn=args.turn_idx)

            # Save results
            save_session_results(final_state, output_dir, idx)

        except Exception as e:
            print(f"Error in session {idx}: {e}")
            continue

    print(f"\nFinished! Session transcripts saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())