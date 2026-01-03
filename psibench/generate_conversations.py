"""Script to generate simulated counseling conversations."""

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict
import traceback

import yaml
from dotenv import load_dotenv
from tqdm import tqdm
import time
load_dotenv()


from psibench.data_loader.main_loader import load_eeyore_dataset
from psibench.agents.patient import PatientAgent
from psibench.agents.therapist import TherapistAgent
from psibench.models.eeyore import prepare_prompt_from_profile
from psibench.models.patient_psi import generate_chain, generate_chain_batch


def validate_patient_turns(messages: list, expected_num_patient_turns: int) -> bool:
    """Validate that the number of non-empty patient messages matches expected count.
    
    Args:
        messages: List of conversation messages
        expected_num_patient_turns: Expected number of patient turns
        
    Returns:
        True if validation passes, False otherwise
    """
    actual_patient_turns = sum(
        1 for msg in messages 
        if msg.get('role') == 'assistant' and msg.get('content', '').strip()
    )
    return actual_patient_turns == expected_num_patient_turns


def session_exists_and_valid(output_path: Path, check_ccd: bool = False) -> tuple:
    """Check if session file exists and passes validation.
    
    Args:
        output_path: Path to the session JSON file
        check_ccd: If True, also check for CCD and return it
        
    Returns:
        If check_ccd=False: bool indicating if session is valid
        If check_ccd=True: tuple (is_valid: bool, ccd: dict or None)
    """
    if not output_path.exists():
        return (False, None) if check_ccd else False
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        messages = session_data.get('messages', [])
        expected_turns = session_data.get('num_patient_turns')
        
        if expected_turns is None:
            return (False, None) if check_ccd else False
        
        is_valid = validate_patient_turns(messages, expected_turns)
        
        if check_ccd:
            ccd = session_data.get('ccd')
            return (is_valid, ccd)
        return is_valid
    except Exception:
        return (False, None) if check_ccd else False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate simulated counseling conversations"
    )
    parser.add_argument(
        "--dataset", type=str, default="esc", help="Dataset type: esc | hope | annomi | all (default: esc)"
    )
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    parser.add_argument("--N", type=int, default=None, 
                    help="Number of conversations to generate (default: all available samples)")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to config file (default: configs/default.yaml)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Number of parallel tasks to run (default: 1)")
    args = parser.parse_args()
    
    # Clean string input arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    args.output_dir = args.output_dir.strip() if args.output_dir else args.output_dir
    args.psi = args.psi.strip().lower() if args.psi else args.psi
    
    return args


async def run_session(
    profile: Dict[str, Any], real_messages: list, config: Dict[str, Any]) -> Dict[str, Any]:
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
    real_num_patient_turns = sum(1 for msg in real_messages if msg.get("role") in ["assistant", "patient"])
    
    # Get max_turns from config and take minimum with real message length
    max_turns = config.get("patient", {}).get("max_turns")
    if max_turns is not None:
        num_patient_turns = min(real_num_patient_turns, max_turns)
        print(f"Limiting conversation length: {num_patient_turns} patient turns (real: {real_num_patient_turns}, max_turns: {max_turns})")
    else:
        num_patient_turns = real_num_patient_turns
        print(f"Matching conversation length: {num_patient_turns} patient turns")

    if real_messages and real_messages[0]["role"] in ["assistant", "patient"]:
        messages.append({"role": "assistant", "content": ""})
        num_patient_turns -= 1  # Reduce count since we're adding initial assistant message
    try:
        # Start conversation with therapist
        try:
            therapist_msg = therapist.respond(messages)
        except Exception as e:
            print(f"Error generating initial therapist message: {e}")
            print(f"Traceback: {traceback.format_exc()}")
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
        print(f"Error type: {type(e).__name__}")
        print(f"Full traceback:\n{traceback.format_exc()}")

    # Validate num_patient_turns matches actual non-empty assistant messages
    if not validate_patient_turns(messages, num_patient_turns):
        actual_patient_turns = sum(
            1 for msg in messages 
            if msg.get('role') == 'assistant' and msg.get('content', '').strip()
        )
        raise ValueError(
            f"Data quality check FAILED: num_patient_turns={num_patient_turns} but "
            f"found {actual_patient_turns} non-empty assistant messages. "
        )
    
    return {"messages": messages, "profile": profile, "num_patient_turns": num_patient_turns}


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

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    config["patient"]["simulator"] = args.psi
     # Extract model name for output directory
    model_name = config.get("patient").get("model")
    # Clean model name for use in path (remove special characters)
    clean_model_name = model_name.replace("/", "_").replace(":", "_")
    
    output_dir = Path(args.output_dir) / args.psi / clean_model_name / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_eeyore_dataset(args.dataset)

    if args.N:
        df = df.head(args.N)

    print(f"Generating {len(df)} conversations from {args.dataset} dataset for PSI: {args.psi}")
    print(f"Batch size: {args.batch_size} parallel tasks")

    # For patientpsi: process in batches (extract profiles per batch, then generate conversations)
    if args.psi == "patientpsi":
        # Prepare all data
        all_data = []
        for idx, row in df.iterrows():
            try:
                profile = json.loads(row["profile"])
                real_messages = row["messages"]
                source = row.get("source")
                all_data.append((idx, real_messages, source))
            except Exception as e:
                print(f"Error loading data for session {idx}: {e}")
                continue
        
        skipped_count = 0
        total_batches = (len(all_data) + args.batch_size - 1) // args.batch_size
        
        for batch_num, batch_start in enumerate(range(0, len(all_data), args.batch_size), 1):
            batch_end = min(batch_start + args.batch_size, len(all_data))
            batch_data = all_data[batch_start:batch_end]
            
            print(f"\n=== Processing batch {batch_num}/{total_batches} (sessions {batch_start}-{batch_end-1}) ===")
            
            # Filter out already-valid sessions and check for existing CCDs
            indices_to_generate = []
            real_messages_to_generate = []
            batch_tasks = []
            existing_ccds = {}  # idx -> ccd mapping
            
            for idx, real_messages, source in batch_data:
                output_path = output_dir / f"session_{idx}.json"
                is_valid, existing_ccd = session_exists_and_valid(output_path, check_ccd=True)
                
                if is_valid:
                    print(f"Skipping session {idx} (already exists and passes validation)")
                    skipped_count += 1
                elif existing_ccd is not None:
                    # Session exists with CCD but conversation incomplete - reuse CCD
                    print(f"Reusing existing CCD for session {idx}")
                    existing_ccds[idx] = existing_ccd
                    batch_tasks.append((idx, real_messages, source))
                else:
                    # Need to generate CCD
                    indices_to_generate.append(idx)
                    real_messages_to_generate.append(real_messages)
                    batch_tasks.append((idx, real_messages, source))
            
            if not batch_tasks:
                print("All sessions in this batch already complete, skipping...")
                continue
            
            # Generate profiles only for sessions without existing CCD
            ccds_and_profiles = []
            if real_messages_to_generate:
                print(f"Generating {len(real_messages_to_generate)} profiles for this batch...")
                ccds_and_profiles = generate_chain_batch(real_messages_to_generate, config)
            
            # Pair profiles with indices and messages
            tasks_to_run = []
            gen_idx = 0
            for idx, real_messages, source in batch_tasks:
                if idx in existing_ccds:
                    # Use existing CCD
                    ccd = existing_ccds[idx]
                    from psibench.models.patient_psi import format_patient_psi_prompt_from_ccd
                    profile = format_patient_psi_prompt_from_ccd(
                        ccd=ccd,
                        patient_type_content=config.get('patient').get('conversation_type'),
                        name="Patient"
                    )
                    tasks_to_run.append((idx, profile, real_messages, ccd, source))
                else:
                    # Use newly generated CCD
                    ccd, profile = ccds_and_profiles[gen_idx]
                    gen_idx += 1
                    if profile is None:
                        print(f"Error generating profile for session {idx}: batch call returned None")
                        continue
                    tasks_to_run.append((idx, profile, real_messages, ccd, source))
            
            if not tasks_to_run:
                continue
            
            # Generate conversations for this batch
            print(f"Generating {len(tasks_to_run)} conversations in parallel...")
            batch_coroutines = [
                run_session(profile, real_messages, config=config)
                for idx, profile, real_messages, ccd, _ in tasks_to_run
            ]
            
            results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
            
            # Save results with CCD
            for (idx, _, _, ccd, source), result in zip(tasks_to_run, results):
                if isinstance(result, Exception):
                    print(f"\nError in session {idx}: {result}")
                    print(f"Error type: {type(result).__name__}")
                    print(f"Full traceback:\n{''.join(traceback.format_exception(type(result), result, result.__traceback__))}")
                else:
                    # Add CCD to result
                    result['ccd'] = ccd
                    result['source'] = source
                    save_session_results(result, output_dir, idx)
        
        print(f"\nFinished! Session transcripts saved to {output_dir}/")
        print(f"Skipped {skipped_count} sessions (already valid)")
        return
    
    # Original logic for eeyore and roleplaydoh
    all_tasks = []
    for idx, row in df.iterrows():
        try:
            profile = json.loads(row["profile"])
            real_messages = row["messages"]
            source = row.get("source")
            if args.psi == "eeyore":
                system_prompt, _, _ = await prepare_prompt_from_profile(profile)
                profile["eeyore_system_prompt"] = system_prompt

            all_tasks.append((idx, profile, real_messages, source))

        except Exception as e:
            print(f"Error preparing session {idx}: {e}")
            continue
    
    # Process tasks in batches
    skipped_count = 0
    for batch_start in range(0, len(all_tasks), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(all_tasks))
        batch = all_tasks[batch_start:batch_end]
        
        # Filter tasks: skip sessions that already exist and are valid
        tasks_to_run = []
        for idx, profile, real_messages, source in batch:
            output_path = output_dir / f"session_{idx}.json"
            if session_exists_and_valid(output_path, check_ccd=False):
                print(f"Skipping session {idx} (already exists and passes validation)")
                skipped_count += 1
            else:
                tasks_to_run.append((idx, profile, real_messages, source))
        
        if not tasks_to_run:
            continue
        
        # Create and execute batch tasks
        batch_coroutines = [
            run_session(profile, real_messages, config=config)
            for idx, profile, real_messages, _ in tasks_to_run
        ]
        
        results = await asyncio.gather(*batch_coroutines, return_exceptions=True)
        
        # Save results for this batch
        for (idx, _, _, source), result in zip(tasks_to_run, results):
            if isinstance(result, Exception):
                print(f"\nError in session {idx}: {result}")
            else:
                result['source'] = source
                save_session_results(result, output_dir, idx)

    print(f"\nFinished! Session transcripts saved to {output_dir}/")
    print(f"Skipped {skipped_count} sessions (already valid)")

if __name__ == "__main__":
    start_time = time.time()
    asyncio.run(main())
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")
