"""Script to generate simulated counseling conversations."""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple, Iterable
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


from psibench.data_loader.main_loader import load_eeyore_dataset
from psibench.agents.patient import PatientAgent
from psibench.agents.therapist import TherapistAgent
from psibench.models.eeyore import prepare_prompt_from_profile


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate simulated counseling conversations")
    parser.add_argument("--dataset", type=str, default="esc", help="Dataset type (default: esc)")
    parser.add_argument("--max-turns", type=int, help="Override max turns from config")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    parser.add_argument("--N", type=int, default=5, 
                       help="Number of conversations to generate (default: all available samples)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of sessions to run in parallel (default: 4)")
    
    args = parser.parse_args()
    
    # Clean string input arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    args.output_dir = args.output_dir.strip() if args.output_dir else args.output_dir
    args.psi = args.psi.strip().lower() if args.psi else args.psi
    
    return args


def chunk_rows(rows: List[Tuple[int, Any]], size: int) -> Iterable[List[Tuple[int, Any]]]:
    for i in range(0, len(rows), size):
        yield rows[i:i + size]


def create_model_interface_client(role_cfg: Dict[str, Any], config: Dict[str, Any]):
    config_path = role_cfg.get("model_interface_config") or config.get("model_interface", {}).get("config_path")
    if not config_path:
        raise ValueError("model_interface backend selected but no config path provided.")
    from model_interface import create_model
    return create_model(config_path)


def _batch_generate(requests: List[Dict[str, Any]], shared_client):
    """Batch generate responses for multiple sessions."""
    if not requests:
        return []

    sample_agent = requests[0]["agent"]
    if sample_agent.backend != "model_interface" or shared_client is None:
        outputs = []
        for req in requests:
            try:
                outputs.append(req["agent"].respond(req["history"], req["latest"]))
            except Exception as exc:
                print(f"[ERROR] Sequential generation failed: {exc}")
                outputs.append(None)
        return outputs

    messages_list = []
    for req in requests:
        inputs = req["agent"].build_inputs(req["history"], req["latest"])
        req["inputs"] = inputs
        messages_list.append(req["agent"].build_model_interface_messages(inputs))

    try:
        raw_outputs = shared_client.generate(messages_list, **sample_agent.generation_kwargs)
    except Exception as exc:
        print(f"[ERROR] Batched generation failed: {exc}")
        return [None] * len(requests)

    responses = []
    for req, raw in zip(requests, raw_outputs):
        if raw is None:
            responses.append(None)
            continue
        try:
            responses.append(req["agent"].parse_model_interface_response(raw))
        except Exception as exc:
            print(f"[ERROR] Failed to parse batched response: {exc}")
            responses.append(None)
    return responses


def run_batched_sessions(
    batch_rows: List[Tuple[int, Any]],
    config: Dict[str, Any],
    psi: str,
    therapist_agent: TherapistAgent,
    patient_client,
) -> List[Tuple[int, Dict[str, Any]]]:
    """Advance multiple sessions in lockstep so each turn can be batched."""
    max_turns = config["session"]["max_turns"]
    therapist_client = therapist_agent.model_interface_client if therapist_agent.backend == "model_interface" else None
    sessions = []

    for idx, row in batch_rows:
        try:
            profile = json.loads(row["profile"])
        except Exception as exc:
            print(f"[ERROR] Failed to load profile for session {idx}: {exc}")
            continue

        if psi == "eeyore":
            system_prompt, _, _ = prepare_prompt_from_profile(profile)
            profile["eeyore_system_prompt"] = system_prompt

        patient_agent = PatientAgent(
            patient_profile=profile,
            config=config,
            model_interface_client=patient_client
        )

        messages = []
        real_messages = row["messages"]
        if real_messages and real_messages[0].get("role") == "assistant":
            messages.append({"role": "assistant", "content": ""})

        sessions.append({
            "idx": idx,
            "patient_agent": patient_agent,
            "messages": messages,
            "active": True,
        })

    # Initial therapist opening
    opening_requests = [
        {
            "agent": therapist_agent,
            "session": session,
            "history": session["messages"],
            "latest": None,
        }
        for session in sessions
    ]
    opening_responses = _batch_generate(opening_requests, therapist_client)
    for req, response in zip(opening_requests, opening_responses):
        if response is None:
            req["session"]["active"] = False
            continue
        req["session"]["messages"].append({"role": "user", "content": response})

    # Turn-by-turn simulation
    for _ in range(max_turns):
        patient_requests = []
        for session in sessions:
            if not session["active"] or not session["messages"]:
                continue
            patient_requests.append({
                "agent": session["patient_agent"],
                "session": session,
                "history": session["messages"][:-1],
                "latest": session["messages"][-1]["content"],
            })

        if not patient_requests:
            break

        patient_responses = _batch_generate(patient_requests, patient_client)
        for req, response in zip(patient_requests, patient_responses):
            if response is None:
                req["session"]["active"] = False
                continue
            req["session"]["messages"].append({"role": "assistant", "content": response})

        therapist_requests = []
        for session in sessions:
            if not session["active"] or not session["messages"]:
                continue
            therapist_requests.append({
                "agent": therapist_agent,
                "session": session,
                "history": session["messages"][:-1],
                "latest": session["messages"][-1]["content"],
            })

        if not therapist_requests:
            break

        therapist_responses = _batch_generate(therapist_requests, therapist_client)
        for req, response in zip(therapist_requests, therapist_responses):
            if response is None:
                req["session"]["active"] = False
                continue
            req["session"]["messages"].append({"role": "user", "content": response})

    final_states = []
    for session in sessions:
        final_states.append((
            session["idx"],
            {
                "messages": session["messages"],
                "profile": session["patient_agent"].patient_profile,
            }
        ))
    return final_states

def run_session(
    profile: Dict[str, Any], real_messages: list, config: Dict[str, Any], psi: str = "eeyore"
) -> Dict[str, Any]:
    """Run a simulated counseling session."""
    
    # Initialize agents
    patient = PatientAgent(patient_profile=profile, config=config)
    therapist = TherapistAgent(config=config)
    
    messages = []
    max_turns = config["session"]["max_turns"]

    if real_messages[0]["role"] == "assistant":
        messages.append({"role": "assistant", "content": ""})
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

    rows = list(df.iterrows())
    patient_cfg = config.get("patient", {})
    therapist_cfg = config.get("therapist", {})
    use_batched = (
        args.concurrency > 1
        and patient_cfg.get("backend", "langchain") == "model_interface"
        and therapist_cfg.get("backend", "langchain") == "model_interface"
    )

    if use_batched:
        try:
            patient_client = create_model_interface_client(patient_cfg, config)
            therapist_client = create_model_interface_client(therapist_cfg, config)
        except ValueError as exc:
            print(f"[WARNING] Falling back to sequential mode: {exc}")
            use_batched = False
        else:
            therapist_agent = TherapistAgent(config=config, model_interface_client=therapist_client)
            total_batches = (len(rows) + args.concurrency - 1) // args.concurrency
            for batch in tqdm(
                chunk_rows(rows, args.concurrency),
                total=total_batches,
                desc="Generating (batched)"
            ):
                batch_results = run_batched_sessions(
                    batch_rows=batch,
                    config=config,
                    psi=args.psi,
                    therapist_agent=therapist_agent,
                    patient_client=patient_client,
                )
                for idx, final_state in batch_results:
                    save_session_results(final_state, output_dir, idx)

    if not use_batched:
        for idx, row in tqdm(rows, desc="Generating"):
            try:
                profile = json.loads(row["profile"])
                messages = row["messages"]
                if args.psi == "eeyore":
                    system_prompt, _, _ = prepare_prompt_from_profile(profile)
                    profile["eeyore_system_prompt"] = system_prompt

                final_state = run_session(profile, messages, config=config, psi=args.psi)
                save_session_results(final_state, output_dir, idx)
            except Exception as e:
                print(f"Error in session {idx}: {e}")
    
    print(f"\nFinished! Session transcripts saved to {output_dir}/")

if __name__ == "__main__":
    main()
