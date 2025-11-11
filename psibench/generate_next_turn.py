"""Simulate next-turn patient responses leveraging batched model_interface calls."""

import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

from psibench.data_loader.main_loader import load_eeyore_dataset
from psibench.agents.patient import PatientAgent
from psibench.models.eeyore import prepare_prompt_from_profile


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generated simulated next turn patient response"
    )

    parser.add_argument("--dataset", type=str, default="esc", help="Dataset type (default: esc)")
    parser.add_argument("--output-dir", type=str, default="data/synthetic", help="Output directory")
    parser.add_argument("--turn_idx", type=int, default=0, help="Start from which therapist turn")
    parser.add_argument("--psi", type=str, default="eeyore", help="Type of patient sim to use")
    parser.add_argument("--N", type=int, default=5, help="Number of conversations to generate")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Number of sessions to run in each batch (default: 4)")
    args = parser.parse_args()

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


def run_batched_turn_sessions(
    batch_rows: List[Tuple[int, Any]],
    config: Dict[str, Any],
    psi: str,
    patient_client,
    start_turn: int,
) -> List[Tuple[int, Dict[str, Any]]]:
    """For each session, keep real history up to start_turn and simulate patient-only turn thereafter."""
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

        real_messages = row["messages"]
        messages = []
        if real_messages and real_messages[0].get("role") == "assistant":
            messages.append({"role": "assistant", "content": ""})

        # Copy placeholder turns up to start_turn
        for i in range(1, len(real_messages), 2):
            if i < start_turn * 2:
                messages.append({"role": "user", "content": ""})
                messages.append({"role": "assistant", "content": ""})
            else:
                break

        sessions.append({
            "idx": idx,
            "patient_agent": patient_agent,
            "messages": messages,
            "real_messages": real_messages,
        })

    # Prepare requests for next patient turn
    requests = []
    for session in sessions:
        real_messages = session["real_messages"]
        target_idx = start_turn * 2
        if target_idx >= len(real_messages):
            continue
        requests.append({
            "agent": session["patient_agent"],
            "session": session,
            "history": real_messages[:target_idx],
            "latest": real_messages[target_idx],
        })

    responses = _batch_generate(requests, patient_client)
    for req, response in zip(requests, responses):
        session = req["session"]
        if response is None:
            continue
        session["messages"].append({"role": "user", "content": ""})
        session["messages"].append({"role": "assistant", "content": response})

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


def save_session_results(session_data: Dict[str, Any], output_dir: Path, session_id: int):
    """Save session results to JSON file."""
    output_path = output_dir / f"session_{session_id}_turn.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(session_data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()

    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    output_dir = Path(args.output_dir) / args.psi / f"{args.dataset}_turn"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df = load_eeyore_dataset(args.dataset)
    
    # Limit number of conversations if specified
    if args.N is not None:
        df = df.head(args.N)

    print(f"Generating {len(df)} conversations from {args.dataset} dataset")

    patient_cfg = config.get("patient", {})
    use_batched = args.concurrency > 1 and patient_cfg.get("backend", "langchain") == "model_interface"

    try:
        patient_client = create_model_interface_client(patient_cfg, config) if use_batched else None
    except ValueError as exc:
        print(f"[WARNING] Falling back to sequential mode: {exc}")
        use_batched = False
        patient_client = None

    rows = list(df.iterrows())

    if use_batched:
        total_batches = (len(rows) + args.concurrency - 1) // args.concurrency
        for batch in tqdm(
            chunk_rows(rows, args.concurrency),
            total=total_batches,
            desc="Generating (batched next-turn)"
        ):
            results = run_batched_turn_sessions(
                batch_rows=batch,
                config=config,
                psi=args.psi,
                patient_client=patient_client,
                start_turn=args.turn_idx,
            )
            for idx, final_state in results:
                save_session_results(final_state, output_dir, idx)
    else:
        for idx, row in tqdm(rows, desc="Generating"):
            try:
                profile = json.loads(row["profile"])
                messages = row["messages"]
                if args.psi == "eeyore":
                    system_prompt, _, _ = prepare_prompt_from_profile(profile)
                    profile["eeyore_system_prompt"] = system_prompt

                final_state = run_batched_turn_sessions(
                    batch_rows=[(idx, row)],
                    config=config,
                    psi=args.psi,
                    patient_client=None,
                    start_turn=args.turn_idx,
                )[0][1]
                save_session_results(final_state, output_dir, idx)
            except Exception as e:
                print(f"Error in session {idx}: {e}")

    print(f"\nFinished! Session transcripts saved to {output_dir}/")


if __name__ == "__main__":
    main()
