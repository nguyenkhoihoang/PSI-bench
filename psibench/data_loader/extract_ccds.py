#!/usr/bin/env python
"""Script to extract CCDs (Cognitive Conceptualization Diagrams) from real conversations.

This script extracts patient cognitive models from real therapy transcripts and saves them
locally. The extracted CCDs can then be pushed to HuggingFace and reused for generating
synthetic conversations without re-extracting.

Usage:
    python psibench/data_loader/extract_ccds.py \
    --dataset all \
    --output-dir data/ccds \
    --batch-size 16 \
    --skip-existing \
    --config configs/ccd_extraction.yaml
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import traceback

import yaml
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Suppress litellm logging
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

from psibench.data_loader.main_loader import load_real_dataset
from psibench.models.patient_psi import generate_chain_batch


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract CCDs from real conversations"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="esc",
        help="Dataset type: esc | hope | annomi | all (default: esc)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/ccds",
        help="Output directory for extracted CCDs (default: data/ccds)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file (default: configs/default.yaml)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Number of CCDs to extract in parallel (default: 10)"
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Number of conversations to process (default: all)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip sessions that already have extracted CCDs"
    )
    
    args = parser.parse_args()
    
    # Clean string arguments
    args.dataset = args.dataset.strip().lower() if args.dataset else args.dataset
    args.output_dir = args.output_dir.strip() if args.output_dir else args.output_dir
    
    return args


def ccd_exists(output_dir: Path, session_id: int) -> bool:
    """Check if CCD file already exists for session."""
    output_path = output_dir / f"ccd_{session_id}.json"
    if not output_path.exists():
        return False
    
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Validate it has required fields
            return 'ccd' in data and 'session_id' in data
    except Exception:
        return False


def save_ccd(output_dir: Path, session_id: int, messages: List[Dict], ccd: Dict[str, Any], source: str):
    """Save extracted CCD to JSON file."""
    output_path = output_dir / f"ccd_{session_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "session_id": session_id,
        "messages": messages,
        "ccd": ccd,
        "source": source.lower()
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


async def main():
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {args.config}")
    # Ensure patient simulator is set to patientpsi (since we're extracting CCDs)
    config["patient"]["simulator"] = "patientpsi"
    
    # Extract model name for output directory
    model_name = config.get("patient", {}).get("model", "unknown")
    clean_model_name = model_name.replace("/", "_").replace(":", "_")
    
    output_dir = Path(args.output_dir) / clean_model_name / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading {args.dataset} dataset...")
    df = load_real_dataset(args.dataset)
    
    if args.N:
        df = df.head(args.N)
    
    print(f"\n{'='*80}")
    print(f"Extracting CCDs from {len(df)} conversations")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*80}\n")
    
    # Prepare all data
    all_data = []
    for idx, row in df.iterrows():
        try:
            real_messages = row["messages"]
            source = row.get("source", "unknown")
            all_data.append((idx, real_messages, source))
        except Exception as e:
            print(f"[ERROR] Failed to load session {idx}: {e}")
            continue
    
    # Process in batches
    total_extracted = 0
    total_skipped = 0
    total_failed = 0
    total_batches = (len(all_data) + args.batch_size - 1) // args.batch_size
    
    for batch_num, batch_start in enumerate(range(0, len(all_data), args.batch_size), 1):
        batch_end = min(batch_start + args.batch_size, len(all_data))
        batch_data = all_data[batch_start:batch_end]
        
        print(f"\n{'='*80}")
        print(f"Batch {batch_num}/{total_batches} (sessions {batch_start}-{batch_end-1})")
        print(f"{'='*80}")
        
        # Filter out existing CCDs if requested
        indices_to_extract = []
        messages_to_extract = []
        batch_tasks = []
        
        for idx, real_messages, source in batch_data:
            if args.skip_existing and ccd_exists(output_dir, idx):
                print(f"  [SKIP] Session {idx} (CCD already exists)")
                total_skipped += 1
            else:
                indices_to_extract.append(idx)
                messages_to_extract.append(real_messages)
                batch_tasks.append((idx, real_messages, source))
        
        if not messages_to_extract:
            print("  All sessions in batch already extracted, skipping...")
            continue
        
        # Extract CCDs for this batch
        print(f"  Extracting {len(messages_to_extract)} CCDs...")
        try:
            ccds_and_profiles = generate_chain_batch(messages_to_extract, config)
        except Exception as e:
            print(f"  [ERROR] Batch extraction failed: {e}")
            print(f"  {traceback.format_exc()}")
            total_failed += len(messages_to_extract)
            continue
        
        # Save results
        for (idx, real_messages, source), (ccd, profile) in zip(batch_tasks, ccds_and_profiles):
            if ccd is None:
                print(f"  [FAIL] Session {idx}: CCD extraction returned None")
                total_failed += 1
            else:
                try:
                    save_ccd(output_dir, idx, real_messages, ccd, source)
                    print(f"  [OK] Session {idx}: CCD saved")
                    total_extracted += 1
                except Exception as e:
                    print(f"  [ERROR] Session {idx}: Failed to save - {e}")
                    total_failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*80}")
    print(f"  Total sessions: {len(all_data)}")
    print(f"  Extracted: {total_extracted}")
    print(f"  Skipped: {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Output directory: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Push to HuggingFace: python psibench/data_loader/push_ccds_hf.py {output_dir} username/repo-name")
    print(f"  2. Generate conversations: python psibench/generate_conversations.py --ccd-dir {output_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    asyncio.run(main())
