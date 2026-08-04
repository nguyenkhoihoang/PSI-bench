#!/usr/bin/env python
"""
Script to push extracted CCDs to HuggingFace Hub as Parquet files.

Usage:
    huggingface-cli login  # One time, with HF token
    python psibench/data_loader/push_ccds_hf.py <ccd_dir> <repo_id> [--version v1.0] [--private]
    
Example:
    python psibench/data_loader/push_ccds_hf.py \
    data/ccds/hosted_vllm_openai_gpt-oss-120b/all \
    hknguyen20/psibench-ccds \
    --version v1.0 \
    --private
"""
import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd

# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi
except ImportError:
    print("[ERROR] huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

try:
    from datasets import Dataset
except ImportError:
    print("[ERROR] datasets not installed. Install with: pip install datasets")
    sys.exit(1)


def validate_ccd_directory(ccd_dir: Path) -> bool:
    """Validate that CCD directory has the expected structure."""
    if not ccd_dir.exists():
        print(f"[ERROR] CCD directory not found: {ccd_dir}")
        return False
    
    # Check for at least one CCD file
    ccd_files = list(ccd_dir.glob('ccd_*.json'))
    if not ccd_files:
        print(f"[ERROR] No CCD files (ccd_*.json) found in {ccd_dir}")
        return False
    
    print(f"[OK] Found {len(ccd_files)} CCD files in {ccd_dir}")
    return True


def load_ccds_to_dataframe(ccd_dir: Path) -> pd.DataFrame:
    """Load all CCDs from directory into a DataFrame.
    
    Special handling:
    - All dict/list values converted to readable JSON strings (no unicode escaping)
    """
    all_ccds = []
    
    for ccd_file in sorted(ccd_dir.glob('ccd_*.json')):
        try:
            with open(ccd_file, 'r', encoding='utf-8') as f:
                ccd_data = json.load(f)
                
                # Ensure session_id is present
                if 'session_id' not in ccd_data:
                    try:
                        ccd_data['session_id'] = int(ccd_file.stem.split('ccd_')[-1])
                    except Exception:
                        print(f"[WARNING] Cannot extract session_id from {ccd_file}")
                        continue
                
                # Convert complex types to JSON strings (readable unicode)
                for key, value in list(ccd_data.items()):
                    if isinstance(value, (dict, list)):
                        ccd_data[key] = json.dumps(value, ensure_ascii=False, indent=2)
                
                all_ccds.append(ccd_data)
        except Exception as e:
            print(f"[WARNING] Failed to load {ccd_file}: {e}")
            continue
    
    return pd.DataFrame(all_ccds)


def create_readme(ccd_dir: Path, dataset_name: str) -> str:
    """Create README content for the CCD dataset."""
    
    readme_content = f"""# PSI-Bench CCDs Dataset

Extracted Cognitive Conceptualization Diagrams (CCDs) from real therapy conversations.

## Dataset Structure

Each row contains:
- `session_id`: Unique identifier for the session
- `messages`: Original therapy conversation (list of turn objects with role/content)
- `ccd`: Extracted cognitive conceptualization diagram with:
  - `life_history`: Patient's background and significant life events
  - `core_beliefs`: Fundamental beliefs (Helpless/Unlovable/Worthless)
  - `core_belief_description`: Detailed core belief descriptions
  - `intermediate_beliefs`: Attitudes, rules, and assumptions
  - `intermediate_beliefs_during_depression`: Beliefs active during depression
  - `coping_strategies`: Methods used to deal with stress
  - `cognitive_models`: List of situation-thought-emotion-behavior patterns
- `source`: Dataset source (esc/hope/annomi)

## Usage

```python
from datasets import load_dataset

# Load full dataset
ds = load_dataset('{dataset_name}', split='train')

# Access a CCD
ccd = json.loads(ds[0]['ccd'])
print(ccd['core_beliefs'])
```

## Generation

Generated on using patient-psi extraction pipeline using gpt-oss-120b with temperature 1.0.

See: https://github.com/nguyenkhoihoang/PSI-bench
"""
    
    return readme_content


def push_ccds_hf(
    ccd_dir: Path,
    repo_id: str,
    version: str = "v1.0",
    private: bool = False,
    token: str = None
):
    """Push CCD dataset to HuggingFace Hub as Parquet files."""
    
    print(f"\n[INFO] Pushing CCDs to HuggingFace as Parquet...")
    print(f"  Repo ID: {repo_id}")
    print(f"  Version: {version}")
    print(f"  Private: {private}")
    print(f"  CCD Directory: {ccd_dir}")
    
    api = HfApi(token=token)
    
    try:
        # Create repo if it doesn't exist
        print(f"\n[STEP 1] Creating/checking HuggingFace repo...")
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"[OK] Repo ready: {repo_url}")
        
        # Load all CCDs into DataFrame
        print(f"\n[STEP 2] Loading and converting CCDs to DataFrame...")
        df = load_ccds_to_dataframe(ccd_dir)
        
        if df.empty:
            print(f"[ERROR] No CCDs loaded from {ccd_dir}")
            return False
        
        print(f"[OK] Loaded {len(df)} CCDs")
        if 'source' in df.columns:
            print(f"  Sources: {sorted(df['source'].unique())}")
        
        # Create temp directory
        temp_dir = Path("/tmp/psibench_ccds_upload")
        temp_dir.mkdir(exist_ok=True)
        
        # Convert to HuggingFace Dataset and save as parquet
        print(f"\n[STEP 3] Converting to HuggingFace Dataset and saving as Parquet...")
        dataset = Dataset.from_pandas(df)
        
        # Save train split as parquet
        train_parquet = temp_dir / "train.parquet"
        dataset.to_parquet(str(train_parquet))
        print(f"[OK] Created train.parquet ({train_parquet.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Create README
        readme_content = create_readme(ccd_dir, repo_id)
        readme_path = temp_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"[OK] Created README.md")
        
        # Upload README
        print(f"\n[STEP 4] Uploading files to HuggingFace...")
        api.upload_file(
            path_or_fileobj=str(readme_path),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"[OK] Uploaded README.md")
        
        # Upload parquet file
        api.upload_file(
            path_or_fileobj=str(train_parquet),
            path_in_repo="train.parquet",
            repo_id=repo_id,
            repo_type="dataset",
        )
        print(f"[OK] Uploaded train.parquet")
        
        # Create a version tag
        if version:
            print(f"\n[STEP 5] Creating version tag: {version}")
            try:
                api.create_tag(
                    repo_id=repo_id,
                    tag=version,
                    repo_type="dataset",
                )
                print(f"[OK] Created tag: {version}")
            except Exception as e:
                print(f"[WARNING] Could not create tag (may already exist): {e}")
        
        print(f"\n{'='*80}")
        print(f"SUCCESS: CCDs pushed to HuggingFace!")
        print(f"{'='*80}")
        print(f"Repository: https://huggingface.co/datasets/{repo_id}")
        print(f"\nLoad with:")
        print(f"  from datasets import load_dataset")
        print(f"  ds = load_dataset('{repo_id}', split='train')")
        if version:
            print(f"  # Or specific version:")
            print(f"  ds = load_dataset('{repo_id}', split='train', revision='{version}')")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to push dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Push extracted CCDs to HuggingFace Hub (public by default)"
    )
    parser.add_argument(
        "ccd_dir",
        type=str,
        help="Path to CCD directory (e.g., data/ccds/gpt-4o-mini/esc/)"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repo ID (e.g., username/psibench-ccds)"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Version tag (default: v1.0)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make dataset private (default is public)"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (will use HF_TOKEN env var if not provided)"
    )
    
    args = parser.parse_args()
    
    ccd_dir = Path(args.ccd_dir.strip())
    
    # Validate structure
    print(f"[INFO] Validating CCD directory...")
    if not validate_ccd_directory(ccd_dir):
        print(f"[ERROR] Invalid CCD directory structure")
        sys.exit(1)
    
    # Push to HF
    success = push_ccds_hf(
        ccd_dir=ccd_dir,
        repo_id=args.repo_id,
        version=args.version,
        private=args.private,
        token=args.token
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
