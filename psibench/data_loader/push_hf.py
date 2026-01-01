#!/usr/bin/env python
"""
Script to push synthetic PSI-bench data to HuggingFace Hub as Parquet files.

Usage:
    huggingface-cli login  # One time, with HF token
    python psibench/data_loader/push_hf.py <data_dir> <repo_id> [--version v1.0] [--private]
    
Example:
    python psibench/data_loader/push_hf.py /work/hdd/bfjp/data/synthetic/test/ hknguyen20/psibench-synthetic --version v1.0
"""
import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import datasets
# Try to import huggingface_hub
try:
    from huggingface_hub import HfApi, RepoUrl
except ImportError:
    print("[ERROR] huggingface_hub not installed. Install with: pip install huggingface_hub")
    sys.exit(1)

try:
    from datasets import load_dataset, Dataset
except ImportError:
    print("[ERROR] datasets not installed. Install with: pip install datasets")
    sys.exit(1)


_DESCRIPTION = "Synthetic psychotherapy conversations generated with PatientPSI and Roleplaydoh using different LLMs"

class PSIBenchSynthetic(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="all", description="All synthetic conversations"),
        datasets.BuilderConfig(name="roleplaydoh", description="Roleplaydoh only"),
        datasets.BuilderConfig(name="patientpsi", description="PatientPSI only"),
    ]
    
    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({
                "session_id": datasets.Value("int32"),
                "messages": datasets.Sequence({
                    "role": datasets.Value("string"),
                    "content": datasets.Value("string"),
                }),
                "profile": datasets.Value("string"),
                "psi": datasets.Value("string"),
                "backend_llm": datasets.Value("string"),
                "dataset": datasets.Value("string"),
                "num_patient_turns": datasets.Value("int32"),
                "ccd": {
                    "life_history": datasets.Value("string"),
                    "core_beliefs": datasets.Value("string"),
                    "core_belief_description": datasets.Value("string"),
                    "intermediate_beliefs": datasets.Value("string"),
                    "intermediate_beliefs_during_depression": datasets.Value("string"),
                    "coping_strategies": datasets.Value("string"),
                    "cognitive_models": datasets.Sequence({
                        "situation": datasets.Value("string"),
                        "automatic_thoughts": datasets.Value("string"),
                        "emotion": datasets.Value("string"),
                        "behavior": datasets.Value("string"),
                    }),
                },
            })
        )

    
    def _split_generators(self, dl_manager):
        return [datasets.SplitGenerator(name=datasets.Split.TRAIN)]
    
    def _generate_examples(self, **kwargs):
        data_dir = Path("data")
        idx = 0
        
        for psi_dir in sorted(data_dir.iterdir()):
            if not psi_dir.is_dir():
                continue
            
            psi_name = psi_dir.name
            if self.config.name != "all" and psi_name != self.config.name:
                continue
            
            for backend_dir in sorted(psi_dir.iterdir()):
                if not backend_dir.is_dir():
                    continue
                
                backend_name = backend_dir.name
                
                for dataset_dir in sorted(backend_dir.iterdir()):
                    if not dataset_dir.is_dir():
                        continue
                    
                    dataset_name = dataset_dir.name
                    
                    for session_file in sorted(dataset_dir.glob('session_*.json')):
                        with open(session_file) as f:
                            data = json.load(f)
                            yield idx, {
                                "session_id": data.get("session_id", idx),
                                "messages": data.get("messages", []),
                                "profile": data.get("profile", ""),
                                "psi": psi_name,
                                "backend_llm": backend_name,
                                "dataset": dataset_name,
                                "num_patient_turns": data.get("num_patient_turns", 0),
                                "ccd": data.get("ccd", {}),
                            }
                            idx += 1
                            
def validate_data_structure(data_dir: Path) -> bool:
    """Validate that data directory has the expected structure."""
    if not data_dir.exists():
        print(f"[ERROR] Data directory not found: {data_dir}")
        return False
    
    # Check for at least one PSI simulator
    psi_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    if not psi_dirs:
        print(f"[ERROR] No PSI simulator directories found in {data_dir}")
        return False
    
    # Check for at least one backend
    backend_found = False
    for psi_dir in psi_dirs:
        backend_dirs = [d for d in psi_dir.iterdir() if d.is_dir()]
        if backend_dirs:
            backend_found = True
            # Check for datasets
            for backend_dir in backend_dirs:
                dataset_dirs = [d for d in backend_dir.iterdir() if d.is_dir()]
                for dataset_dir in dataset_dirs:
                    sessions = list(dataset_dir.glob('session_*.json'))
                    if sessions:
                        print(f"[OK] Found {len(sessions)} sessions in {psi_dir.name}/{backend_dir.name}/{dataset_dir.name}")
                        return True
    
    if not backend_found:
        print(f"[ERROR] No backend directories found in {data_dir}")
        return False
    
    return False


def create_dataset_script(output_dir: Path, data_dir: Path):
    """Create the dataset loading script (dataset_infos.json and README)."""
    
    # Create README
    readme_content = f"""# PSI-Bench Synthetic Dataset

Synthetic psychotherapy conversations generated with patient simulators.
```
"""
    
    readme_path = output_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    print(f"[OK] Created {readme_path}")
    
    # Create dataset_infos.json
    dataset_infos = {
        "all": {"description": "All synthetic conversations", "citation": "", "homepage": "", "license": "", "features": None},
        "roleplaydoh": {"description": "Roleplaydoh only", "citation": "", "homepage": "", "license": "", "features": None},
        "patientpsi": {"description": "PatientPSI only", "citation": "", "homepage": "", "license": "", "features": None},
    }
    
    infos_path = output_dir / "dataset_infos.json"
    with open(infos_path, 'w') as f:
        json.dump(dataset_infos, f, indent=2)
    print(f"[OK] Created {infos_path}")


def load_all_data_to_dataframe(data_dir: Path) -> pd.DataFrame:
    """Load all synthetic data from nested directory structure into a DataFrame.
    
    Special handling:
    - For patientpsi: uses 'ccd' as 'profile' (since ccd is the profile equivalent)
    - For others: keeps 'profile' field as-is
    - All dict/list values converted to readable JSON strings (no unicode escaping)
    """
    all_sessions = []
    
    for psi_dir in sorted(data_dir.iterdir()):
        if not psi_dir.is_dir():
            continue
        
        psi_name = psi_dir.name
        
        for backend_dir in sorted(psi_dir.iterdir()):
            if not backend_dir.is_dir():
                continue
            
            backend_name = backend_dir.name
            
            for dataset_dir in sorted(backend_dir.iterdir()):
                if not dataset_dir.is_dir():
                    continue
                
                dataset_name = dataset_dir.name
                
                for session_file in sorted(dataset_dir.glob('session_*.json')):
                    try:
                        with open(session_file, 'r', encoding='utf-8') as f:
                            session_data = json.load(f)
                            
                            # Convert all complex types to JSON strings (with readable unicode)
                            for key, value in list(session_data.items()):
                                if isinstance(value, (dict, list)):
                                    # ensure_ascii=False keeps unicode readable (no \uXXXX escaping)
                                    session_data[key] = json.dumps(value, ensure_ascii=False, indent=2)
                            
                            # Special handling for patientpsi: use ccd as profile
                            if psi_name.lower() == 'patientpsi':
                                if 'ccd' in session_data:
                                    # For patientpsi, rename ccd to profile
                                    session_data['profile'] = session_data.pop('ccd')
                                # Remove any other profile field if it exists
                                if 'profile_old' in session_data:
                                    del session_data['profile_old']
                            
                            # Add metadata
                            session_data['psi'] = psi_name
                            session_data['backend_llm'] = backend_name
                            session_data['dataset'] = dataset_name
                            all_sessions.append(session_data)
                    except Exception as e:
                        print(f"[WARNING] Failed to load {session_file}: {e}")
                        continue
    
    return pd.DataFrame(all_sessions)


def push_hf(data_dir: Path, repo_id: str, version: str = "v1.0", private: bool = False, token: str = None):
    """Push dataset to HuggingFace Hub as Parquet files."""
    
    print(f"\n[INFO] Pushing dataset to HuggingFace as Parquet...")
    print(f"  Repo ID: {repo_id}")
    print(f"  Version: {version}")
    print(f"  Private: {private}")
    print(f"  Data Directory: {data_dir}")
    
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
        
        # Load all data into DataFrame
        print(f"\n[STEP 2] Loading and converting data to DataFrame...")
        df = load_all_data_to_dataframe(data_dir)
        print(f"[OK] Loaded {len(df)} sessions")
        print(f"  PSI simulators: {sorted(df['psi'].unique())}")
        print(f"  Backend LLMs: {sorted(df['backend_llm'].unique())}")
        print(f"  Datasets: {sorted(df['dataset'].unique())}")
        
        # Create temp directory
        temp_dir = Path("/tmp/psibench_upload")
        temp_dir.mkdir(exist_ok=True)
        
        # Convert to HuggingFace Dataset and save as parquet
        print(f"\n[STEP 3] Converting to HuggingFace Dataset and saving as Parquet...")
        dataset = Dataset.from_pandas(df)
        
        # Save train split as parquet
        train_parquet = temp_dir / "train.parquet"
        dataset.to_parquet(str(train_parquet))
        print(f"[OK] Created train.parquet ({train_parquet.stat().st_size / 1024 / 1024:.2f} MB)")
        
        # Create README
        create_dataset_script(temp_dir, data_dir)
        
        # Upload README
        print(f"\n[STEP 4] Uploading files to HuggingFace...")
        api.upload_file(
            path_or_fileobj=temp_dir / "README.md",
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
        
        # Create a version tag (optional)
        if version:
            print(f"\n[STEP 5] Creating version tag: {version}")
            api.create_tag(
                repo_id=repo_id,
                tag=version,
                repo_type="dataset",
            )
            print(f"[OK] Created tag: {version}")
        
        print(f"\n[SUCCESS] Dataset pushed to HuggingFace!")
        print(f"  Repository: https://huggingface.co/datasets/{repo_id}")
        print(f"\n  Load with:")
        print(f"    from datasets import load_dataset")
        print(f"    ds = load_dataset('{repo_id}', split='train')")
        if version:
            print(f"    # Or specific version:")
            print(f"    ds = load_dataset('{repo_id}', split='train', revision='{version}')")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to push dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Push PSI-bench synthetic data to HuggingFace Hub"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Path to data directory (e.g., /work/hdd/bfjp/data/synthetic/test/)"
    )
    parser.add_argument(
        "repo_id",
        type=str,
        help="HuggingFace repo ID (e.g., username/psibench-synthetic)"
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
        help="Make dataset private"
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="HuggingFace API token (will use HF_TOKEN env var if not provided)"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir.strip())
    
    # Validate structure
    print(f"[INFO] Validating data structure...")
    if not validate_data_structure(data_dir):
        print(f"[ERROR] Invalid data structure")
        sys.exit(1)
    
    # Push to HF
    success = push_hf(
        data_dir=data_dir,
        repo_id=args.repo_id,
        version=args.version,
        private=args.private,
        token=args.token
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
