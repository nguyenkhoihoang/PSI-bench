

### 1. Install dependencies
```
conda create -n psibench python=3.11
conda activate psibench
pip install -e .
```

Create .env file inside psibench/ and put in your env variables OPENAI_API_KEY, OPENAI_BASE_URL, HF_TOKEN

If you want to use model_interface, install the dependencies
```
conda create -n psibench python=3.11
conda activate psibench
bash model_interface/install.sh
pip install -e .
```

#### Using the unified model_interface backend
- Edit `configs/default.yaml` and set `patient.backend` / `therapist.backend` to `model_interface`.
- Provide a config file path via `patient.model_interface_config`, `therapist.model_interface_config`, or the shared `model_interface.config_path`.
- Config files under `model_interface/configs/**` describe OpenAI, Google, vLLM server, or offline vLLM models; feel free to copy and customize them for other open-source checkpoints.
- All calls will now be routed through the unified interface, which batches requests with `num_workers` and lets you point PSI-bench to any provider that supports the OpenAI chat schema.


### 2. Generate synthetic conversations

To generate synthetic data, specify dataset source `--dataset`  ('esc' (default), 'hope', 'annomi') and type of patient simulator you want `--psi` ('eeyore' (default), 'patientpsi', 'roleplaydoh')

AI patient - AI therapist convo:

```
python -m psibench.generate_conversations --dataset esc --concurrency 8
```
> When both `patient.backend` and `therapist.backend` are set to `model_interface`, `--concurrency` controls how many sessions advance in lockstep. Each turn batches all active conversations into a single `model_interface.generate()` call, so increase it to saturate your provider.

AI patient respond given previous history (`--turn_idx` default=0)
```
python -m psibench.generate_next_turn --dataset esc --turn_idx 3 --concurrency 8
```


Note: If have error like `ModuleNotFoundError: No module named 'data_loader'`
export PYTHONPATH to your repo/psi-bench, e.g.
```
export PYTHONPATH=/u/nhoang1/PSI-bench/psibench/
```
The synthetic data will be saved in data/synthetic/{psi}/{dataset}

For e.g. `data/synthetic/eeyore/esc/`

### 3. Compare real and synthetic convo
Specify some index of session u want to compare. e.g. (Look for idx in synthetic data folder)

Note: Only ESC has real situation from original dataset. Others just have situation loaded from eeyore.

```
python -m psibench.eval.read_compare_convo 3
python -m psibench.eval.read_compare_convo --dataset annomi 39
```
<!-- (Later: haven't checked since delta GPU been down lately)
Run eval comparison on delta (may need to edit your account in sbatch script) 
bash eval_sim.sl -->
