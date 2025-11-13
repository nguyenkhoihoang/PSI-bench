

### 1. Install dependencies
```
conda create -n psibench python=3.11
conda activate psibench
pip install -e .
```

Create .env file inside psibench/ and put in your env variables OPENAI_API_KEY, OPENAI_BASE_URL, HF_TOKEN

### 2. Generate synthetic conversations

To generate synthetic data, specify dataset source `--dataset`  ('esc' (default), 'hope', 'annomi') and type of patient simulator you want `--psi` ('eeyore' (default), 'patientpsi', 'roleplaydoh')

AI patient - AI therapist convo:

```
python -m psibench.generate_conversations --dataset esc
```

AI patient respond given previous history (`--turn_idx` default=0)
```
python -m psibench.generate_next_turn --dataset --turn_idx 3
```


Note: If have error like `ModuleNotFoundError: No module named 'data_loader'`
export PYTHONPATH to your repo/psi-bench, e.g.
```
export PYTHONPATH=/u/nhoang1/PSI-bench/psibench/
```
The synthetic data will be saved in data/synthetic/{psi}/{dataset}

For e.g. `data/synthetic/eeyore/esc/`

### 3. Evaluation

#### Compare real and synthetic convo
Specify some index of session u want to compare. e.g. (Look for idx in synthetic data folder)

Note: Only ESC has real situation from original dataset. Others just have situation loaded from eeyore.

```
python -m psibench.eval.read_compare_convo 3
python -m psibench.eval.read_compare_convo --dataset annomi 39
```
#### P,T,C Classification
```
python -m psibench.eval.ptc_classification \
    --dataset esc \
    --synthetic-dir data/synthetic/roleplaydoh/esc \
    --compare \
```
<!-- (Later: haven't checked since delta GPU been down lately)
Run eval comparison on delta (may need to edit your account in sbatch script) 
bash eval_sim.sl -->