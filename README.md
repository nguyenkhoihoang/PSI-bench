

### 1. Install dependencies
```
conda create -n psibench python=3.11
conda activate psibench
pip install -e .
```

Create .env file inside psibench/ and put in your env variables OPENAI_API_KEY, OPENAI_BASE_URL, HF_TOKEN

### 2. Generate synthetic conversations
```
python -m psibench.generate_conversations --dataset esc
```

Note: If have error like `ModuleNotFoundError: No module named 'data_loader'`
export PYTHONPATH to your repo/psi-bench, e.g.
```
export PYTHONPATH=/u/nhoang1/PSI-bench/psibench/
```

### 3. Compare real and synthetic convo
Specify some index of session u want to compare. e.g.
```
python -m psibench.eval.read_compare_convo 3
```
<!-- (Later: haven't checked since delta GPU been down lately)
Run eval comparison on delta (may need to edit your account in sbatch script) 
bash eval_sim.sl -->