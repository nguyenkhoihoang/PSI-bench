 Note that to evaluate synthetic data, because the messages are very long, and will parse up to 32 messages to model for evaluation in one go, when hosting judge should set
 `--max-model-len` to a large number, for e.g. 20000
```
python -m psibench.eval.ptc_classification --dataset esc --synthetic-dir data/synthetic/eeyore/hosted_vllm_openai_gpt-oss-120b/esc/ --compare --batch-size 10
```
Just synthetic:
```
python -m psibench.eval.ptc_classification \
  --synthetic-dir data/synthetic/eeyore/hosted_vllm_openai_gpt-oss-120b/esc/ \
  --batch-size 10 \
  --N 20 \
  --config configs/default.yaml
```
Add `--debug` to run with debug enabled