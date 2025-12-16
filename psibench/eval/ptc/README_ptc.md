 Note that to evaluate synthetic data, because the messages are very long, and will parse up to 32 messages to model for evaluation in one go, when hosting judge should set
 `--max-model-len` to a large number, for e.g. 20000

single-turn mode
```
python -m psibench.eval.ptc.ptc_classification \
  --dataset esc \
  --synthetic-dir /work/hdd/bfjp/data/synthetic/test/patientpsi/hosted_vllm_openai_gpt-oss-120b/esc \
  --compare \
  --batch-size 10 \
  --single-turn \
  --num-messages 6 \
  --exact-turns 12 \
  --config configs/default.yaml
```
full conversation mode
```
python -m psibench.eval.ptc.ptc_classification \
  --dataset hope \
  --synthetic-dir /work/hdd/bfjp/data/synthetic/test/patientpsi/hosted_vllm_openai_gpt-oss-120b/hope \
  --compare \
  --batch-size 10 \
  --config configs/default.yaml
```

Just synthetic:
```
python -m psibench.eval.ptc.ptc_classification \
  --synthetic-dir data/synthetic/eeyore/hosted_vllm_openai_gpt-oss-120b/esc/ \
  --batch-size 10 \
  --N 20 \
  --config configs/default.yaml
```
Add `--debug` to run with debug enabled