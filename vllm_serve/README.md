# vLLM Server Setup Guide

This guide explains how to set up and use the vLLM server for model inference.

## Prerequisites

SSH to the dt-login04 node which has CUDA 12.8 installed.
```bash
nvcc --version
```
Verify that CUDA version 12.8 is available.

## Installation

1. Create a new conda environment for vLLM:
   ```bash
   conda create -n vllm python=3.12 -y
   conda activate vllm
   ```

2. Install vLLM with CUDA 12.8 support:
   ```bash
   pip install --upgrade uv
   uv pip install vllm --torch-backend=cu128
   ```

3. Set the shared Hugging Face cache directory to save space:
   ```bash
   export HF_HOME=/work/hdd/bfjp/huggingface
   ```

## GPU Allocation

Request GPU resources using srun:
```bash
srun -A bfjp-delta-gpu --time=01:00:00 --nodes=1 --ntasks-per-node=1 --cpus-per-task=40 --partition=gpuH200x8 --gpus=1 --mem=300g --pty /bin/bash
```

- Use `--gpus=1` for a single H200 GPU (sufficient for `gpt-oss-20b` or `gpt-oss-120b`)
- Use `--gpus=2` for two H200 GPUs (allows for larger `max-num-seqs` values for higher throughput)
  - If using 2 GPUs, update `CUDA_VISIBLE_DEVICES=0,1` in `launch.sh`

## Usage

### Launching the Server

1. Start the vLLM server:
   ```bash
   bash vllm_serve/launch.sh
   ```

2. Check the logs to find the public endpoint. You should see something like:
   ```
   Public endpoint: 141.142.254.223:8000
   ```

3. Copy this endpoint URL and update api_base to "http://141.142.254.223:8000/v1" in `test.py`.

### Testing the Server

1. Open a new terminal and activate the psibench environment:
   ```bash
   conda activate psibench
   ```

2. Run the test script:
   ```bash
   python vllm_serve/test_gpt-oss.py
   ```

## Tested Models

The following models have been tested and verified to work with this setup:

### Reasoning LLMs
These models support reasoning with thinking process output:
- `openai/gpt-oss-120b`
- `openai/gpt-oss-20b`
- `Qwen/Qwen3-30B-A3B-Thinking-2507`

**Test scripts:**
- `vllm_serve/test_qwen3_reasoning.py` - Test Qwen3 reasoning model
- `vllm_serve/test_gpt-oss.py` - Test GPT-OSS models

### Standard LLMs
- `Qwen/Qwen3-30B-A3B-Instruct-2507`

**Test script:**
- `vllm_serve/test_qwen3_instruct.py` - Test Qwen3 instruct model

## Reasoning Models

For reasoning models, you can configure the reasoning effort level when making API calls to control the trade-off between response quality and inference time.

Example from `test.py`:
```python
response = litellm.completion(
    model="hosted_vllm/openai/gpt-oss-120b",
    messages=messages,
    api_base="http://141.142.254.223:8000/v1",
    reasoning_effort="medium",  # can be "low", "medium", "high"
    temperature=0.9,
    max_tokens=2000
)

# Access the reasoning content
content = response.choices[0].message.content
reasoning = response.choices[0].message.reasoning_content
```

Available effort levels: `"low"`, `"medium"`, `"high"`

## Notes

- The HF_HOME environment variable is set to `/work/hdd/bfjp/huggingface` to use a unified location and save disk space.
- Make sure to update the endpoint URL in `test.py` to match the public endpoint shown in your server logs.
