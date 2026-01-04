#!/bin/bash

# Model name from Hugging Face
# Instruct LLM
# meta-llama/Llama-3.1-8B-Instruct

# Reasoning LLM
# openai/gpt-oss-120b
# openai/gpt-oss-20b
# Qwen/Qwen3-30B-A3B-Thinking-2507

# LLM
# Qwen/Qwen3-30B-A3B-Instruct-2507

MODEL_NAME="openai/gpt-oss-120b"

# Specify which GPUs to use (0 means first GPU, use "0,1,2,3" for multiple GPUs)
export CUDA_VISIBLE_DEVICES=0,1,2,3
# export HF_HOME=/work/hdd/bfjp/huggingface

if [ -f "psibench/.env" ]; then
    echo "Loading environment from psibench/.env"
    export $(grep -v '^#' psibench/.env | xargs)
fi

# Verify environment variables
echo "HF_TOKEN is set: ${HF_TOKEN:+yes}"
echo "HF HOME: $HF_HOME"
# Automatically count the number of GPUs based on CUDA_VISIBLE_DEVICES
GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Port for the vLLM server to listen on
VLLM_SERVER_PORT=9000

# Directory to store log files
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi

# Log file path (replaces / with _ in model name for valid filename)
LOG_PATH="$LOG_DIR/${MODEL_NAME//\//_}.log"
echo "=== Starting vLLM serve on $(date) ===" > $LOG_PATH
echo "Public endpoint: $(curl -s ifconfig.me):$VLLM_SERVER_PORT" >> $LOG_PATH

# Launch vLLM serve.
#   --tensor-parallel-size : number of GPUs used for tensor parallelism
#   --max-num-seqs         : max concurrent sequences
#   --async-scheduling     : enable async scheduling for throughput
#   --port                 : HTTP port for the server
#   --max-model-len        : max context length
#   --gpu-memory-utilization : fraction of GPU memory to allocate
# --disable-custom-all-reduce: reduce speed but improve stability
vllm serve $MODEL_NAME \
    --tensor-parallel-size $GPU_NUM \
    --max-num-seqs 16 \
    --async-scheduling \
    --port $VLLM_SERVER_PORT \
    --max-model-len 20000 \
    --gpu-memory-utilization 0.9 \
    >> $LOG_PATH 2>&1
