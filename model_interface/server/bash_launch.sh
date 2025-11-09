################### Text-Only Models ###################
# Qwen/Qwen2.5-3B-Instruct

export CUDA_VISIBLE_DEVICES=0
GPU_NUM=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

VLLM_SERVER_PORT=8000
MODEL_NAME="Qwen/Qwen2.5-3B-Instruct"

LOG_PATH="${SCRIPT_DIR}/logs/${MODEL_NAME//\//_}.log"
mkdir -p ${SCRIPT_DIR}/logs
echo "=== Starting vLLM serve on $(date) ===" > $LOG_PATH
echo "Public endpoint: $(curl -s ifconfig.me):$VLLM_SERVER_PORT" >> $LOG_PATH

vllm serve $MODEL_NAME \
    --dtype bfloat16 \
    --api-key token-abc123 \
    --task generate \
    --tensor-parallel-size $GPU_NUM \
    --port $VLLM_SERVER_PORT \
    --max-num-seqs 128 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.95 \
    >> $LOG_PATH 2>&1