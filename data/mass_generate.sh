#!/bin/bash

# Define datasets and psi values
datasets=("esc" "hope" "annomi")
psis=("eeyore" "roleplaydoh" "patientpsi")

# Define the batch size (max possible) and other parameters
config="configs/llama-3.3-70b-instruct.yaml"
output_dir="/work/hdd/bfjp/data/synthetic"
# Get the directory where this script is located
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
# Directory to store log files
LOG_DIR="$SCRIPT_DIR/logs"

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    mkdir -p "$LOG_DIR"
fi
# Loop through each combination of dataset and psi
for dataset in "${datasets[@]}"; do
  for psi in "${psis[@]}"; do
    echo "Running for dataset: $dataset, psi: $psi"
    # nohup python -m psibench.generate_conversations 
    nohup python -m psibench.generate_next_turn \
      --dataset "$dataset" \
      --psi "$psi" \
      --batch-size 32 \
      --config "$config" \
      --output-dir "$output_dir" \
      > "$LOG_DIR/${dataset}_${psi}.log" 2>&1 &
    echo "background process ID: $!"
  done
done

echo "All processes started in the background."
