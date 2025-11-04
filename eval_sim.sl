#!/bin/bash
#SBATCH --job-name=eval-sim
#SBATCH --account=bfjp-delta-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --partition=gpuA40x4

# Load required modules
module load cuda

# Initialize conda
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda init bash
source ~/.bashrc

# Activate conda environment
conda activate psibench

# Print environment info
which python
python --version
echo "Conda environment: $CONDA_DEFAULT_ENV"

# Run the similarity analysis
echo "Starting similarity analysis..."

# Compare real and synthetic
python -m psibench.eval.situation_convo_similarity --mode compare --data-dir data/synthetic/esc

# # Analyze just real data
# python -m psibench.eval.situation_convo_similarity --mode analyze --data-type real

# # Analyze synthetic data
# python -m psibench.eval.situation_convo_similarity --mode analyze --data-type synthetic --data-dir data/synthetic/esc