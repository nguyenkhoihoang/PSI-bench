#!/bin/bash
conda init
conda activate psibench

cd /scratch/bfjp/jenny0830/PSI-bench

export PYTHONPATH=/scratch/bfjp/jenny0830/PSI-bench/psibench
python -m psibench.eval.perplexity

echo "Perplexity calculation complete!"