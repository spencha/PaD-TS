#!/bin/bash

#SBATCH -A aqu2_lab_gpu
#SBATCH -J padts_stocks
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --error=logs/padts_stocks-%J.err
#SBATCH --output=logs/padts_stocks-%J.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=shilligo@uci.edu

# ================================================================
# SLURM submission script for PaD-TS training on the Stocks dataset
#
# Usage:
#   sbatch run_padts_stocks.sh
# ================================================================

# Load modules and activate conda environment
module purge
module load anaconda/2024.06
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate padts

# Change to project directory
cd ~/PaD-TS || { echo "ERROR: Could not cd to ~/PaD-TS"; exit 1; }

# Create output directories
mkdir -p logs results

# Log job information
echo "=========================================="
echo "PaD-TS Training - Stocks Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

python run.py -d stock 2>&1 | tee results/padts_stocks_${SLURM_JOB_ID}.txt

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
