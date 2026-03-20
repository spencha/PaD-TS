#!/bin/bash
#SBATCH -A aqu2_lab_gpu
#SBATCH -J padts_mhealth
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --error=logs/padts_mhealth-%J.err
#SBATCH --output=logs/padts_mhealth-%J.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=shilligo@uci.edu

set -o pipefail

# ================================================================
# SLURM submission script for PaD-TS training on the MHEALTH dataset
#
# Prerequisites:
#   1. Download MHEALTH dataset from UCI ML Repository
#   2. Run: python preprocess_mhealth.py --input_dir /path/to/MHEALTHDATASET
#      This creates ./dataset/mhealth_data.csv (23 sensor channels)
#
# Usage:
#   sbatch run_padts_mhealth.sh
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

# Verify data exists
if [ ! -f ./dataset/mhealth_data.csv ]; then
    echo "ERROR: ./dataset/mhealth_data.csv not found."
    echo "Run: python preprocess_mhealth.py --input_dir /path/to/MHEALTHDATASET"
    exit 1
fi

# Log job information
echo "=========================================="
echo "PaD-TS Training - MHEALTH Dataset"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Uses run_mhealth.py (standalone runner) so run.py is unmodified
python run_mhealth.py 2>&1 | tee results/padts_mhealth_${SLURM_JOB_ID}.txt

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Training completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

exit $EXIT_CODE
