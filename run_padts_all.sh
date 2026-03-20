#!/bin/bash
set -o pipefail

#SBATCH -A aqu2_lab_gpu
#SBATCH -J padts_all
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=64GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --error=logs/padts_all-%J.err
#SBATCH --output=logs/padts_all-%J.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=shilligo@uci.edu

# ================================================================
# SLURM submission script for PaD-TS training (all datasets)
#
# Trains on: stock, energy, sine, fmri, mujoco, mhealth
# (sequentially on a single GPU)
#
# Usage:
#   sbatch run_padts_all.sh                                    # all 6 datasets
#   sbatch --export=DATASETS="stock;energy" run_padts_all.sh   # subset
#
# Environment variables:
#   DATASETS - SEMICOLON-separated dataset names to train
#              (default: stock;energy;sine;fmri;mujoco;mhealth)
#
# Notes:
#   - mhealth requires ./dataset/mhealth_data.csv to exist
#     (run preprocess_mhealth.py first)
#   - mujoco requires dm-control and mujoco packages
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
echo "PaD-TS Training - All Datasets"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"
echo "Working Directory: $(pwd)"
echo "DATASETS: ${DATASETS:-stock;energy;sine;fmri;mujoco;mhealth (default)}"
echo "=========================================="

# Default dataset order
DATASETS="${DATASETS:-stock;energy;sine;fmri;mujoco;mhealth}"

# Track overall exit code
OVERALL_EXIT=0

# Loop through datasets
IFS=';' read -ra DATASET_LIST <<< "$DATASETS"
for dataset in "${DATASET_LIST[@]}"; do

    echo ""
    echo "=========================================="
    echo "Training: $dataset"
    echo "Started:  $(date)"
    echo "=========================================="

    if [ "$dataset" == "mhealth" ]; then
        # mhealth uses standalone runner (run_mhealth.py)
        if [ ! -f ./dataset/mhealth_data.csv ]; then
            echo "WARNING: ./dataset/mhealth_data.csv not found, skipping mhealth."
            echo "Run: python preprocess_mhealth.py --input_dir /path/to/MHEALTHDATASET"
            continue
        fi
        python run_mhealth.py 2>&1 | tee results/padts_${dataset}_${SLURM_JOB_ID}.txt
    else
        python run.py -d "$dataset" 2>&1 | tee results/padts_${dataset}_${SLURM_JOB_ID}.txt
    fi

    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "WARNING: $dataset training exited with code $EXIT_CODE"
        OVERALL_EXIT=1
    else
        echo "$dataset training completed successfully."
    fi

    echo "Finished: $(date)"
    echo "=========================================="
done

# Log completion
echo ""
echo "=========================================="
echo "All training completed at: $(date)"
echo "Exit code: $OVERALL_EXIT"
echo "=========================================="

exit $OVERALL_EXIT
