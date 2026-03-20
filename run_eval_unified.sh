#!/bin/bash
#SBATCH -A aqu2_lab_gpu
#SBATCH -J eval_unified
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --error=logs/eval_unified-%J.err
#SBATCH --output=logs/eval_unified-%J.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=shilligo@uci.edu

set -o pipefail

# ================================================================
# Unified evaluation: run all 6 metrics on PaD-TS and Diffusion-TS
# outputs for direct comparison.
#
# Prerequisites:
#   - PaD-TS training completed (outputs in ./OUTPUT/)
#   - Diffusion-TS training completed (outputs in ~/Diffusion-TS/OUTPUT/)
#   - Both repos cloned on HPC3
#
# Usage:
#   sbatch run_eval_unified.sh
# ================================================================

# Load modules and activate conda environment
module purge
module load anaconda/2024.06
eval "$(conda shell.bash hook)"
conda activate padts

# Change to project directory
cd ~/PaD-TS || { echo "ERROR: Could not cd to ~/PaD-TS"; exit 1; }

mkdir -p logs results/eval

echo "=========================================="
echo "Unified Evaluation"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Time: $(date)"
echo "=========================================="

OVERALL_EXIT=0

# ── Stock dataset ──────────────────────────────────────────────

REAL="./OUTPUT/samples/stock_norm_truth_24_train.npy"

if [ -f "$REAL" ]; then
    # PaD-TS
    PADTS_FAKE="./OUTPUT/stock_24/ddpm_fake_stock_24.npy"
    if [ -f "$PADTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating PaD-TS on Stock"
        python eval_unified.py \
            --fake "$PADTS_FAKE" --real "$REAL" \
            --name stock --method padts \
            --output results/eval/stock_padts.json \
            2>&1 | tee results/eval/stock_padts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    else
        echo "SKIP: PaD-TS stock output not found at $PADTS_FAKE"
    fi

    # Diffusion-TS
    DIFFTS_FAKE="$HOME/Diffusion-TS/OUTPUT/stocks/ddpm_fake_stocks.npy"
    if [ -f "$DIFFTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating Diffusion-TS on Stock"
        python eval_unified.py \
            --fake "$DIFFTS_FAKE" --real "$REAL" \
            --name stock --method diffts \
            --fake_range original \
            --scaler_data ./dataset/stock_data.csv \
            --output results/eval/stock_diffts.json \
            2>&1 | tee results/eval/stock_diffts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    else
        echo "SKIP: Diffusion-TS stock output not found at $DIFFTS_FAKE"
    fi
else
    echo "SKIP: Stock ground truth not found. Run PaD-TS training first to generate it."
fi

# ── Energy dataset ─────────────────────────────────────────────

REAL="./OUTPUT/samples/energy_norm_truth_24_train.npy"

if [ -f "$REAL" ]; then
    PADTS_FAKE="./OUTPUT/energy_24/ddpm_fake_energy_24.npy"
    if [ -f "$PADTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating PaD-TS on Energy"
        python eval_unified.py \
            --fake "$PADTS_FAKE" --real "$REAL" \
            --name energy --method padts \
            --output results/eval/energy_padts.json \
            2>&1 | tee results/eval/energy_padts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi

    DIFFTS_FAKE="$HOME/Diffusion-TS/OUTPUT/energy/ddpm_fake_energy.npy"
    if [ -f "$DIFFTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating Diffusion-TS on Energy"
        python eval_unified.py \
            --fake "$DIFFTS_FAKE" --real "$REAL" \
            --name energy --method diffts \
            --fake_range original \
            --scaler_data ./dataset/energy_data.csv \
            --output results/eval/energy_diffts.json \
            2>&1 | tee results/eval/energy_diffts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi
else
    echo "SKIP: Energy ground truth not found."
fi

# ── Sine dataset ───────────────────────────────────────────────

REAL="./OUTPUT/samples/sine_ground_truth_24_train.npy"

if [ -f "$REAL" ]; then
    PADTS_FAKE="./OUTPUT/sine_24/ddpm_fake_sine_24.npy"
    if [ -f "$PADTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating PaD-TS on Sine"
        python eval_unified.py \
            --fake "$PADTS_FAKE" --real "$REAL" \
            --name sine --method padts \
            --output results/eval/sine_padts.json \
            2>&1 | tee results/eval/sine_padts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi

    DIFFTS_FAKE="$HOME/Diffusion-TS/OUTPUT/sines/ddpm_fake_sines.npy"
    if [ -f "$DIFFTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating Diffusion-TS on Sine"
        python eval_unified.py \
            --fake "$DIFFTS_FAKE" --real "$REAL" \
            --name sine --method diffts \
            --fake_range zero1 \
            --output results/eval/sine_diffts.json \
            2>&1 | tee results/eval/sine_diffts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi
else
    echo "SKIP: Sine ground truth not found."
fi

# ── fMRI dataset ───────────────────────────────────────────────

REAL="./OUTPUT/samples/fmri_norm_truth_24_train.npy"

if [ -f "$REAL" ]; then
    PADTS_FAKE="./OUTPUT/fmri_24/ddpm_fake_fmri_24.npy"
    if [ -f "$PADTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating PaD-TS on fMRI"
        python eval_unified.py \
            --fake "$PADTS_FAKE" --real "$REAL" \
            --name fmri --method padts \
            --output results/eval/fmri_padts.json \
            2>&1 | tee results/eval/fmri_padts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi

    DIFFTS_FAKE="$HOME/Diffusion-TS/OUTPUT/fmri/ddpm_fake_fmri.npy"
    if [ -f "$DIFFTS_FAKE" ]; then
        echo ""
        echo ">>> Evaluating Diffusion-TS on fMRI"
        python eval_unified.py \
            --fake "$DIFFTS_FAKE" --real "$REAL" \
            --name fmri --method diffts \
            --fake_range original \
            --scaler_data ./dataset/fMRI \
            --output results/eval/fmri_diffts.json \
            --skip cfid xcorr \
            2>&1 | tee results/eval/fmri_diffts_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi
else
    echo "SKIP: fMRI ground truth not found."
fi

echo ""
echo "=========================================="
echo "Evaluation completed at: $(date)"
echo "Results in: results/eval/"
echo "Exit code: $OVERALL_EXIT"
echo "=========================================="

exit $OVERALL_EXIT
