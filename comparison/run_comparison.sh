#!/bin/bash
#SBATCH -A aqu2_lab_gpu
#SBATCH -J compare_all
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --error=../logs/compare_all-%J.err
#SBATCH --output=../logs/compare_all-%J.out
#SBATCH --mail-type=fail,end
#SBATCH --mail-user=shilligo@uci.edu

set -o pipefail

# ================================================================
# Full comparison pipeline for Stocks and MHEALTH datasets.
# Runs metrics + visualizations + tables for both datasets.
#
# Prerequisites:
#   - PaD-TS training completed for target datasets
#   - Diffusion-TS training completed for target datasets
#
# Usage:
#   sbatch run_comparison.sh
# ================================================================

module purge
module load anaconda/2024.06
eval "$(conda shell.bash hook)"
conda activate padts

cd ~/PaD-TS/comparison || { echo "ERROR: Could not cd to ~/PaD-TS/comparison"; exit 1; }

mkdir -p results/stocks results/mhealth tables figures ../logs

PADTS_OUTPUT="$HOME/PaD-TS/OUTPUT"
DIFFTS_OUTPUT="$HOME/Diffusion-TS/OUTPUT"

OVERALL_EXIT=0

echo "=========================================="
echo "Comparison Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Time: $(date)"
echo "=========================================="

# ── Helper: find .npy files by pattern ─────────────────────────────────────

find_npy() {
    # Usage: find_npy <directory> <pattern>
    # Returns first match or empty string
    find "$1" -maxdepth 3 -name "$2" -type f 2>/dev/null | head -1
}

# ── STOCKS ─────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "DATASET: Stocks"
echo "=========================================="

STOCK_REAL=$(find_npy "$PADTS_OUTPUT" "stock_norm_truth_24_train.npy")
STOCK_PADTS=$(find_npy "$PADTS_OUTPUT" "ddpm_fake_stock_24.npy")
STOCK_DIFFTS=$(find_npy "$DIFFTS_OUTPUT/stocks" "ddpm_fake_stock*.npy")

echo "  Real:       ${STOCK_REAL:-NOT FOUND}"
echo "  PaD-TS:     ${STOCK_PADTS:-NOT FOUND}"
echo "  Diffusion-TS: ${STOCK_DIFFTS:-NOT FOUND}"

if [ -n "$STOCK_REAL" ]; then
    METRIC_ARGS="--real $STOCK_REAL --name stock"
    VIZ_ARGS="--real $STOCK_REAL --name stock --outdir ./figures"

    [ -n "$STOCK_PADTS" ] && METRIC_ARGS="$METRIC_ARGS --fake_padts $STOCK_PADTS --padts_range neg1to1"
    [ -n "$STOCK_DIFFTS" ] && METRIC_ARGS="$METRIC_ARGS --fake_diffts $STOCK_DIFFTS --diffts_range zero1"

    if [ -n "$STOCK_PADTS" ] || [ -n "$STOCK_DIFFTS" ]; then
        echo ""
        echo ">>> Computing metrics for Stocks..."
        python compute_population_metrics.py $METRIC_ARGS \
            --output results/stocks/metrics.json \
            2>&1 | tee results/stocks/metrics_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi

    if [ -n "$STOCK_PADTS" ] && [ -n "$STOCK_DIFFTS" ]; then
        echo ""
        echo ">>> Generating visualizations for Stocks..."
        python compare_visualizations.py $VIZ_ARGS \
            --diffts "$STOCK_DIFFTS" --padts "$STOCK_PADTS" \
            --diffts_range zero1 --padts_range neg1to1 \
            2>&1 | tee results/stocks/viz_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi
else
    echo "  SKIP: No ground truth found for Stocks."
fi

# ── MHEALTH ────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
echo "DATASET: MHEALTH"
echo "=========================================="

MHEALTH_REAL=$(find_npy "$PADTS_OUTPUT" "mhealth_norm_truth_24_train.npy")
MHEALTH_PADTS=$(find_npy "$PADTS_OUTPUT" "ddpm_fake_mhealth_24.npy")
MHEALTH_DIFFTS=$(find_npy "$DIFFTS_OUTPUT/mhealth" "ddpm_fake_mhealth*.npy")

echo "  Real:       ${MHEALTH_REAL:-NOT FOUND}"
echo "  PaD-TS:     ${MHEALTH_PADTS:-NOT FOUND}"
echo "  Diffusion-TS: ${MHEALTH_DIFFTS:-NOT FOUND}"

if [ -n "$MHEALTH_REAL" ]; then
    METRIC_ARGS="--real $MHEALTH_REAL --name mhealth"
    VIZ_ARGS="--real $MHEALTH_REAL --name mhealth --outdir ./figures"

    [ -n "$MHEALTH_PADTS" ] && METRIC_ARGS="$METRIC_ARGS --fake_padts $MHEALTH_PADTS --padts_range neg1to1"
    [ -n "$MHEALTH_DIFFTS" ] && METRIC_ARGS="$METRIC_ARGS --fake_diffts $MHEALTH_DIFFTS --diffts_range zero1"

    if [ -n "$MHEALTH_PADTS" ] || [ -n "$MHEALTH_DIFFTS" ]; then
        echo ""
        echo ">>> Computing metrics for MHEALTH..."
        python compute_population_metrics.py $METRIC_ARGS \
            --output results/mhealth/metrics.json \
            2>&1 | tee results/mhealth/metrics_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1
    fi

    if [ -n "$MHEALTH_PADTS" ] && [ -n "$MHEALTH_DIFFTS" ]; then
        echo ""
        echo ">>> Generating visualizations for MHEALTH..."
        python compare_visualizations.py $VIZ_ARGS \
            --diffts "$MHEALTH_DIFFTS" --padts "$MHEALTH_PADTS" \
            --diffts_range zero1 --padts_range neg1to1 \
            2>&1 | tee results/mhealth/viz_log.txt
        [ $? -ne 0 ] && OVERALL_EXIT=1

        SUBJ_LABELS="$HOME/PaD-TS/dataset/mhealth_window_subjects.npy"
        if [ -f "$SUBJ_LABELS" ]; then
            echo ""
            echo ">>> Running subject-level heterogeneity analysis for MHEALTH..."
            HETERO_ARGS="--real $MHEALTH_REAL --subject_labels $SUBJ_LABELS --outdir results/mhealth"
            [ -n "$MHEALTH_DIFFTS" ] && HETERO_ARGS="$HETERO_ARGS --diffts $MHEALTH_DIFFTS --diffts_range zero1"
            [ -n "$MHEALTH_PADTS" ] && HETERO_ARGS="$HETERO_ARGS --padts $MHEALTH_PADTS --padts_range neg1to1"
            python analyze_heterogeneity.py $HETERO_ARGS \
                2>&1 | tee results/mhealth/heterogeneity_log.txt
            [ $? -ne 0 ] && OVERALL_EXIT=1
        else
            echo "  SKIP heterogeneity: $SUBJ_LABELS not found."
            echo "  Run: python preprocess_mhealth.py --input_dir /path/to/MHEALTHDATASET"
        fi
    fi
else
    echo "  SKIP: No ground truth found for MHEALTH."
fi

# ── GENERATE COMBINED TABLE ───────────────────────────────────────────────

echo ""
echo "=========================================="
echo "Generating combined comparison table..."
echo "=========================================="

JSON_FILES=""
[ -f results/stocks/metrics.json ] && JSON_FILES="$JSON_FILES results/stocks/metrics.json"
[ -f results/mhealth/metrics.json ] && JSON_FILES="$JSON_FILES results/mhealth/metrics.json"

if [ -n "$JSON_FILES" ]; then
    python generate_comparison_table.py --inputs $JSON_FILES --outdir ./tables
    [ $? -ne 0 ] && OVERALL_EXIT=1
else
    echo "  SKIP: No metric JSON files found."
fi

echo ""
echo "=========================================="
echo "Comparison pipeline completed at: $(date)"
echo "Exit code: $OVERALL_EXIT"
echo ""
echo "Outputs:"
echo "  Metrics:  results/stocks/metrics.json, results/mhealth/metrics.json"
echo "  Figures:  figures/"
echo "  Tables:   tables/"
echo "=========================================="

exit $OVERALL_EXIT
