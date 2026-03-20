#!/bin/bash
# setup_env.sh — Create the "padts" conda environment on UCI HPC3
# Usage: chmod +x setup_env.sh && ./setup_env.sh

set -eo pipefail

module purge
module load anaconda/2024.06
eval "$(conda shell.bash hook)"

ENV_NAME="padts"

# Remove existing env if present
if conda info --envs | grep -q "^${ENV_NAME} "; then
    echo "Removing existing '${ENV_NAME}' environment..."
    conda env remove -n "${ENV_NAME}" -y
fi

echo "Creating conda environment '${ENV_NAME}' with Python 3.8..."
conda create -n "${ENV_NAME}" python=3.8 -y

echo "Activating '${ENV_NAME}'..."
conda activate "${ENV_NAME}"

# Verify activation
ACTIVE_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
if [ "$ACTIVE_ENV" != "$ENV_NAME" ]; then
    echo "ERROR: conda activate failed. Active env is '$ACTIVE_ENV', expected '$ENV_NAME'."
    echo "Try running manually: conda activate $ENV_NAME && pip install ..."
    exit 1
fi
echo "Confirmed active environment: $ACTIVE_ENV"

# PyTorch with CUDA 11.8 support
echo "Installing PyTorch (CUDA 11.8)..."
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118

# TensorFlow (used by evaluation metrics)
echo "Installing TensorFlow..."
pip install tensorflow==2.10.0

# Core scientific stack
echo "Installing scientific dependencies..."
pip install \
    numpy==1.24.3 \
    scipy==1.8.1 \
    scikit-learn==1.1.2 \
    pandas==1.5.0 \
    matplotlib==3.6.0

# Model architecture dependencies
echo "Installing model dependencies..."
pip install \
    timm==0.9.16 \
    einops==0.6.0 \
    torchsummary==1.4 \
    torch-summary==1.4.5

# Utilities
echo "Installing utilities..."
pip install \
    tqdm==4.66.1 \
    h5py==3.11.0 \
    pyyaml==6.0 \
    accelerate==0.26.1

echo ""
echo "============================================"
echo "Environment '${ENV_NAME}' created successfully."
echo "Activate with: conda activate ${ENV_NAME}"
echo "============================================"
