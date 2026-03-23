#!/bin/bash
#SBATCH -A aqu2_lab_gpu
#SBATCH -J fdds_mh
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/fdds_mh-%j.out
#SBATCH --error=logs/fdds_mh-%j.err

module purge
module load anaconda/2024.06
eval "$(conda shell.bash hook)"
conda activate padts
cd ~/PaD-TS/comparison
python compute_fdds_mhealth.py
