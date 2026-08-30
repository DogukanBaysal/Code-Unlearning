#!/usr/bin/env bash
#SBATCH --job-name=new-npo-kl-eval
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gpus=4
#SBATCH --output=Results/new_secret_npo_kl_eval_suite/slurm-%j.out
#SBATCH --error=Results/new_secret_npo_kl_eval_suite/slurm-%j.err

# Evaluate all new NPO+KL adapters with the established pass@10 settings:
# eight Secret variants and two Code-unit variants, each at three checkpoints.
# The underlying runner assigns jobs across all GPUs exposed by Slurm.

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${REPO_ROOT}"

mkdir -p Results/new_secret_npo_kl_eval_suite

nvidia-smi

# Uncomment/adapt these two lines if your cluster exposes Conda through modules.
# module load mamba
# source activate ft
conda activate ft

echo "Started: $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not-set}"
echo "SLURM job ID: ${SLURM_JOB_ID:-not-set}"

bash scripts/run_new_secret_npo_kl_eval_suite.sh \
    2>&1 | tee "Results/new_secret_npo_kl_eval_suite/slurm-${SLURM_JOB_ID:-manual}.log"

echo "Finished: $(date --iso-8601=seconds)"
