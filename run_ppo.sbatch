#!/bin/bash
# Stage 4, step 3: PPO fine-tuning, one job per seed.   Submit:  sbatch run_ppo.sbatch      (seeds 1-5 run as an array)
# Uses NumPy + MuJoCo only (no GPU needed), with --workers parallel simulators, one per CPU.
# RBCCPS cluster: partition debug, account YOUR_ACCOUNT, conda environment `rl` (do NOT reinstall it or replace its PyTorch).
#SBATCH --job-name=arm6_ppo
#SBATCH --partition=debug
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --array=1-5
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ppo_%A_%a.out
#SBATCH --error=logs/ppo_%A_%a.err

set -e
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs runs

# conda activation that works in a non-interactive batch shell (submit from a shell where `conda` works)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1      # one BLAS thread per simulator process
export PYTHONPATH=.
SEED=${SLURM_ARRAY_TASK_ID:-1}
python3 -c "import numpy, scipy, mujoco; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'mujoco', mujoco.__version__)"
python3 train_ppo.py --init runs/il_policy.npz --out runs/ppo_s${SEED} --seed ${SEED} \
    --workers ${SLURM_CPUS_PER_TASK:-4} --iters 300 --steps 32768 --eval-every 5 --eval-eps 20
