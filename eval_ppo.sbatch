#!/bin/bash
# Stage 4: fair evaluation of the imitation policy, the 5 PPO seeds and the scripted expert on unseen tasks.   Submit: sbatch eval_ppo.sbatch
#SBATCH --job-name=arm6_eval
#SBATCH --partition=debug
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs results
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rl
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=.
python3 eval_policies.py --episodes 40 --workers 16 --out results/stage4_eval.csv --expert \
  --policy il=runs/il_policy.npz \
  --policy ppo1=runs/ppo_s1/last_policy.npz --policy ppo2=runs/ppo_s2/last_policy.npz \
  --policy ppo3=runs/ppo_s3/last_policy.npz --policy ppo4=runs/ppo_s4/last_policy.npz \
  --policy ppo5=runs/ppo_s5/last_policy.npz
