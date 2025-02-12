#!/bin/bash
#SBATCH --job-name=gaussians_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=03:15:00
#SBATCH --mem=6G

#SBATCH --array=1-1000

module purge
module load python/intel/3.8.6

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 1 --n 8192 --n_tests 10 --mean_diff 0.03 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_complete --no_cheap_perm --no_cross_mmd
