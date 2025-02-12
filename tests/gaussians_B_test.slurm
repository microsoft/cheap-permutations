#!/bin/bash
#SBATCH --job-name=gaussians_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=01:00:00
#SBATCH --mem=4G

#SBATCH --array=1-1000

module purge
module load python/intel/3.8.6

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 19 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 39 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 79 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 159 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 319 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 639 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 1279 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --n 8192 --mean_diff 0.06 --B 2559 --n_tests 10 --total_n_tests 10000 --d 10 --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --no_wilcoxon
