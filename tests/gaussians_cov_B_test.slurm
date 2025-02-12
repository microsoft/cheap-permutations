#!/bin/bash
#SBATCH --job-name=gaussians_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=00:20:00
#SBATCH --mem=4G

#SBATCH --array=1-1000

module purge
module load python/intel/3.8.6

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 19 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 39 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 79 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 159 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 319 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 639 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 1279 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 2559 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross
