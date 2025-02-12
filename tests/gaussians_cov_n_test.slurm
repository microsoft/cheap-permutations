#!/bin/bash
#SBATCH --job-name=gaussians_grid_job
#SBATCH --open-mode=append
#SBATCH --output=./logs/%x_%A_%a.out
#SBATCH --error=./logs/%x_%A_%a.err
#SBATCH --export=ALL
#SBATCH --time=00:25:00
#SBATCH --mem=4G

#SBATCH --array=1-1000

module purge
module load python/intel/3.8.6

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 512 --n_tests 10 --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 1024 --n_tests 10 --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 1536 --n_tests 10 --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 3072 --n_tests 10 --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists

python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 4096 --n_tests 10 --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists
