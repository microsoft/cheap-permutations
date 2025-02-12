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

# python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 1024 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_cheap_perm --no_cross_mmd

# python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 2048 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_cheap_perm --no_cross_mmd

# python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 3072 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_complete

# python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 4096 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_complete

# python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 6144 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_complete

python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 8129 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_complete

# python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 12288 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_complete

# python ./test_cheap.py --name gaussians --task_id $SLURM_ARRAY_TASK_ID --d 10 --n 16384 --n_tests 10 --mean_diff 0.06 --B 1279 --seed_0 0 --no_cheap_rff --no_rff --no_wilcoxon --no_complete
