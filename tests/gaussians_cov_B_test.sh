#!/bin/bash
# Run Gaussian independence test (gaussians_cov_B) experiment as B varies for fixed n
# 
# Args:
#   first_task_id (optional) - which replication to run (default: 1)
#   last_task_id - replication number >= last_task_id (default: first_task_id)
#
# Usage:
#   bash gaussians_cov_B_test.sh 1
#   bash gaussians_cov_B_test.sh 1 1000

first_task_id="${1:-1}"
last_task_id="${2:-$first_task_id}"

n_tests=10
# Extra flags to append
flags=""

echo "Running ${0} tasks [$first_task_id,$last_task_id]"
for task_id in $(seq $first_task_id $last_task_id);
do
    echo "Running task ${task_id}"
    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 19 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags

    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 39 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags

    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 79 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags

    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 159 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags

    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 319 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags

    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 639 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags

    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 1279 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags

    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 2559 --seed_0 0 --no_ind_cheap_perm_WB --no_ind_cross $flags
done


