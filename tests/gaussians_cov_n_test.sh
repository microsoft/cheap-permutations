#!/bin/bash
# Run Gaussian independence test (gaussians_cov_n) experiment as n and s vary
# 
# Args:
#   first_task_id (optional) - which replication to run (default: 1)
#   last_task_id - replication number >= last_task_id (default: first_task_id)
#
# Usage:
#   bash gaussians_cov_n_test.sh 1
#   bash gaussians_cov_n_test.sh 1 10


first_task_id="${1:-1}"
last_task_id="${2:-$first_task_id}"

n_tests=10
# Extra flags to append
flags="--recompute_ind_cheap_perm_WB"

echo "Running ${0} tasks [$first_task_id,$last_task_id]"
for task_id in $(seq $first_task_id $last_task_id);
do
    echo "Running task ${task_id}"
    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 512 --n_tests $n_tests --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists $flags
    
    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 1024 --n_tests $n_tests --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists $flags
    
    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 1536 --n_tests $n_tests --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists $flags
    
    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 2048 --n_tests $n_tests --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists $flags
    
    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 3072 --n_tests $n_tests --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists $flags
    
    python ./test_cheap.py --name gaussians_cov --test_type independence --task_id $task_id --d 10 --n 4096 --n_tests $n_tests --cross_covariance 0.047 --B 1279 --seed_0 0 --n_lists $flags
done
