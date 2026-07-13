#!/bin/bash
# Run Higgs aggregated MMD two-sample test experiment as n and s vary
# 
# Args:
#   first_task_id (optional) - which replication to run (default: 1)
#   last_task_id - replication number >= last_task_id (default: first_task_id)
#
# Usage:
#   bash higgs_n_test.sh 

echo "Running ${0}"
n_tests=2500
counter=95 # Initialize counter to number of CPU cores minus 1
for n in 1024 2048 4096 8192 16384; do
    # Restrict each job to a single CPU core
	taskset -c $counter python test_cheap.py --name Higgs --task_id 1 --n_components 2 --n $n --p_poisoning 0.0 --n_tests $n_tests --B 299 --B_2 200 --seed_0 0 --mixing --n_bandwidths 5 &
	((counter--))
done
