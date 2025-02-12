#!/bin/bash
# Run replicates first_task_id, ..., last_task_id (inclusive) 
# of gaussians_cov_n_test in parallel across CPUs
# 
# Args:
#   first_task_id - replication number >= 1 (default: 1)
#   last_task_id - replication number >= last_task_id (default: first_task_id)
#   num_cpus (optional) - how many cpus to use (default: 20)
#
# Usage:
#   bash parallel_gaussians_cov_n_test.sh 12 100

# total number of CPUs available
total_cpus=24

first_task_id="${1:-1}"
last_task_id="${2:-$first_task_id}"
num_cpus="${3:-20}"

num_tasks=$(($last_task_id-$first_task_id+1))
echo "Running $num_tasks ${0} tasks [$first_task_id,$last_task_id] on $num_cpus CPUs"
num_tasks_per_cpu=$(($num_tasks / $num_cpus))
remainder=$(($num_tasks % $num_cpus))
echo "Using $num_tasks_per_cpu tasks per CPU with remainder $remainder"

start_task=$first_task_id
for ii in $(seq 1 $num_cpus);
do
    if [ $ii -gt $remainder ]; then
        end_task=$(($start_task + $num_tasks_per_cpu - 1))
    else
        # Extra task to cover remainder
        end_task=$(($start_task + $num_tasks_per_cpu))
    fi
    # Assign task to run on single CPU
    # Run in background
    cpu=$(($total_cpus-$ii))
    echo "taskset --cpu-list $cpu bash gaussians_cov_n_test.sh $start_task $end_task &"
    taskset --cpu-list $cpu bash gaussians_cov_n_test.sh $start_task $end_task &
    start_task=$(($end_task+1))
done