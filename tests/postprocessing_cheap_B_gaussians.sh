#!/bin/bash

# Description:
# This script runs the postprocessing_cheap.py Python script with different values of 'B'.

# List of n values to iterate over
b_values=(19 39 79 159 319 639 1279 2559)

# Loop through each n value
for b in "${b_values[@]}"
do
    echo "----------------------------------------"
    echo "Running postprocessing_cheap.py with n=8192, b=$b"
    
    # Execute the Python command with the current n value
    python postprocessing_cheap.py \
        --name gaussians \
        --n 8192 \
        --mean_diff 0.06 \
        --B "$b" \
        --n_tests 10 \
        --total_n_tests 10000 \
        --d 10 \
        --no_cheap_perm \
        --no_cheap_rff \
        --no_rff \
        --no_wilcoxon \
	--no_cross_mmd
    
    # Optional: Check if the Python command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Python script failed for n=8192, b=$b"
        exit 1
    fi
    
    echo "Completed run for n=8192, b=$b"
    echo "----------------------------------------"
done

echo "All runs completed successfully."
