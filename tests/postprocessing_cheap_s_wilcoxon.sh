#!/bin/bash

# Description:
# This script runs the postprocessing_cheap.py Python script with different values of 'n'.
# The 'n' values used are: 1536, 2048, 3072, 4096, 6144, 8192, 12244, 16384.

# List of n values to iterate over
n_values=(8192)
# n_values=(1024 2048 4096 8192 12288 16384)

# Loop through each n value
for n in "${n_values[@]}"
do
    echo "----------------------------------------"
    echo "Running postprocessing_cheap.py with n=$n"
    
    # Execute the Python command with the current n value
    python postprocessing_cheap.py \
        --name gaussians \
        --n "$n" \
        --mean_diff 0.03 \
        --B 1279 \
        --n_tests 10 \
        --total_n_tests 10000 \
        --d 1 \
        --no_cheap_perm \
        --no_cheap_rff \
        --no_rff \
        --no_complete \
	--no_cross_mmd
    
    # Optional: Check if the Python command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Python script failed for n=$n"
        exit 1
    fi
    
    echo "Completed run for n=$n"
    echo "----------------------------------------"
done

echo "All runs completed successfully."
