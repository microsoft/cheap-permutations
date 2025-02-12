#!/bin/bash

# Description:
# This script runs the postprocessing_cheap.py Python script with different values of 'b'.
# The 'b' values used are: 19, 39, 79, 159, 319, 639, 1279.

# List of n values to iterate over
b_values=(19 39 79 159 319 639 1279 2559)

# Loop through each n value
for b in "${b_values[@]}"
do
    echo "----------------------------------------"
    echo "Running postprocessing_cheap.py with n=2048, b=$b"

    # Execute the Python command with the current n value
    python postprocessing_cheap.py \
        --name gaussians_cov \
	--test_type independence \
        --n 2048 \
        --cross_covariance 0.047 \
        --B "$b" \
        --n_tests 10 \
        --total_n_tests 10000 \
        --d 10 \
	--no_ind_cross \
	--no_ind_cheap_perm_WB

    # Optional: Check if the Python command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Python script failed for n=$n"
        exit 1
    fi

    echo "Completed run for n=2048, b=$b"
    echo "----------------------------------------"
done

echo "All runs completed successfully."
