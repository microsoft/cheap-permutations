#!/bin/bash
#
# This script runs the postprocessing_cheap.py Python script with different values
# of 'n' to prepare the aggregated Higgs two-sample test results (aggregated complete
# and aggregated cheap tests) used by fig_n_two_sample_aggregated in figures_cheap.py.
#
# Example usage: bash postprocessing_cheap_n_aggregated_higgs.sh

# List of n values to iterate over (same range as postprocessing_cheap_n_gaussians.sh)
n_values=(1024 2048 4096 8192 16384)

for p_poisoning in 0.5 0.0
do
    # Loop through each n value
    for n in "${n_values[@]}"
    do
        echo "----------------------------------------"
        echo "Running postprocessing_cheap.py with n=$n and p_poisoning=$p_poisoning"

        # Execute the Python command with the current n value
        python postprocessing_cheap.py \
            --name Higgs \
            --n "$n" \
            --n_components 2 \
            --p_poisoning "$p_poisoning" \
            --mixing \
            --B 299 \
            --B_2 200 \
            --B_3 20 \
            --n_bandwidths 5 \
            --n_tests 1 \
            --total_n_tests 400

        # Optional: Check if the Python command was successful
        if [ $? -ne 0 ]; then
            echo "Error: Python script failed for n=$n"
            exit 1
        fi

        echo "Completed run for n=$n and p_poisoning=$p_poisoning"
        echo "----------------------------------------"
    done
done

echo "All runs completed successfully."
