# Cheap Permutation Testing

This repository replicates the experiments of [Cheap Permutation Testing](https://arxiv.org/pdf/2502.07672).

```
@article{domingoenrich2025cheap,
  title={Cheap Permutation Testing},
  author={Carles Domingo-Enrich and Raaz Dwivedi and Lester Mackey},
  journal={arXiv preprint arxiv:2502.07672},
  year={2025}
}
```

> Permutation tests are a popular choice for distinguishing distributions and testing independence, due to their exact, finite-sample control of false positives and their minimax optimality when paired with U-statistics. However, standard permutation tests are also expensive, requiring a test statistic to be computed hundreds or thousands of times to detect a separation between distributions. In this work, we offer a simple approach to accelerate testing: group your datapoints into bins and permute only those bins. For U and V-statistics, we prove that these cheap permutation tests have two remarkable properties. First, by storing appropriate sufficient statistics, a cheap test  can be run in time comparable to evaluating a single test statistic. Second, cheap permutation power closely approximates standard permutation power. As a result, cheap tests inherit the exact false positive control and minimax optimality of standard permutation tests while running in a fraction of the time. We complement these findings with improved power guarantees for standard permutation testing and experiments demonstrating the benefits of cheap permutations over standard maximum mean discrepancy (MMD), Hilbert-Schmidt independence criterion (HSIC), random Fourier feature, Wilcoxon-Mann-Whitney, cross-MMD, and cross-HSIC tests. 

## Installation

All homogeneity experiments were run with Python 3.8.6 on a Slurm cluster of Intel Xeon Platinum 8268 CPU. 
All independence experiments were run with Python 3.13.0 on a single AMD EPYC 7V13 CPU.

Create and activate the conda environment `cheap-env` by running the commands
```
conda env create -f cheap-env.yml
conda activate cheap-env
```
To install the local package `cheaper`, run the following command from the folder that contains this README file:
```
pip install -e .
```
All commands must be run from the `tests` folder.

## Running homogeneity tests:

Cheap MMD tests with $n=16384$ samples and varying number of bins $s$, cheap RFF tests with $n=16384$ and varying $s$, rank $r$: 
```
sbatch gaussians_s_test.slurm 
```
Cheap and standard MMD tests and CrossMMD tests, with varying $n$ and $s$:
```
sbatch gaussians_n_test.slurm
```
Standard MMD tests with $n=16384$ and varying number of permutations $B$:
```
sbatch gaussians_B_test.slurm
```
Cheap Wilcoxon tests with $n=16384$ samples and varying $s$:
```
sbatch gaussians_wilcoxon_s_test.slurm
```
Cheap and standard Wilcoxon tests with varying $n$ and $s$:
```
sbatch gaussians_wilcoxon_n_test.slurm
```

## Running independence tests:

Cheap HSIC tests with $n=2048$ samples and varying number of bins $s$:
```
sbatch gaussians_cov_s_test.slurm
```
Cheap and standard HSIC tests and CrossHSIC tests, with varying $n$ and $s$:
```
sbatch gaussians_cov_n_test.slurm
```
Standard HSIC tests with $n=2048$ and varying number of permutations $B$:
```
sbatch gaussians_cov_B_test.slurm
```

## Post-processing homogeneity tests:

Cheap MMD tests with $n=16384$ samples and varying number of bins $s$, cheap RFF tests with $n=16384$ and varying $s$, rank $r$:
```
./postprocessing_cheap_s_gaussians.sh 
```
Cheap and standard MMD tests and CrossMMD tests, with varying $n$ and $s$:
```
./postprocessing_cheap_n_gaussians.sh
```
Standard MMD tests with $n=16384$ and varying number of permutations $B$:
```
./postprocessing_cheap_B_gaussians.sh
```
Cheap Wilcoxon tests with $n=16384$ samples and varying $s$:
```
./postprocessing_cheap_s_wilcoxon.sh
```
Cheap and standard Wilcoxon tests with varying $n$ and $s$:
```
./postprocessing_cheap_n_wilcoxon.sh
```

## Post-processing independence tests:

Cheap HSIC tests with $n=2048$ samples and varying number of bins $s$:
```
./postprocessing_cheap_s_gaussians_cov.sh
```
Cheap and standard HSIC tests and CrossHSIC tests, with varying $n$ and $s$:
```
./postprocessing_cheap_n_gaussians_cov.sh
```
Standard HSIC tests with $n=2048$ and varying number of permutations $B$:
```
./postprocessing_cheap_B_gaussians_cov.sh
```

## Figures homogeneity tests:

Cheap MMD tests with $n=16384$ samples and varying number of bins $s$, power vs. time:
```
python figures_cheap.py --name gaussians --n 8192 --d 10 --n_tests 1000 --total_n_tests 10000 --mean_diff 0.06 --log_time_scale --B 1279 --wilson_intervals --no_rff --no_cheap_rff --no_cross_mmd --no_complete --no_nominal_level --no_title
```
Cheap RFF tests with $n=16384$ and varying $s$, rank $r$, power vs. time:
```
python figures_cheap.py --name gaussians --n 8192 --d 10 --n_tests 1000 --total_n_tests 10000 --mean_diff 0.06 --log_time_scale --B 1279 --wilson_intervals --no_complete --no_cheap_perm --no_cross_mmd --no_wilcoxon --no_rff --no_nominal_level --no_title
```
Cheap and standard MMD tests, with varying $n$ and $s$, power vs. time:
```
python figures_cheap.py --name gaussians --n 8192 --d 10 --n_tests 1000 --total_n_tests 10000 --mean_diff 0.06 --log_time_scale --B 1279 --wilson_intervals --no_rff --no_cheap_rff --no_cross_mmd --no_wilcoxon --no_nominal_level --plot_n_samples --no_title --n_lists
```
Cheap and standard MMD tests and CrossMMD tests, with varying $n$ and $s$, power vs. number of samples:
```
python figures_cheap.py --name gaussians --n 8192 --d 10 --n_tests 1000 --total_n_tests 10000 --mean_diff 0.06 --log_time_scale --B 1279 --wilson_intervals --no_rff --no_cheap_rff --no_wilcoxon --no_nominal_level --plot_n_samples_vs_power --no_title --n_lists
```
Cheap and standard MMD tests and CrossMMD tests, with varying $n$ and $s$, power vs. time:
```
python figures_cheap.py --name gaussians --n 8192 --d 10 --n_tests 1000 --total_n_tests 10000 --mean_diff 0.06 --log_time_scale --B 1279 --wilson_intervals --no_rff --no_cheap_rff --no_wilcoxon --no_nominal_level --plot_n_samples --no_title --n_lists
```
Standard MMD tests with $n=16384$ and varying number of permutations $B$, power vs. time:
```
python figures_cheap.py --name gaussians --n 8192 --d 10 --n_tests 1000 --total_n_tests 10000 --mean_diff 0.06 --log_time_scale --B 1279 --wilson_intervals --plot_n_permutations --no_cross_mmd --no_rff --no_cheap_rff --no_cheap_perm --no_nominal_level --no_title
```
Cheap Wilcoxon tests with $n=16384$ samples and varying $s$, power vs. time:
```
python figures_cheap.py --name gaussians --n 8192 --mean_diff 0.03 --B 1279 --n_tests 10 --total_n_tests 10000 --d 1 --log_time_scale --wilson_intervals --no_complete --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --plot_wilcoxon --no_nominal_level --no_title
```
Cheap and standard Wilcoxon tests with varying $n$ and $s$, power vs. time:
```
python figures_cheap.py --name gaussians --n 8192 --mean_diff 0.03 --B 1279 --n_tests 10 --total_n_tests 10000 --d 1 --log_time_scale --wilson_intervals --no_complete --no_cheap_perm --no_cross_mmd --no_rff --no_cheap_rff --plot_wilcoxon --plot_n_samples --no_nominal_level --no_title --n_lists
```

## Figures independence tests:

Cheap HSIC tests with $n=2048$ samples and varying number of bins $s$, power vs. time:
```
python figures_cheap.py --name gaussians_cov --test_type independence --n 2048 --d 10 --n_tests 1000 --total_n_tests 10000 --cross_covariance 0.047 --log_time_scale --B 1279 --wilson_intervals --no_ind_complete_WB --no_ind_cross --no_nominal_level --no_title
```
Cheap and standard HSIC tests, with varying $n$ and $s$, power vs. time:
```
python figures_cheap.py --name gaussians_cov --test_type independence --n 2048 --d 10 --n_tests 1000 --total_n_tests 10000 --cross_covariance 0.047 --log_time_scale --B 1279 --wilson_intervals --no_ind_cross --no_nominal_level --plot_n_samples --no_title --n_lists
```
Cheap and standard HSIC tests and CrossHSIC tests, with varying $n$ and $s$, power vs. time:
```
python figures_cheap.py --name gaussians_cov --test_type independence --n 2048 --d 10 --n_tests 1000 --total_n_tests 10000 --cross_covariance 0.047 --log_time_scale --B 1279 --wilson_intervals --no_nominal_level --plot_n_samples --no_title --n_lists
```
Cheap and standard HSIC tests and CrossHSIC tests, with varying $n$ and $s$, power vs. number of samples:
```
python figures_cheap.py --name gaussians_cov --test_type independence --n 2048 --d 10 --n_tests 1000 --total_n_tests 10000 --cross_covariance 0.047 --log_time_scale --B 1279 --wilson_intervals --no_nominal_level --plot_n_samples_vs_power --no_title --n_lists
```
Standard HSIC tests with $n=2048$ and varying number of permutations $B$, power vs. time:
```
python figures_cheap.py --name gaussians_cov --test_type independence --n 2048 --d 10 --n_tests 1000 --total_n_tests 10000 --cross_covariance 0.047 --log_time_scale --B 1279 --wilson_intervals --plot_n_permutations --no_ind_cheap_perm_WB --no_ind_cross --no_title
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
