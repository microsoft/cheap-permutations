import argparse

def get_args_test():
    parser = argparse.ArgumentParser(description='MMD tests')
    
    #General arguments
    parser.add_argument('--name', type=str, default='MNIST', help='experiment name')
    parser.add_argument('--use_grid', action='store_true', help='use grid')
    parser.add_argument('--n', type=int, default=1024, help='number of samples')
    parser.add_argument('--d', type=int, default=10, help='dimension')
    parser.add_argument('--B', type=int, default=39, help='number of permutations/Rademacher variables used')
    parser.add_argument('--seed', type=int, default=38, help='seed')
    parser.add_argument('--seed_0', type=int, default=0, help='when running sweep jobs, first value of the seed')
    parser.add_argument('--alpha', type=float, default=0.05, help='level of the test')
    parser.add_argument('--n_tests', type=int, default=1, help='number of tests')
    parser.add_argument('--interactive', action='store_true', help='interactive, i.e. do not save results')
    parser.add_argument('--task_id', type=int, default=None, help='task id for sweep jobs')
    parser.add_argument('--number_of_jobs', type=int, default=1000, help='number of sweep jobs')
    parser.add_argument('--test_type', '-t', type=str, default='two_sample', 
                        choices=['two_sample', 'independence'], 
                        help='type of test to run')
    parser.add_argument('--n_lists', action='store_true', help='use s_lists for n_samples plots')
    parser.add_argument('--save_objects', action='store_true', help='save objects for each test')
    
    #Argument for gaussians
    parser.add_argument('--mean_diff', type=float, default=0.024, help='difference between means (for gaussians)')
    
    #Argument for blobs
    parser.add_argument('--grid_size', type=int, default=3, help='dimension of the grid of the distribution (for blobs)')
    parser.add_argument('--epsilon', type=float, default=2, help='covariance eigenvalue (for blobs)')
    
    #Argument for MNIST and EMNIST
    parser.add_argument('--p_even', type=float, default=0.49, help='joint probability of all even digits')
    
    #Arguments for Higgs
    parser.add_argument('--mixing', action='store_true', help='if passed use test mixing between classes')
    parser.add_argument('--null', action='store_true', help='if passed use null hypothesis, else use alternative')
    parser.add_argument('--n_components', type=int, default=4, help='number of dimensions to use')
    parser.add_argument('--p_poisoning', type=float, default=0.9, help='if mixing, poisoning probability of class 1 with class 0')
    
    #Argument for gaussians_cov
    parser.add_argument('--cross_covariance', type=float, default=0.1, help='cross covariance')

    #Arguments for aggregated tests
    parser.add_argument('--n_bandwidths', type=int, default=1, help='number of bandwidths used in the aggregated test')
    parser.add_argument('--B_2', type=int, default=200, help='number of permutations used for Monte Carlo estimation in agg.')
    parser.add_argument('--B_3', type=int, default=20, help='number of bisection iterations for aggregated test')

    #Arguments to avoid computing specific test groups, overriding default behavior 
    #(which is computing test groups for which no result files exist)
    parser.add_argument('--no_complete', action='store_true', help='if passed do not compute complete tests')
    parser.add_argument('--no_cheap_perm', action='store_true', help='if passed do not compute cheap_perm tests')
    parser.add_argument('--no_cross_mmd', action='store_true', help='if passed do not compute cross_mmd tests')
    parser.add_argument('--no_rff', action='store_true', help='if passed do not compute rff tests')
    parser.add_argument('--no_cheap_rff', action='store_true', help='if passed do not compute cheap_rff tests')
    parser.add_argument('--no_wilcoxon', action='store_true', help='if passed do not compute Wilcoxon tests')
    parser.add_argument('--no_ind_complete_WB', action='store_true', help='if passed do not compute complete tests')
    parser.add_argument('--no_ind_cheap_perm_WB', action='store_true', help='if passed do not compute cheap permutation tests')
    parser.add_argument('--no_ind_cross', action='store_true', help='if passed do not compute cross independence tests')

    #Arguments to recompute specific test groups, overriding default behavior
    parser.add_argument('--recompute_all', action='store_true', help='if passed recompute all tests')
    parser.add_argument('--recompute_complete', action='store_true', help='if passed recompute complete tests')
    parser.add_argument('--recompute_cheap_perm', action='store_true', help='if passed recompute cheap_perm tests')
    parser.add_argument('--recompute_cross_mmd', action='store_true', help='if passed recompute cross_mmd tests')
    parser.add_argument('--recompute_rff', action='store_true', help='if passed recompute rff tests')
    parser.add_argument('--recompute_cheap_rff', action='store_true', help='if passed recompute cheap_rff tests')
    parser.add_argument('--recompute_wilcoxon', action='store_true', help='if passed recompute Wilcoxon tests')
    parser.add_argument('--recompute_ind_complete_WB', action='store_true', help='if passed recompute complete tests')
    parser.add_argument('--recompute_ind_cheap_perm_WB', action='store_true', help='if passed recompute cheap permutation tests')
    parser.add_argument('--recompute_ind_cross', action='store_true', help='if passed recompute cross independence tests')

    #set argument values
    args = parser.parse_known_args()[0]
    
    return args
