import numpy as np
import time
import math
import os
import argparse
import pickle
import scipy
from cheaper import ctt
from functools import partial
from sklearn.datasets import fetch_openml
import itertools
from itertools import product
import util_classes
import util_classes_cheap
import util_sqMMD_estimators
import util_sqMMD_estimators_cheap
import util_parser_cheap
import util_sampling
import util_tests_cheap

def complete_test(X1,X2,B,alpha,lam,seed,group_results_dict,args):
    """
    Complete tests
    ##Change For each block size in args.wb_block_size_list and number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['complete'].compute_group:
        for size in args.complete_list:
            if group_results_dict['complete'].group_tests[str(size)].compute:
                # util_sqMMD_estimators_cheap.complete(X1,X2,B,lam,group_results_dict,seed=seed)
                print(f'size: {size}, X1.shape[0]: {X1.shape[0]}')
                util_sqMMD_estimators_cheap.cheap_perm(X1,X2,B,size,lam,group_results_dict,null_seed=seed,save_complete=True)

        # Get statistic value
        group_results_dict['complete'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['complete'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['complete'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['complete'].set_reject(alpha)
        # Save group results
        group_results_dict['complete'].set_group_results()
        group_results_dict['complete'].save_results(args)
        # In principle, no need to save the TestResults objects
        if args.save_objects:
            group_results_dict['complete'].save_objects(args)
        
def cheap_perm_test(X1,X2,B,alpha,lam,seed,group_results_dict,args):
    """
    Cheap permutation tests
    ##Change For each block size in args.wb_block_size_list and number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['cheap_perm'].compute_group:
        cheap_perm_list = args.cheap_perm_list_n if args.n_lists else args.cheap_perm_list
        for s in cheap_perm_list:
            if group_results_dict['cheap_perm'].group_tests[str(s)].compute:
                util_sqMMD_estimators_cheap.cheap_perm(X1,X2,B,s,lam,group_results_dict,null_seed=seed) ## Change
            
        # Get statistic value
        group_results_dict['cheap_perm'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['cheap_perm'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['cheap_perm'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['cheap_perm'].set_reject(alpha)
        # Save group results
        group_results_dict['cheap_perm'].set_group_results()
        group_results_dict['cheap_perm'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        if args.save_objects:
            group_results_dict['cheap_perm'].save_objects(args)
        
def cross_mmd_test(X1,X2,alpha,lam,seed,group_results_dict,args):
    """
    Cross MMD tests
    ##Change For each block size in args.wb_block_size_list and number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['cross_mmd'].compute_group:
        for size in args.cross_mmd_list:
            if group_results_dict['cross_mmd'].group_tests[str(size)].compute:
                util_sqMMD_estimators_cheap.cross_mmd(X1,X2,lam,alpha,group_results_dict) ## Change
    
        # Check if test statistic is above threshold to compute reject
        group_results_dict['cross_mmd'].set_reject(alpha, reject_with_threshold=True)
        # Save group results
        group_results_dict['cross_mmd'].set_group_results()
        group_results_dict['cross_mmd'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        if args.save_objects:
            group_results_dict['cross_mmd'].save_objects(args)
        
def rff_test(X1,X2,B,alpha,lam,seed,group_results_dict,args):
    """
    Random Fourier Features (RFF) Test
    Computes and stores RFF tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of permutations used
    alpha: float, level of the tests
    lam: vector of positive real-valued kernel bandwidths
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored
    args: arguments
    """
    # Create null and statistic seeds from input seed
    rng = np.random.default_rng(seed)
    ss_test = rng.bit_generator._seed_seq
    child_ss_test = ss_test.spawn(2)
    # Use integer seeds since they will be shared across multiple experimental 
    # settings
    null_seed = child_ss_test[0].generate_state(1)
    statistic_seed = child_ss_test[1].generate_state(1)
    
    if group_results_dict['rff'].compute_group:
        for r in args.rff_list:
            if group_results_dict['rff'].group_tests[str(r)].compute:
                util_sqMMD_estimators_cheap.rff(X1,X2,r,B,lam,group_results_dict,null_seed=null_seed, statistic_seed=statistic_seed) ## Change
            
        # Get statistic value
        group_results_dict['rff'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['rff'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['rff'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['rff'].set_reject(alpha)
        # Save group results
        group_results_dict['rff'].set_group_results()
        group_results_dict['rff'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        if args.save_objects:
            group_results_dict['rff'].save_objects(args)

def cheap_rff_test(X1,X2,B,alpha,lam,seed,group_results_dict,args):
    """
    Random Fourier Features (RFF) Test
    Computes and stores RFF tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of permutations used
    alpha: float, level of the tests
    lam: vector of positive real-valued kernel bandwidths
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored
    args: arguments
    """
    # Create null and statistic seeds from input seed
    rng = np.random.default_rng(seed)
    ss_test = rng.bit_generator._seed_seq
    child_ss_test = ss_test.spawn(2)
    # Use integer seeds since they will be shared across multiple experimental 
    # settings
    null_seed = child_ss_test[0].generate_state(1)
    statistic_seed = child_ss_test[1].generate_state(1)
    
    if group_results_dict['cheap_rff'].compute_group:
        for r in args.rff_list:
            for s in args.cheap_rff_list_s:
                if group_results_dict['cheap_rff'].group_tests[str(r)+'_'+str(s)].compute:
                    util_sqMMD_estimators_cheap.cheap_rff(X1,X2,r,s,B,lam,group_results_dict,null_seed=null_seed, statistic_seed=statistic_seed) ## Change
            
        # Get statistic value
        group_results_dict['cheap_rff'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['cheap_rff'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['cheap_rff'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['cheap_rff'].set_reject(alpha)
        # Save group results
        group_results_dict['cheap_rff'].set_group_results()
        group_results_dict['cheap_rff'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        if args.save_objects:
            group_results_dict['cheap_rff'].save_objects(args)
            
def wilcoxon_test(X1,X2,B,alpha,seed,group_results_dict,args):
    """
    Wilcoxon tests
    ##Change For each block size in args.wb_block_size_list and number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['wilcoxon'].compute_group:
        cheap_perm_list = args.wilcoxon_list_n if args.n_lists else args.wilcoxon_list
        print(f"group_results_dict['wilcoxon'].group_tests.keys(): {group_results_dict['wilcoxon'].group_tests.keys()}")
        for s in cheap_perm_list:
            if group_results_dict['wilcoxon'].group_tests[str(s)].compute:
                util_sqMMD_estimators_cheap.wilcoxon(X1,X2,B,s,group_results_dict,null_seed=seed) ## Change
            
        # Get statistic value
        group_results_dict['wilcoxon'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['wilcoxon'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['wilcoxon'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['wilcoxon'].set_reject(alpha)
        # Save group results
        group_results_dict['wilcoxon'].set_group_results()
        group_results_dict['wilcoxon'].save_results(args)
        # In principle, no need to save the TestResults objects
        if args.save_objects:
            group_results_dict['wilcoxon'].save_objects(args)
        
def complete_WB_independence_test(X,B,alpha,lam_1,lam_2,seed,group_results_dict,args):
    """
    Complete independence tests
    
    X: 2D array of size (n_samples,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam_1: bandwidth for the first set of variables (float)
    lam_2: bandwidth for the second set of variables (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    if group_results_dict['ind_complete_WB'].compute_group:
        for size in args.complete_list:
            if group_results_dict['ind_complete_WB'].group_tests[str(size)].compute:
                #util_sqMMD_estimators_cheap.complete_independence(X,B,lam_1,lam_2,group_results_dict,seed=seed)
                util_sqMMD_estimators_cheap.cheap_perm_independence(X,B,X.shape[0],lam_1,lam_2,group_results_dict,seed=seed, save_complete=True)
            
        # Get statistic value
        group_results_dict['ind_complete_WB'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['ind_complete_WB'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['ind_complete_WB'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['ind_complete_WB'].set_reject(alpha)
        # Save group results
        group_results_dict['ind_complete_WB'].set_group_results()
        group_results_dict['ind_complete_WB'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        if args.save_objects:
            group_results_dict['ind_complete_WB'].save_objects(args)
            
def cheap_perm_WB_independence_test(X,B,alpha,lam_1,lam_2,seed,group_results_dict,args):
    """
    Cheap permutation independence tests
    
    X1: 2D array of size (n_samples_1,d)
    X2: 2D array of size (n_samples_2,d)
    B: int, number of Rademacher variables used
    alpha: float, level of the tests
    lam: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    print(f'cheap_perm_WB_independence_test function')
    if group_results_dict['ind_cheap_perm_WB'].compute_group:
        print(f'ind_cheap_perm_WB to be computed')
        cheap_perm_list = args.cheap_perm_list_n if args.n_lists else args.cheap_perm_list
        print(f'args.n_lists: {args.n_lists}, cheap_perm_list: {cheap_perm_list}')
        for s in cheap_perm_list:
            if group_results_dict['ind_cheap_perm_WB'].group_tests[str(s)].compute:
                print(f'ind_cheap_perm_WB {s} to be computed')
                util_sqMMD_estimators_cheap.cheap_perm_independence(X,B,s,lam_1,lam_2,group_results_dict,seed=seed)
            
        # Get statistic value
        group_results_dict['ind_cheap_perm_WB'].set_statistic_value()
        # Reorder list of values for each estimator
        group_results_dict['ind_cheap_perm_WB'].sort_estimator_values()      
        # Compute threshold by looking at the right quantile of sqMMD_list
        group_results_dict['ind_cheap_perm_WB'].set_threshold(alpha)
        # Check if test statistic is above threshold to compute reject
        group_results_dict['ind_cheap_perm_WB'].set_reject(alpha)
        # Save group results
        group_results_dict['ind_cheap_perm_WB'].set_group_results()
        group_results_dict['ind_cheap_perm_WB'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        if args.save_objects:
            group_results_dict['ind_cheap_perm_WB'].save_objects(args)
            
def cross_independence_test(X,B,alpha,lam_1,lam_2,seed,group_results_dict,args):
    """
    Cross HSIC tests
    ##Change For each block size in args.wb_block_size_list and number of pairs in args.wb_incomplete_list, 
    simultaneously computes and stores the corresponding wild bootstrap tests of level alpha with B permutations
    
    X: 2D array of size (n_samples,d)
    alpha: float, level of the tests
    lam_1: bandwidth (float)
    seed: random seed (numpy random number generator)
    group_results_dict: list of dictionaries. Each dictionary corresponds to a test, and contains one 
      util_classes.Group_Results object for each test group, where results are stored   
    args: arguments
    """
    print(f'cross_independence_test function')
    if group_results_dict['ind_cross'].compute_group:
        print(f'cross_independence to be computed')
        for size in args.cross_hsic_list:
            if group_results_dict['ind_cross'].group_tests[str(size)].compute:
                util_sqMMD_estimators_cheap.cross_hsic_independence(X,B,lam_1,lam_2,alpha,group_results_dict)
    
        # Check if test statistic is above threshold to compute reject
        group_results_dict['ind_cross'].set_reject(alpha, reject_with_threshold=True)
        # Save group results
        group_results_dict['ind_cross'].set_group_results()
        group_results_dict['ind_cross'].save_results(args)
        # In principle, no need to save the TestResults objects; uncomment the next line if needed
        if args.save_objects:
            group_results_dict['ind_cross'].save_objects(args)
        
def set_args_for_task_id(args, task_id):
    """
    Sets arguments in args for each job
    
    args: arguments
    task_id: job number
    """
    grid = {
        'seed': [i for i in range(args.seed_0,args.seed_0+args.number_of_jobs)]
    }
    gridlist = list(dict(zip(grid.keys(), vals)) for vals in product(*grid.values()))
    assert task_id >= 1 and task_id <= len(gridlist), 'wrong task_id!'
    elem = gridlist[task_id - 1]
    for k, v in elem.items():
        setattr(args, k, v)

def run_two_sample_tests():
    """
    Runs a total of args.n_tests non-aggregated tests and stores results into group_results_dict
    """
    # Build list of estimators
    args.estimator_list = args.estimators['complete'] + args.estimators['cheap_perm'] + args.estimators['cross_mmd'] + args.estimators['rff'] + args.estimators['cheap_rff'] + args.estimators['wilcoxon']
    
    # Build list of test groups
    test_groups = ['complete', 'cheap_perm', 'cross_mmd', 'rff', 'cheap_rff', 'wilcoxon']
    
    # Get directories to store the results of each test group
    resdir = util_classes_cheap.get_group_directories(args, test_groups)
            
    # Get file names to store the results
    groupname, testname = util_classes_cheap.get_test_file_names(args, test_groups, resdir, n_tests=args.n_tests, aggregated=False)
    
    fname, file_exists, fname_group, file_exists_group = util_classes.get_fname_and_file_exists(args, test_groups, resdir, groupname, testname)
        
    group_results_dict = dict()
        
    # Initialize base random number generator
    rng = np.random.default_rng(args.seed)
    # Create seed sequence for each test
    seed_seqs = rng.bit_generator._seed_seq.spawn(args.n_tests)
    for t in range(args.n_tests):
        print(f'Test number {t}')
        # From this test's seed sequence, construct two seeds,
        # one for randomness in constructing the test data and 
        # one for randomness in the test itself
        child_seed_seqs = seed_seqs[t].spawn(2)
        data_seed = child_seed_seqs[0]
        data_rng = np.random.default_rng(data_seed)
        # Create integer seed for test randomness
        test_seed = child_seed_seqs[1].generate_state(1)
        
        [X1, X2] = util_sampling.generate_samples(args,data_rng)
        
        for group in test_groups:
            group_results_dict[group] = util_classes.GroupResults(args.n_tests, args.B, fname_group[t][group], file_exists_group[t][group], 1, lam)
            
        for group in test_groups:
            group_results_dict[group].set_compute_group(no_compute[group], recompute[group])
            group_results_dict[group].set_group_names(args.estimators[group], args.estimator_names[group], args.estimator_labels[group])
            group_results_dict[group].set_compute(no_compute[group], recompute[group], fname[t][group], file_exists[t][group])
            print(f'compute_group for {group}: {group_results_dict[group].compute_group}, file_exists for {group}: {group_results_dict[group].file_exists}')
            
        # Compute complete permutation tests
        complete_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict,args)
                
        # Compute cheap permutation tests
        cheap_perm_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict,args)
        
        # Compute cross MMD tests
        cross_mmd_test(X1,X2,args.alpha,lam,test_seed,group_results_dict,args)
        
        # Compute random Fourier features permutation tests
        rff_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict,args)
        
        # Compute cheap random Fourier features permutation tests
        cheap_rff_test(X1,X2,args.B,args.alpha,lam,test_seed,group_results_dict,args)
        
        # Compute complete permutation tests
        wilcoxon_test(X1,X2,args.B,args.alpha,test_seed,group_results_dict,args)
        
def run_independence_tests():
    """
    Runs a total of args.n_tests independence tests and stores results into group_results_dict
    """
    # Build list of estimators
    args.estimator_list = args.estimators['ind_complete_WB'] + args.estimators['ind_cheap_perm_WB'] + args.estimators['ind_cross']
    
    # Build list of test groups
    test_groups = ['ind_complete_WB','ind_cheap_perm_WB','ind_cross']
    
    # Get directories to store the results of each test group
    resdir = util_classes_cheap.get_group_directories(args, test_groups)
    
    # Get file names to store the results
    groupname, testname = util_classes_cheap.get_test_file_names(args, test_groups, resdir, n_tests=args.n_tests, aggregated=False)
    
    fname, file_exists, fname_group, file_exists_group = util_classes.get_fname_and_file_exists(args, test_groups, resdir, groupname, testname)
        
    group_results_dict = dict()
    
    # Initialize base random number generator
    rng = np.random.default_rng(args.seed)
    # Create seed sequence for each test
    seed_seqs = rng.bit_generator._seed_seq.spawn(args.n_tests)
    for t in range(args.n_tests):
        print(f'Test number {t}')
        # From this test's seed sequence, construct two seeds,
        # one for randomness in constructing the test data and 
        # one for randomness in the test itself
        child_seed_seqs = seed_seqs[t].spawn(2)
        data_seed = child_seed_seqs[0]
        data_rng = np.random.default_rng(data_seed)
        # Create integer seed for test randomness
        test_seed = child_seed_seqs[1].generate_state(1)
        
        X = util_sampling.generate_samples(args,data_rng)
        
        for group in test_groups:
            group_results_dict[group] = util_classes.GroupResults(args.n_tests, args.B, fname_group[t][group], file_exists_group[t][group], 1, [lam_1,lam_2])
            
        for group in test_groups:
            print(f'no_compute[group]: {no_compute[group]}, recompute[group]: {recompute[group]}')
            group_results_dict[group].set_compute_group(no_compute[group], recompute[group])
            print(f'group_results_dict[group].compute_group: {group_results_dict[group].compute_group}')
            group_results_dict[group].set_group_names(args.estimators[group], args.estimator_names[group], args.estimator_labels[group])
            group_results_dict[group].set_compute(no_compute[group], recompute[group], fname[t][group], file_exists[t][group])
            print(f'compute_group for {group}: {group_results_dict[group].compute_group}, file_exists for {group}: {group_results_dict[group].file_exists}')
            
        # Compute complete permutation tests
        complete_WB_independence_test(X,args.B,args.alpha,lam_1,lam_2,test_seed,group_results_dict,args)
        
        # Compute cheap permutation tests
        cheap_perm_WB_independence_test(X,args.B,args.alpha,lam_1,lam_2,test_seed,group_results_dict,args)
        
        # Compute cross HSIC tests
        cross_independence_test(X,args.B,args.alpha,lam_1,lam_2,test_seed,group_results_dict,args)
    

if __name__ == '__main__':
    
    # Get arguments
    args = util_parser_cheap.get_args_test()
    
    if args.test_type == 'two_sample':
        if args.name == 'gaussians':
            print(f'args.mean_diff: {args.mean_diff}')
        elif args.name == 'MNIST' or args.name == 'EMNIST':
            print(f'args.p_even: {args.p_even}. args.n: {args.n}')
        
        util_tests_cheap.get_attributes_two_sample_tests(args)
        
        # Store no-compute choices for each test group
        test_groups = ['complete', 'cheap_perm', 'cross_mmd', 'rff', 'cheap_rff', 'wilcoxon']
        no_compute = dict()
        ### Change
        no_compute['complete'] = args.no_complete
        no_compute['cheap_perm'] = args.no_cheap_perm
        no_compute['cross_mmd'] = args.no_cross_mmd
        no_compute['rff'] = args.no_rff
        no_compute['cheap_rff'] = args.no_cheap_rff
        no_compute['wilcoxon'] = args.no_wilcoxon

        # Store recompute choices for each test group
        recompute = dict()
        ### Change
        recompute['complete'] = args.recompute_complete
        recompute['cheap_perm'] = args.recompute_cheap_perm
        recompute['cross_mmd'] = args.recompute_cross_mmd
        recompute['rff'] = args.recompute_rff
        recompute['cheap_rff'] = args.recompute_cheap_rff
        recompute['wilcoxon'] = args.recompute_wilcoxon
        if args.recompute_all:
            for group in test_groups:
                recompute[group] = True

        # Reset default values depending on args.name
        if args.name == 'blobs':
            args.d = 2

        if args.name == 'MNIST' or args.name == 'EMNIST':
            args.d = 49

        if args.name == 'Higgs':
            args.d = args.n_components
            if args.p_poisoning > 0:
                args.mixing = True

        if args.name == 'sine':
            args.d = 10

        if args.task_id is not None:
            set_args_for_task_id(args, args.task_id)
        
        # Compute lam
        rng = np.random.default_rng(10)
        lam_computation_samples = np.minimum(args.n,512)
        [X1, X2] = util_sampling.generate_samples(args, rng)
        lam = util_sqMMD_estimators.median_criterion(X1[:lam_computation_samples,:],X2[:lam_computation_samples,:]) 
        print(f'lambda: {lam}')

        # Run tests
        run_two_sample_tests()
        
    elif args.test_type == 'independence':
        util_tests_cheap.get_attributes_independence_tests(args)
        
        # Store no-compute choices for each test group
        test_groups = ['ind_complete_WB','ind_cheap_perm_WB','ind_cross']
        no_compute = dict()
        ### Change
        no_compute['ind_complete_WB'] = args.no_ind_complete_WB
        no_compute['ind_cheap_perm_WB'] = args.no_ind_cheap_perm_WB
        no_compute['ind_cross'] = args.no_ind_cross

        # Store recompute choices for each test group
        recompute = dict()
        ### Change
        recompute['ind_complete_WB'] = args.recompute_ind_complete_WB
        recompute['ind_cheap_perm_WB'] = args.recompute_ind_cheap_perm_WB
        recompute['ind_cross'] = args.recompute_ind_cross
        if args.recompute_all:
            for group in test_groups:
                recompute[group] = True

        if args.task_id is not None:
            set_args_for_task_id(args, args.task_id)
                
        # Compute lam
        rng = np.random.default_rng(10)
        lam_computation_samples = np.minimum(args.n,512)
        X = util_sampling.generate_samples(args, rng)
        lam_1 = util_sqMMD_estimators.median_criterion_one_sample(X[:lam_computation_samples,:args.d])
        lam_2 = util_sqMMD_estimators.median_criterion_one_sample(X[:lam_computation_samples,args.d:]) 
        print(f'lambda_1: {lam_1}, lambda_2: {lam_2}')

        # Run tests
        run_independence_tests()
        
    print(f'args.seed: {args.seed}')
            
    
