import numpy as np
import time
import math
import os
import argparse
import pickle
import scipy
from cheaper import gaussianc, cttc
from cheaper import ctt
from cheaper.cttc import signed_matrix_sum, permutation_matrix_sum
from functools import partial
from itertools import product
import util_sqMMD_estimators

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Constants %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
SQRT_2 = np.sqrt(2)

"""
%%%%%%%%%%%%%%%%%%%%%%%% Squared MMD estimators %%%%%%%%%%%%%%%%%%%%%%%%%%
"""

def complete(X1,X2,B,lam,group_results_dict,seed=None,name="gauss"):
    """
    Computation of the block squared MMD for several Rademacher vectors: https://proceedings.neurips.cc/paper/2013/file/a49e9411d64ff53eccfdd09ad10a15b3-Paper.pdf
    Returns float
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    B: number of Rademacher vectors (int)
    lam: bandwidth (float)
    block_size: size of each block
    group_results_dict: object used to store results
    name: name of the kernel; implemented for Gaussian kernels
    """
    if name=="gauss":
        n_samples = X1.shape[0]
        
        test_rng = np.random.default_rng(seed)
        
        epsilon = 2*test_rng.integers(2, size=(n_samples,B))-1
        
        split_values = np.zeros(B+1)
        sqMMD_values = np.zeros(B+1)
        
        time = gaussianc.block_sqMMD_gaussian_Rademacher(X1,X2,lam**2,n_samples,epsilon,split_values,sqMMD_values)
        
        # Store results
        group_results_dict['complete'].group_tests[str(n_samples)].estimator_values[:] = sqMMD_values
        group_results_dict['complete'].group_tests[str(n_samples)].total_times = time 
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))

def cheap_perm(X1,X2,B,s,lam,group_results_dict,kernel="gauss",null_seed=None,save_complete=False):
    """Cheap permutations two-sample test with sample sequences X1 and X2
    and auxiliary kernel k' = target kernel k.
        
    Note: Assumes that bin_size = max(1, (n1+n2)//(2s)) evenly divides n1 and n2
    
    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      B: number of permutations (int)
      s: total number of compression bins will be num_bins = min(2*s, n1+n2);
        X1 will be divided into num_bins * n1 / (n1 + n2) compression bins;
        X2 will be divided into num_bins * n2 / (n1 + n2) compression bins
      lam: positive kernel bandwidth 
      kernel: kernel name; valid options include "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      null_seed: seed used to initialize random number generator for
        randomness in simulating null; can be any valid input to 
        numpy.random.default_rng
      statistic_seed: seed used to initialize random number generator for
        randomness in computing test statistic; can be any valid input to 
        numpy.random.default_rng
    
    """
    # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
    
    # Store squared bandwidth in an array
    lam_sqd = np.array([lam**2])
    
    # Number of sample poins
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    # Number of bins per dataset
    num_bins_total = min(2*s, n1+n2)
    bin_size = (n1+n2) // num_bins_total
    num_bins1 = n1 // bin_size
    num_bins2 = num_bins_total - num_bins1

    # Compute sum of Gaussian kernel evaluations between each pair
    # of coresets
    avg_matrix = np.empty((num_bins_total,num_bins_total))
    gaussianc.sum_gaussian_kernel_by_bin(X1, X2, lam_sqd, avg_matrix)
    # Normalize each sum by the number of kernel evaluations
    avg_matrix /= (bin_size**2)

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)
    
    # Compute permuted squared MMD values
    estimator_values = np.empty(B+1)
    for b in range(B+1):
        # First select permuted order
        if b == B:
            # Store original order in final permutation bin
            perm_signs = np.arange(num_bins_total, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            perm_signs = test_rng.permutation(num_bins_total)
        # Then assign a +/-1 sign to each datapoint indicating if
        # bin was assigned a permutation index < num_bins1
        perm_signs = (perm_signs < num_bins1)*2-1

        # Finally compute the unnormalized squared MMD as a signed matrix sum
        # sum_{i,j} sign(i) sign(j) avg_matrix[i,j] 
        estimator_values[b] = (
            signed_matrix_sum(avg_matrix, perm_signs) )

    # Normalize squared MMDs
    estimator_values /= (num_bins1*num_bins2)
    print(estimator_values)

    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
    
    # Store results
    if save_complete:
        group_results_dict['complete'].group_tests[str(s)].estimator_values[:] = estimator_values
        group_results_dict['complete'].group_tests[str(s)].total_times = total_time
    else:
        group_results_dict['cheap_perm'].group_tests[str(s)].estimator_values[:] = estimator_values
        group_results_dict['cheap_perm'].group_tests[str(s)].total_times = total_time
    
def cross_mmd(X1,X2,lam,alpha,group_results_dict,kernel="gauss"):
    
    # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
    
    # Store squared bandwidth in an array
    lam_sqd = np.array([lam**2])
    
    # Number of sample points
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n1a = n1 // 2
    n2a = n2 // 2
    n1b = n1 - n1a
    n2b = n2 - n2a
    
    U_x = np.empty(n1a)
    U_y = np.empty(n2a)
    results = np.empty(3)
    gaussianc.cross_mmd(X1,X2,lam_sqd,n1a,n2a,n1b,n2b,U_x,U_y,results)
    
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
    
    group_results_dict['cross_mmd'].group_tests[str(n1)].statistic_values = results[0]
    group_results_dict['cross_mmd'].group_tests[str(n1)].total_times = total_time
    
    # Define the Gaussian inverse CDF and compute the inverse CDF value for 1-alpha 
    invPhi = lambda x: scipy.stats.norm.ppf(x)
    stdev_factor = invPhi(1-alpha)
        
    group_results_dict['cross_mmd'].group_tests[str(n1)].threshold_values = stdev_factor
    

def rff(X1,X2,r,B,lam,group_results_dict,kernel="gauss",
        null_seed=None,statistic_seed=None):
    """Random Fourier Features two-sample test with sample sequences X1 and X2.
    
    Runs RFF two-sample test on the sample sequences
    X1 and X2 and returns the results as a TestResults object.
    
    Args:
      X1: 2D array of size (n1,d)
      X2: 2D array of size (n2,d)
      r: number of random Fourier features; positive integer
      B: number of permutations (int)
      lam: positive kernel bandwidth (float)
      kernel: kernel name; valid options include "gauss" for Gaussian kernel
        exp(-||x-y||^2/lam^2)
      null_seed: seed used to initialize random number generator for
        randomness in simulating null; can be any valid input to 
        numpy.random.default_rng
      statistic_seed: seed used to initialize random number generator for
        randomness in computing test statistic; can be any valid input to 
        numpy.random.default_rng
    """
    # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
        
    # Store squared bandwidth in an array
    lam_sqd = np.array([lam**2])
    
    # Number of sample points
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    n_both = n1 + n2
    
    #
    # RFF phase
    #

    # Prepare random number generator
    stat_rng = np.random.default_rng(statistic_seed)

    # Get random Fourier features  
    # for the concatenated datasets [hatX1; hatX2]
    features = ctt.kernel_features(X1,X2,r,lam=lam,kernel=kernel,
                               feat_type="rff",seed=stat_rng,bin_size=1)

    #
    # Permutation phase
    #

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)

    # Compute permuted squared MMD estimates
    estimator_values = np.empty(B+1)
    for b in range(B+1):
        # First select permuted order
        if b == B:
            # Store original order in final permutation bin
            perm_signs = np.arange(n_both, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            perm_signs = test_rng.permutation(n_both)
        # Then assign a +/-1 sign to each datapoint indicating if
        # bin was assigned a permutation index < perm_bins1
        perm_signs = (perm_signs < n1)*2-1

        # Compute the unnormalized squared MMD as a signed vector squared norm
        # ||sum_{i} sign(i) features[i,:]||^2 
        estimator_values[b] = (
            cttc.signed_vector_sqd_norm(features, perm_signs))

    # Normalize squared MMDs
    estimator_values /= (n1*n2)
    
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
    
    # Store results
    group_results_dict['rff'].group_tests[str(r)].estimator_values[:] = estimator_values
    group_results_dict['rff'].group_tests[str(r)].total_times = total_time
    
def cheap_rff(X1,X2,r,s,B,lam,group_results_dict,kernel="gauss",
        null_seed=None,statistic_seed=None):
    
     # Measure runtime
    time_0 = time.time()
    
    if kernel != "gauss":
        raise ValueError(f"Unsupported kernel name {kernel}")
    
    # Store squared bandwidth in an array
    lam_sqd = np.array([lam**2])
    
    # Number of sample points
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    # Number of permutation bins per dataset
    n_both = n1 + n2
    perm_bins = min(2*s, n_both)
    perm_bins1 = n1 * perm_bins // n_both
    perm_bins2 = perm_bins - perm_bins1
    
    #
    # RFF phase
    #
    
    # Prepare random number generator
    stat_rng = np.random.default_rng(statistic_seed)

    # Get average random Fourier features per permutation bin 
    # for the concatenated datasets [hatX1; hatX2]
    perm_bin_size = X1.shape[0] // perm_bins1
    features = ctt.kernel_features(
        X1,X2,r,lam=lam,kernel=kernel,
        seed=stat_rng,bin_size=perm_bin_size)

    #
    # Permutation phase
    #

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)

    # Compute RFF squared MMD estimate for original data ordering
    # and for each permuted / signed version
    estimator_values = np.empty(B+1)
    
    # If number of features is larger than number of permutation bins
    # form perm_bins by perm_bins linear kernel matrix explicitly
    form_kernel_matrix = (r >= perm_bins) 
    if form_kernel_matrix:
        # Compute the linear kernel matrix features * features^t
        K = np.empty((perm_bins, perm_bins))
        cttc.linear_kernel_same(features, K)

    for b in range(B+1):
        # First select permuted order
        if b == B:
            # Store original order in final permutation bin
            perm_signs = np.arange(perm_bins, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            perm_signs = test_rng.permutation(perm_bins)
        # Then assign a +/-1 sign to each datapoint indicating if
        # bin was assigned a permutation index < perm_bins1
        perm_signs = (perm_signs < perm_bins1)*2-1

        if form_kernel_matrix:
            # Compute the unnormalized squared MMD as a signed sum
            # sum_{i,j} sign(i) sign(j) K[i,j] 
            estimator_values[b] = (
                cttc.signed_matrix_sum(K, perm_signs))
        else:
            # Compute the unnormalized squared MMD as a signed vector squared norm
            # ||sum_{i} sign(i) features[i,:]||^2 
            estimator_values[b] = (
                cttc.signed_vector_sqd_norm(features, perm_signs))

    # Normalize squared MMDs
    estimator_values /= (perm_bins1*perm_bins2)
    
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
    
    # Store results
    group_results_dict['cheap_rff'].group_tests[str(r)+'_'+str(s)].estimator_values[:] = estimator_values
    group_results_dict['cheap_rff'].group_tests[str(r)+'_'+str(s)].total_times = total_time
    
def wilcoxon(X1,X2,B,s,group_results_dict,null_seed=None):
    """
    Computation of the Wilcoxon-Mann-Whitney test
    Returns float
    
    X1: 2D array of size (n1,d)
    X2: 2D array of size (n2,d)
    B: number of Rademacher vectors (int)
    s: number of bins
    group_results_dict: object used to store results
    """
        
    # Measure runtime
    time_0 = time.time()
    
    # Number of sample poins
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    
    # Number of bins per dataset
    num_bins_total = min(2*s, n1+n2)
    bin_size = (n1+n2) // num_bins_total
    num_bins1 = n1 // bin_size
    num_bins2 = num_bins_total - num_bins1

    # Compute sum of Gaussian kernel evaluations between each pair
    # of coresets
    avg_matrix = np.empty((num_bins_total,num_bins_total))
    print(f'X1.shape: {X1.shape}, X2.shape: {X2.shape}, avg_matrix.shape: {avg_matrix.shape}')
    gaussianc.sum_order_kernel_by_bin(np.squeeze(X1), np.squeeze(X2), avg_matrix)
    # Normalize each sum by the number of kernel evaluations
    avg_matrix /= (bin_size**2)

    # Initialize generator for generating permutations
    test_rng = np.random.default_rng(null_seed)
    
    # Compute permuted squared MMD values
    estimator_values = np.empty(B+1)
    for b in range(B+1):
        # First select permuted order
        if b == B:
            # Store original order in final permutation bin
            permutation = np.arange(num_bins_total, dtype=int)
        else:
            # Generate random permutation of num_bins_total indices
            permutation = test_rng.permutation(num_bins_total)
#         # Then assign a +/-1 sign to each datapoint indicating if
#         # bin was assigned a permutation index < num_bins1
#         perm_signs = (perm_signs < num_bins1)*2-1

#         # Finally compute the unnormalized squared MMD as a signed matrix sum
#         # sum_{i,j} sign(i) sign(j) avg_matrix[i,j] 
#         estimator_values[b] = (
#             signed_matrix_sum(avg_matrix, perm_signs) )
        estimator_values[b] = (
            permutation_matrix_sum(avg_matrix, permutation) )

    # Normalize estimator values
    estimator_values /= (num_bins1*num_bins2)
    
    # Flip order
    estimator_values = 1 - estimator_values
    
    # Measure runtime
    time_1 = time.time()
    total_time = time_1 - time_0
    
    # Store results
    group_results_dict['wilcoxon'].group_tests[str(s)].estimator_values[:] = estimator_values
    group_results_dict['wilcoxon'].group_tests[str(s)].total_times = total_time
    
def complete_independence(X,B,lam_1,lam_2,group_results_dict,seed=None,name="gauss"):
    """
    Computation of the complete independence test
    Returns float
    
    X: 2D array of size (n,d)
    B: number of Rademacher vectors (int)
    lam_1: bandwidth (float) for first component
    lam_2: bandwidth (float) for second component
    group_results_dict: object used to store results
    name: name of the kernel; implemented for Gaussian kernels
    """
     # Measure runtime
    time_0 = time.time()
    
    if name=="gauss":
        n_samples = X.shape[0]
        n_over_2 = int(n_samples//2)
        two_d = X.shape[1]
        d = int(two_d//2)
        
        test_rng = np.random.default_rng(seed)
        
        epsilon = test_rng.integers(2, size=(n_over_2,B+1))
        epsilon[:,B] = 1
        pi = np.zeros((n_samples,B+1)).astype(int)
        
        K1 = np.zeros((n_samples,n_samples))
        K2 = np.zeros((n_samples,n_samples))
        meanK1 = np.zeros(n_samples)
        meanK2 = np.zeros(n_samples)
        statistics = np.zeros(B+1)
        
        gaussianc.independence_test_WB(X[:,:d],X[:,d:],epsilon,n_samples,B,lam_1**2,lam_2**2,pi,K1,K2,meanK1,meanK2,statistics)
        
        # Measure runtime
        time_1 = time.time()
        total_time = time_1 - time_0
        
        # Store results
        group_results_dict['ind_complete_WB'].group_tests[str(n_samples)].estimator_values[:] = statistics
        group_results_dict['ind_complete_WB'].group_tests[str(n_samples)].total_times = total_time
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name)) 
        
def cheap_perm_independence(X,B,s,lam_1,lam_2,group_results_dict,seed=None,name="gauss",save_complete=False):
    """
    Computation of the cheap permutations independence test
    Returns float
    
    X: 2D array of size (n,d)
    B: number of Rademacher vectors (int)
    s: total number of bins
    lam_1: bandwidth (float) for first component
    lam_2: bandwidth (float) for second component
    group_results_dict: object used to store results
    name: name of the kernel; implemented for Gaussian kernels
    """
    # Measure runtime
    time_0 = time.time()
    
    if name=="gauss":
        n_samples = X.shape[0]
        n_over_2 = int(n_samples//2)
        s_over_2 = int(s//2)
        two_d = X.shape[1]
        d = int(two_d//2)
        m = int(n_samples//s)
        
        V = np.zeros((s_over_2,s_over_2,2,2)) 
        # Create swap mapping for each index in [n]
        M = np.zeros((n_samples,2)) 
        M[:,0] = np.arange(n_samples)
        M[::2,0] += 1
        M[1::2,0] -= 1
        ###
        # start = 0; stop = m
        # for i in range(s_over_2):
        #     M[start:stop,0] += m
        #     start = stop; stop += m
        #     M[start:stop,0] -= m
        #     start = stop; stop += m
        ###
        M[:,1] = np.arange(n_samples)
        M = M.astype(int)
        
        test_rng = np.random.default_rng(seed)
        
        epsilon = test_rng.integers(2, size=(s_over_2,B+1))
        epsilon[:,B] = 1
        epsilon = epsilon.astype(int)
        
        K1 = np.zeros((n_samples,n_samples))
        K2 = np.zeros((n_samples,n_samples))
        meanK1 = np.zeros(n_samples)
        meanK2 = np.zeros(n_samples)
        statistics = np.zeros(B+1)
       
        # Old, slower version of cheap independence tests
        # gaussianc.cheap_independence_test_WB(X[:,:d],X[:,d:],epsilon,M,n_samples,B,s,lam_1**2,lam_2**2,pi,V,
        #                                      K1,K2,meanK1,meanK2,statistics)
    
        # New, faster version of cheap independence tests
        if False:
            gaussianc.new_cheap_independence_test_WB(X[:,:d],X[:,d:],epsilon,M,n_samples,B,s,lam_1**2,lam_2**2,V,
                                                     K1,K2,meanK1,meanK2,statistics)
        else:
            gaussianc.norecenter_cheap_independence_test_WB(X[:,:d],X[:,d:],epsilon,M,n_samples,B,s,lam_1**2,lam_2**2,V, 
                                                            K1,K2,
                                                            meanK1,meanK2,statistics)

        # Measure runtime
        time_1 = time.time()
        total_time = time_1 - time_0
        
        # Store results
        print(f'statistics s={s}: {statistics}')
        print(f'total_time s={s}: {total_time}')
        if save_complete:
            group_results_dict['ind_complete_WB'].group_tests[str(s)].estimator_values[:] = statistics
            group_results_dict['ind_complete_WB'].group_tests[str(s)].total_times = total_time
        else:
            group_results_dict['ind_cheap_perm_WB'].group_tests[str(s)].estimator_values[:] = statistics
            group_results_dict['ind_cheap_perm_WB'].group_tests[str(s)].total_times = total_time
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))
        
def cross_hsic_independence(X,B,lam_1,lam_2,alpha,group_results_dict,seed=None,name="gauss"):
    """
    Computation of the cross HSIC independence test
    Returns float
    
    X: 2D array of size (n,d)
    B: number of Rademacher vectors (int)
    lam_1: bandwidth (float) for first component
    lam_2: bandwidth (float) for second component
    alpha: nominal level (float)
    group_results_dict: object used to store results
    name: name of the kernel; implemented for Gaussian kernels
    """
     # Measure runtime
    time_0 = time.time()
    
    if name=="gauss":
        n_samples = X.shape[0]
        n_over_2 = int(n_samples//2)
        two_d = X.shape[1]
        d = int(two_d//2)
        
        K = np.zeros((n_over_2,n_over_2))
        L = np.zeros((n_over_2,n_over_2))
        u_K = np.zeros(n_over_2)
        l_K = np.zeros(n_over_2)
        u_L = np.zeros(n_over_2)
        l_L = np.zeros(n_over_2)
        K_circ_L = np.zeros((n_over_2,n_over_2))
        KL_u = np.zeros(n_over_2)
        LK_u = np.zeros(n_over_2)
        K_circ_L_l = np.zeros(n_over_2)
        w = np.zeros(n_over_2)
        results = np.zeros(B+2)
        gaussianc.cross_independence_test(X[:,:d],X[:,d:],n_samples,lam_1**2,lam_2**2,K,L,u_K,l_K,u_L,l_L,K_circ_L,KL_u,LK_u,
                                          K_circ_L_l,w,results)
    
        # Measure runtime
        time_1 = time.time()
        total_time = time_1 - time_0
    
        print(f'total_time: {total_time}')
        group_results_dict['ind_cross'].group_tests[str(n_samples)].statistic_values = results[0]*np.sqrt(n_over_2)/np.sqrt(results[1])
        group_results_dict['ind_cross'].group_tests[str(n_samples)].total_times = total_time
        
        # Define the Gaussian inverse CDF and compute the inverse CDF value for 1-alpha 
        invPhi = lambda x: scipy.stats.norm.ppf(x)
        stdev_factor = invPhi(1-alpha)

        group_results_dict['ind_cross'].group_tests[str(n_samples)].threshold_values = stdev_factor
        
    else:
        raise ValueError("Unrecognized kernel name {}".format(name))
        
