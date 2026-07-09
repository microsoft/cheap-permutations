import numpy as np
import time
import math
import os
import argparse
import pickle
import scipy
import urllib.request
from functools import partial
from sklearn.datasets import fetch_openml
from itertools import product

"""
%%%%%%%%%%% Sampling functions %%%%%%%%%%%
"""
    
def generate_samples_gaussians(args,rng):
    """
    Generates two sets of multivariate Gaussians in dimension args.d, with identity covariances
      and means that are args.mean_diff apart
    """
    mean_1 = np.zeros(args.d)
    mean_2 = np.zeros(args.d)
    mean_1[args.d-1] = args.mean_diff/2
    mean_2[args.d-1] = -args.mean_diff/2
    mean_1 = tuple(mean_1)
    mean_2 = tuple(mean_2)
    cov_1 = np.eye(args.d)
    cov_2 = np.eye(args.d)
    
    X1 = rng.multivariate_normal(mean_1, cov_1, args.n) 
    X2 = rng.multivariate_normal(mean_2, cov_2, args.n)
    return X1, X2

def generate_samples_gaussians_cov(args,rng):
    """
    Generates two sets of multivariate Gaussians in dimension args.d, with identity covariances
      and means that are args.mean_diff apart
    """
    mean = np.zeros(2*args.d)
    mean = tuple(mean)
    cov = np.eye(2*args.d)
    for i in range(args.d):
        cov[i,args.d+i] = args.cross_covariance
        cov[args.d+i,i] = args.cross_covariance
    
    X = rng.multivariate_normal(mean, cov, args.n) 
    return X

def generate_samples_Higgs(args,rng):
    """
    Generates two sets P, Q of samples from the Higgs dataset
    """
    start = time.time()
    
    # Load data
    if not args.mixing:
        if not args.null:
            data1 = pickle.load(open('./data/higgs_14_17_0.pckl', 'rb'))
            data2 = pickle.load(open('./data/higgs_14_17_1.pckl', 'rb'))
            #data1 = pickle.load(open('./data/higgs_14_21_0.pckl', 'rb'))
            #data2 = pickle.load(open('./data/higgs_14_21_1.pckl', 'rb'))
            #data1 = data[0]
            #data2 = data[1]
        else:
            #data1 = data[0]
            #data2 = data[0]
            data1 = pickle.load(open('./data/higgs_14_17_0.pckl', 'rb'))
            #data1 = pickle.load(open('./data/higgs_14_21_0.pckl', 'rb'))
            data2 = data1
            
        idx_1 = rng.integers(data1.shape[0], size=args.n)
        X1 = data1[idx_1,:]    

        idx_2 = rng.integers(data2.shape[0], size=args.n)
        X2 = data2[idx_2,:]
        
        return X1, X2

        #X1 = data1[idx_1]
        #X2 = data2[idx_2]

        #X1 = np.expand_dims(X1, axis=1)
        #X2 = np.expand_dims(X2, axis=1)
    else:
        data = pickle.load(open('./data/HIGGS_TST.pckl', 'rb'))
        data1 = data[0]
        data2 = data[1]    

        idx_1 = rng.integers(data1.shape[0], size=args.n)
        X1 = data1[idx_1,:args.d]

        n_poisoned = np.sum(rng.binomial(1, args.p_poisoning, size=args.n))
        print(f'args.n: {args.n}. n_poisoned: {n_poisoned}')
        idx_2_poisoned = rng.integers(data1.shape[0], size=n_poisoned)
        idx_2_true = rng.integers(data2.shape[0], size=args.n-n_poisoned)

        print(f'data1[idx_2_poisoned,:].shape: {data1[idx_2_poisoned,:args.d].shape}')
        print(f'data2[idx_2_true,:].shape: {data2[idx_2_true,:args.d].shape}')

        X2 = np.concatenate((data1[idx_2_poisoned,:args.d],data2[idx_2_true,:args.d]), axis=0)
        rng.shuffle(X2)

        print(f'X1.shape: {X1.shape}')
        print(f'X2.shape: {X2.shape}')

        end = time.time()
        print(f'Time elapsed: {end-start}.')

        return X1, X2

def generate_samples(args,rng,seed=None,f_theta_seed=None):
    """
    Depending on args.name, returns two sets of samples for the Gaussians, 
      Blobs or MNIST distributions
    """
    if args.name == 'gaussians':
        return generate_samples_gaussians(args,rng)
    elif args.name == 'Higgs':
        return generate_samples_Higgs(args,rng)
    if args.name == 'gaussians_cov':
        return generate_samples_gaussians_cov(args,rng)
