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

def generate_samples(args,rng,seed=None,f_theta_seed=None):
    """
    Depending on args.name, returns two sets of samples for the Gaussians, 
      Blobs or MNIST distributions
    """
    if args.name == 'gaussians':
        return generate_samples_gaussians(args,rng)
    elif args.name == 'blobs':
        X1 = generate_samples_blobs(args,0,rng)
        cov = (args.epsilon-1)/(args.epsilon+1)
        X2 = generate_samples_blobs(args,cov,rng)
        return X1, X2
    elif args.name == 'EMNIST':
        return generate_samples_EMNIST(args,rng)
    elif args.name == 'Higgs':
        return generate_samples_Higgs(args,rng)
    elif args.name == 'sine':
        return generate_samples_sine(args,rng)
    if args.name == 'gaussians_cov':
        return generate_samples_gaussians_cov(args,rng)
