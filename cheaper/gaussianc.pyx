"""Gaussian kernel functionality.

Cython implementation of functions involving Gaussian kernel evaluation.
"""
import numpy as np
cimport numpy as np
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport cython
from libc.math cimport sqrt, log, exp, cos
from libc.stdlib cimport rand, RAND_MAX, srand
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
# It's necessary to call "import_array" if you use any part of the
# numpy PyArray_* API. From Cython 3, accessing attributes like
# ".shape" on a typed Numpy array use this API. 
np.import_array()

'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Gaussian Kernel Functionality %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
'''

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cdef double gaussian_kernel_two_points(const double[:] X1,
                                       const double[:] X2,
                                       const double[:] lam_sqd) noexcept nogil:
    """
    Computes a sum of Gaussian kernels sum_j exp(-||X1-X2||_2^2/lam_sqd[j]) 
    between two points X1 and X2
    
    Args:
      X1: array of size d
      X2: array of size d
      lam_sqd: array of squared kernel bandwidths
    """
    
    cdef long d = X1.shape[0]
    cdef long num_kernels = lam_sqd.shape[0]
    
    cdef long j
    cdef double arg, kernel_sum
    
    # Compute the squared Euclidean distance between X1 and X2
    arg = 0
    for j in range(d):
        arg += (X1[j]-X2[j])**2
    
    # Compute the kernel sum
    kernel_sum = exp(-arg/lam_sqd[0])
    for j in range(1,num_kernels):
        kernel_sum += exp(-arg/lam_sqd[j])
    return(kernel_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void gaussian_kernel(const double[:,:] X1,
                           const double[:,:] X2,
                           const double lam_sqd,
                           double[:,:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between each rows of X1 and each rows of X2
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Matrix of size (n1,n2) to store kernel matrix
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    
    for i in range(n1):
        for j in range(n2):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X2[j,k])**2
            K[i,j] = exp(-arg/lam_sqd)
                
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void gaussian_kernel_same(const double[:,:] X1,
                                const double lam_sqd,
                                double[:,:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between each pair of rows in X1
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Empty matrix of size (n1,n1) to store kernel matrix
    """
    
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    
    for i in range(n1):
        K[i,i] = 1
        for j in range(i):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X1[j,k])**2
            K[i,j] = exp(-arg/lam_sqd)
            K[j,i] = K[i,j]
            
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void gaussian_kernel_by_row(const double[:] X1,
                                  const double[:,:] X2,
                                  const double lam_sqd,
                                  double[:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between X1 and each row of X2
    and stores in K
    
    Args:
      X1: Vector of size d
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Vector of size d to store kernel values
    """
    
    cdef long n2 = X2.shape[0]
    cdef long d = X2.shape[1]
    
    cdef long i
    cdef double arg
    for i in range(n2):
        arg = 0
        for k in range(d):
            arg += (X1[k]-X2[i,k])**2
        K[i] = exp(-arg/lam_sqd)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_gaussian_kernel(const double[:,:] X1,
                                 const double[:,:] X2,
                                 const double lam_sqd) noexcept nogil:
    """
    Returns the sum of Gaussian kernel evaluations between each row of X1 
    and each row of X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    for i in range(n1):
        for j in range(n2):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X2[j,k])**2
            total_sum += exp(-arg/lam_sqd)
            
    return(total_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_gaussian_kernel_vector_matrix(const double[:,:] X1,
                                               const double[:,:] X2,
                                               const long r1,
                                               const long r2,
                                               const double lam_sqd) noexcept nogil:
    """
    Returns the sum of Gaussian kernel evaluations between the row r2 of X1 
    and each row of X2 starting from the r2-th row
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      r1: long,
      r2: long,
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n2 = X2.shape[0]
    cdef long d = X2.shape[1]
    
    cdef long i, j
    cdef double arg
    for j in range(r2,n2):
        arg = 0
        for k in range(d):
            arg += (X1[r1,k]-X2[j,k])**2
        total_sum += exp(-arg/lam_sqd)
            
    return(total_sum)
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_gaussian_kernel_same(const double[:,:] X1,
                                      const double lam_sqd) noexcept nogil:
    """
    Returns the sum of Gaussian kernel evaluations between each pair of 
    rows of X1
    
    Args:
      X1: Matrix of size (n1,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    for i in range(n1):
        for j in range(i+1):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X1[j,k])**2
            if j < i:    
                total_sum += 2*exp(-arg/lam_sqd)
            else:
                total_sum += exp(-arg/lam_sqd)
            
    return(total_sum)        
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_gaussian_kernel_linear_eval(const double[:,:] X1,
                                             const double[:,:] X2,
                                             const double lam_sqd) noexcept nogil:
    """
    Computes the sum of Gaussian kernel evaluations between the 
    i-th row of X1 and the i-th row of X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i
    cdef double arg
    for i in range(n):
        arg = 0
        for k in range(d):
            arg += (X1[i,k]-X2[i,k])**2
        total_sum += exp(-arg/lam_sqd)
            
    return(total_sum)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_by_bin(const double[:,:] X1,
                                      const double[:,:] X2,
                                      const double lam_sqd,
                                      double[:,:] K_sum) noexcept nogil:
    """
    Partitions the rows of X1 and the rows of X2 into bins of size 
    bin_size = (n1+n2) // num_bins, computes sum_gaussian_kernel for 
    each pair of bins, and stores the result in K_sum
    
    Note: Assumes that bin_size evenly divides n1 and n2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K_sum: Matrix of size (num_bins,num_bins) to store kernel sums
    """
    
    cdef long num_bins = K_sum.shape[0]
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long bin_size = (n1+n2) // num_bins
    cdef long num_bins1 = n1 // bin_size
    cdef long num_bins2 = n2 // bin_size

    # Compute kernel sum for each pair of X1 bins
    cdef const double[:,:] bin1, bin2
    cdef long k1, k2
    cdef long bin1_start = 0
    cdef long bin1_end = bin_size
    cdef long bin2_start
    cdef long bin2_end
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X1[bin2_start:bin2_end,:]
            K_sum[k1,k2] = sum_gaussian_kernel(bin1,bin2,lam_sqd)
            K_sum[k2,k1] = K_sum[k1,k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        K_sum[k1,k1] = sum_gaussian_kernel_same(bin1,lam_sqd)
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sum for each pair of X2 bins
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins2):
        bin1 = X2[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X2[bin2_start:bin2_end,:]
            K_sum[num_bins1+k1,num_bins1+k2] = sum_gaussian_kernel(bin1,bin2,lam_sqd)
            K_sum[num_bins1+k2,num_bins1+k1] = K_sum[num_bins1+k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        K_sum[num_bins1+k1,num_bins1+k1] = sum_gaussian_kernel_same(bin1,lam_sqd)
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sum between each pairing of one X1 and one X2 coreset
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(num_bins2):
            bin2 = X2[bin2_start:bin2_end,:]
            K_sum[k1,num_bins1+k2] = sum_gaussian_kernel(bin1,bin2,lam_sqd)
            K_sum[num_bins1+k2,k1] = K_sum[k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        bin1_start = bin1_end
        bin1_end += bin_size
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_by_bin_aggregated(const double[:,:] X1,
                                                 const double[:,:] X2,
                                                 const double[:] lam_sqd,
                                                 double[:,:,:] K_sum) noexcept nogil:
    """
    Partitions the rows of X1 and the rows of X2 into bins of size 
    bin_size = (n1+n2) // num_bins, computes sum_gaussian_kernel for 
    each pair of bins and each squared bandwidth, and stores the result 
    in K_sum. 
    
    Note: Only populates the lower-triangular entries K_sum[k1,k2,:] 
          for k1 >= k2.
    
    Note: Assumes that bin_size evenly divides n1 and n2.
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: Length l vector of kernel squared bandwidths for the kernel 
        exp(-||x-y||_2^2/lam_sqd)
      K_sum: Zeros matrix of size (num_bins,num_bins,l) to store kernel sums
    """
    
    cdef long num_bins = K_sum.shape[0]
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long bin_size = (n1+n2) // num_bins
    cdef long num_bins1 = n1 // bin_size
    cdef long num_bins2 = n2 // bin_size
    
    # Compute kernel sums for each pair of X1 bins
    cdef const double[:,:] bin1, bin2
    cdef long k1, k2
    cdef long bin1_start = 0
    cdef long bin1_end = bin_size
    cdef long bin2_start
    cdef long bin2_end
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X1[bin2_start:bin2_end,:]
            sum_gaussian_kernel_aggregated(bin1,bin2,lam_sqd,K_sum[k1,k2,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        sum_gaussian_kernel_same_aggregated(bin1,lam_sqd,K_sum[k1,k1,:])
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sums for each pair of X2 bins
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins2):
        bin1 = X2[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X2[bin2_start:bin2_end,:]
            sum_gaussian_kernel_aggregated(bin1,bin2,lam_sqd,K_sum[num_bins1+k1,num_bins1+k2,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        sum_gaussian_kernel_same_aggregated(bin1,lam_sqd,K_sum[num_bins1+k1,num_bins1+k1,:])
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sums between each pairing of one X1 and one X2 coreset
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end,:]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(num_bins2):
            bin2 = X2[bin2_start:bin2_end,:]
            sum_gaussian_kernel_aggregated(bin1,bin2,lam_sqd,K_sum[num_bins1+k2,k1,:])
            bin2_start = bin2_end
            bin2_end += bin_size
        bin1_start = bin1_end
        bin1_end += bin_size

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_aggregated(const double[:,:] X1,
                                          const double[:,:] X2,
                                          const double[:] lam_sqd,
                                          double[:] results) noexcept nogil:
    """
    Computes the sum of Gaussian kernel evaluations between all rows of X1 and all rows of X2, 
    for all the squared bandwidths in lam_sqd and stores in results.
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: Vector of length L of squared bandwidths for the kernels  
        exp(-||x-y||_2^2/lam_sqd[l])
      results: Zeros vector of length L in which results will be stored
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j, k, l
    cdef double arg
    for i in range(n1):
        for j in range(n2):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X2[j,k])**2
            for l in range(len(lam_sqd)):
                results[l] += exp(-arg/lam_sqd[l])       
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_gaussian_kernel_same_aggregated(const double[:,:] X1,
                                               const double[:] lam_sqd,
                                               double[:] results) noexcept nogil:
    """
    Computes the sum of Gaussian kernel evaluations between all rows of X1 and all rows of X1,
    for all the squared bandwidths in lam_sqd and stores in results.
    
    Args:
      X1: Matrix of size (n1,d)
      lam_sqd: Vector of length L of squared bandwidths for the kernels 
        exp(-||x-y||_2^2/lam_sqd[l])
      results: Zeros vector of length L in which results will be stored
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j, k, l
    cdef double arg
    for i in range(n1):
        for j in range(i+1):
            arg = 0
            for k in range(d):
                arg += (X1[i,k]-X1[j,k])**2
            if j < i:
                for l in range(len(lam_sqd)):
                    results[l] += 2*exp(-arg/lam_sqd[l])
            else:
                for l in range(len(lam_sqd)):
                    results[l] += exp(-arg/lam_sqd[l])
                    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void order_kernel(const double[:] X1,
                        const double[:] X2,
                        long[:,:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between each rows of X1 and each rows of X2
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Matrix of size (n1,n2) to store kernel matrix
    """
    
    cdef long n1 = len(X1)
    cdef long n2 = len(X2)
#     cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    
    for i in range(n1):
        for j in range(n2):
            K[i,j] = 1 if X1[i] < X2[i] else 0
            
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void order_kernel_same(const double[:] X1,
                             long[:,:] K) noexcept nogil:
    """
    Computes the Gaussian kernel matrix between each rows of X1 and each rows of X2
    and stores in K
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K: Matrix of size (n1,n2) to store kernel matrix
    """
    
    cdef long n = len(X1)
    
    cdef long i, j
    cdef double arg
    
    for i in range(n):
        for j in range(n):
            K[i,j] = 1 if X1[i] < X1[j] else 0
            
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sum_order_kernel(const double[:] X1,
                              const double[:] X2) noexcept nogil:
    """
    Returns the sum of Gaussian kernel evaluations between each row of X1 
    and each row of X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
    """
    
    cdef double total_sum = 0
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef long i, j
    cdef double arg
    for i in range(n1):
        for j in range(n2):
            total_sum += 1 if X1[i] < X2[j] else 0
            
    return(total_sum)
            
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void sum_order_kernel_by_bin(const double[:] X1,
                                   const double[:] X2,
                                   double[:,:] K_sum) noexcept nogil:
    """
    Partitions the rows of X1 and the rows of X2 into bins of size 
    bin_size = (n1+n2) // num_bins, computes sum_gaussian_kernel for 
    each pair of bins, and stores the result in K_sum
    
    Note: Assumes that bin_size evenly divides n1 and n2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: squared bandwidth of the kernel exp(-||x-y||_2^2/lam^2)
      K_sum: Matrix of size (num_bins,num_bins) to store kernel sums
    """
    
    cdef long num_bins = K_sum.shape[0]
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long bin_size = (n1+n2) // num_bins
    cdef long num_bins1 = n1 // bin_size
    cdef long num_bins2 = n2 // bin_size

    # Compute kernel sum for each pair of X1 bins
    cdef const double[:] bin1, bin2
    cdef long k1, k2
    cdef long bin1_start = 0
    cdef long bin1_end = bin_size
    cdef long bin2_start
    cdef long bin2_end
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X1[bin2_start:bin2_end]
            K_sum[k1,k2] = sum_order_kernel(bin1,bin2)
            K_sum[k2,k1] = bin_size**2 - K_sum[k1,k2]
            bin2_start = bin2_end
            bin2_end += bin_size
#         K_sum[k1,k1] = sum_gaussian_kernel_same(bin1,lam_sqd)
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sum for each pair of X2 bins
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins2):
        bin1 = X2[bin1_start:bin1_end]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(k1):
            bin2 = X2[bin2_start:bin2_end]
            K_sum[num_bins1+k1,num_bins1+k2] = sum_order_kernel(bin1,bin2)
            K_sum[num_bins1+k2,num_bins1+k1] = bin_size**2 - K_sum[num_bins1+k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
#         K_sum[num_bins1+k1,num_bins1+k1] = sum_gaussian_kernel_same(bin1,lam_sqd)
        bin1_start = bin1_end
        bin1_end += bin_size
                    
    # Compute kernel sum between each pairing of one X1 and one X2 coreset
    bin1_start = 0
    bin1_end = bin_size
    for k1 in range(num_bins1):
        bin1 = X1[bin1_start:bin1_end]
        bin2_start = 0
        bin2_end = bin_size
        for k2 in range(num_bins2):
            bin2 = X2[bin2_start:bin2_end]
            K_sum[k1,num_bins1+k2] = sum_order_kernel(bin1,bin2)
            K_sum[num_bins1+k2,k1] = bin_size**2 - K_sum[k1,num_bins1+k2]
            bin2_start = bin2_end
            bin2_end += bin_size
        bin1_start = bin1_end
        bin1_end += bin_size

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double biased_sqMMD_gaussian(const double[:,:] X1,
                                   const double[:,:] X2,
                                   const double lam_sqd) noexcept nogil:
    """
    Computes the biased quadratic squared MMD estimator for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      lam: double, bandwidth of the kernel
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef double first_term = sum_gaussian_kernel(X1,X1,lam_sqd)
    cdef double second_term = sum_gaussian_kernel(X1,X2,lam_sqd)
    cdef double third_term = sum_gaussian_kernel(X2,X2,lam_sqd)
    cdef double bsqMMD = first_term/(n1*n1) - 2*second_term/(n1*n2) + third_term/(n2*n2)
            
    return(bsqMMD)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double unbiased_sqMMD_gaussian(const double[:,:] X1,
                                     const double[:,:] X2,
                                     const double lam_sqd) noexcept nogil:
    """
    Computes the unbiased quadratic squared MMD estimator for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    
    cdef long n1 = X1.shape[0]
    cdef long n2 = X2.shape[0]
    cdef long d = X1.shape[1]
    
    cdef double first_term = sum_gaussian_kernel_same(X1,lam_sqd)
    cdef double second_term = sum_gaussian_kernel(X1,X2,lam_sqd)
    cdef double third_term = sum_gaussian_kernel_same(X2,lam_sqd)
    cdef double extra_terms_first = sum_gaussian_kernel_linear_eval(X1,X1,lam_sqd)
    cdef double extra_terms_third = sum_gaussian_kernel_linear_eval(X2,X2,lam_sqd)
    cdef double usqMMD = (first_term-extra_terms_first)/(n1*(n1-1)) - 2*second_term/(n1*n2) + (third_term-extra_terms_third)/(n2*(n2-1))
            
    return(usqMMD)
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void block_sqMMD_gaussian(const double[:,:] X1,
                                const double[:,:] X2,
                                const double lam_sqd,
                                const long block_size,
                                double[:] results) noexcept nogil:
    """Computes the block quadratic squared MMD estimator for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      lam_sqd: double, squared bandwidth of the kernel
      block_size: long, size of blocks
    """

    cdef long n = X1.shape[0]
    # Compute ceil(n/block_size)
    cdef long n_splits = n//block_size + (n % block_size != 0)
    cdef long first_index
    cdef long last_index
    cdef double results_1

    cdef double total_sum = 0
    cdef double total_sum_sqd = 0
    cdef double split_sum
    cdef long i
    for i in range(n_splits):
        split_sum = 0
        first_index = i*block_size
        if i != n_splits - 1:
            last_index = (i+1)*block_size
        else:
            last_index = n
        for j in range(first_index,last_index):
            for k in range(first_index,j): #we only need to look at the lower triangular matrix
                h_value = h_gaussian(X1[j],X1[k],X2[j],X2[k],lam_sqd)
                split_sum += h_value
        split_sum = 2*split_sum/(1.0*(last_index-first_index))/(1.0*(last_index-first_index-1))
        total_sum += split_sum
        total_sum_sqd += split_sum**2
    results[0] = total_sum/n_splits
    if n_splits != 1:
        #results_1 = (total_sum_sqd/n_splits - (total_sum/n_splits)**2*((n_splits+1)/(n_splits-1)))/n_splits ##we use an unbiased estimate of the variance of the block values
        results_1 = (total_sum_sqd/n_splits - (total_sum/n_splits)**2)/(n_splits-1)
        if results_1 <= 0:
            results[1] = 0
        else:
            results[1] = results_1
    else:
        results[1] = 0
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void block_sqMMD_gaussian_reordered(const double[:,:] X1,
                                          const double[:,:] X2,
                                          const double lam_sqd,
                                          const long block_size,
                                          const long seed,
                                          long[:] epsilon,
                                          double[:] results) noexcept nogil:
    """Computes the block squared MMD estimator for the Gaussian kernel from samples X1 and X2, reordering before constructing blocks
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n12,d)
      lam_sqd: double, squared bandwidth of the kernel
      block_size: long, size of blocks
      seed: random seed
      epsilon: used to store random vector
    """
    #printf('epsilon cython: %u, %u\n', epsilon[0], epsilon[1])
    
    cdef long n = X1.shape[0]
    # Compute ceil(n/block_size)
    cdef long n_splits = n//block_size + (n % block_size != 0)
    cdef long first_index
    cdef long last_index
    cdef double results_2, results_3

    cdef double total_sum = 0
    cdef double total_sum_sqd = 0
    cdef double total_sum_epsilon = 0
    cdef double total_sum_sqd_epsilon = 0
    cdef double split_sum
    cdef double split_sum_epsilon
    cdef double h_value
    cdef long i, j, k
    
    srand(seed)
    
    for i in range(n_splits):
        srand(i)
        for j in range(block_size):
            epsilon[j] = rand() % 2
        split_sum = 0
        split_sum_epsilon = 0 
        first_index = i*block_size
        if i != n_splits - 1:
            last_index = (i+1)*block_size
        else:
            last_index = n
        for j in range(first_index,last_index):
            for k in range(first_index,j): #we only need to look at the lower triangular matrix
                h_value = h_gaussian(X1[j],X1[k],X2[j],X2[k],lam_sqd)
                split_sum += h_value
                split_sum_epsilon += (2.0*epsilon[j-first_index]-1.0)*(2.0*epsilon[k-first_index]-1.0)*h_value
        split_sum = 2*split_sum/(1.0*(last_index-first_index)*(last_index-first_index-1))
        split_sum_epsilon = 2*split_sum_epsilon/(1.0*(last_index-first_index)*(last_index-first_index-1))
        total_sum += split_sum
        total_sum_epsilon += split_sum_epsilon
        total_sum_sqd += split_sum**2
        total_sum_sqd_epsilon += split_sum_epsilon**2
    results[0] = total_sum/(n_splits*1.0)
    results[1] = total_sum_epsilon/(n_splits*1.0)
    if n_splits != 1:
        results_2 = (total_sum_sqd/(n_splits*1.0) - (total_sum/(1.0*n_splits))**2)/(1.0*n_splits-1) ##we use an unbiased estimate of the variance of the block values 
        if results_2 <= 0:
            results[2] = 0
        else:
            results[2] = results_2
        results_3 = (total_sum_sqd_epsilon/(n_splits*1.0) - (total_sum_epsilon/(1.0*n_splits))**2)/(1.0*n_splits-1)
        if results_3 <= 0:
            results[3] = 0
        else:
            results[3] = results_3
    else:
        results[2] = 0
        results[3] = 0
        
    printf('n_splits: %u, results[0]: %.15f, results[1]: %.15f, results[2]: %.15f, results[3]: %.15f\n', n_splits, results[0], results[1], results[2], results[3])

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double block_sqMMD_gaussian_Rademacher(const double[:,:] X1,
                                             const double[:,:] X2,
                                             const double lam_sqd,
                                             const long block_size,
                                             const long long[:,:] epsilon,
                                             double[:] split_values,
                                             double[:] sqMMD_values) noexcept nogil:
    """Computes the block squared MMD estimator for the Gaussian kernel from samples X1 and X2, 
    for different Rademacher vectors
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, bandwidth of the kernel
      block_size: long, size of blocks
      epsilon: used to store random vector
      split_values: used in the procedure
      sqMMD_values: used to store results
    """
    cdef long n = X1.shape[0]
    cdef long n_splits = n//block_size + (n % block_size != 0)
    cdef long B = epsilon.shape[1]
    
    cdef long first_index, last_index
    cdef long i, j, k, l, m
    cdef long long eps_j, eps_k
    cdef timespec ts
    cdef double start, end
    
    cdef double h_value, time
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    for i in range(n_splits):
        
        #printf('Split %u of %u\n', i, n_splits)
        for j in range(len(split_values)):
            split_values[j] = 0
        first_index = i*block_size
        if i != n_splits - 1:
            last_index = (i+1)*block_size
        else:
            last_index = n
        
        for j in range(first_index,last_index):
            for k in range(first_index,j): #we only need to look at the lower triangular matrix
                h_value = h_gaussian(X1[j],X1[k],X2[j],X2[k],lam_sqd)
                split_values[B] += h_value
                for l in range(B):
                    split_values[l] += epsilon[j,l]*epsilon[k,l]*h_value
        
        for j in range(len(split_values)):
            split_values[j] = 2*split_values[j]/(1.0*(last_index-first_index)*(last_index-first_index-1)*n_splits)
            sqMMD_values[j] += split_values[j]
     
    clock_gettime(CLOCK_REALTIME, &ts)
    end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    time = end-start
    
    printf('Test statistic value %f in time %f. n_splits: %u, block_size: %u\n', sqMMD_values[B], time, n_splits, block_size)
    
    return time

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double h_gaussian(const double[:] x1,
                        const double[:] x2,
                        const double[:] y1,
                        const double[:] y2,
                        const double lam_sqd) noexcept nogil:
    """
    Computes the kernel function h((x1,y1),(x2,y2)) = k(x1,x1) + k(y2,y2) - k(x1,y2) - k(x2,y1), where k is the Gaussian kernel
    
    Args:
      x1: vector of size d
      x2: vector of size d
      y1: vector of size d
      y2: vector of size d
      lam_sqd: squared bandwidth
    """
    
    cdef double arg_x1_x2 = 0
    cdef double arg_y1_y2 = 0
    cdef double arg_x1_y2 = 0
    cdef double arg_x2_y1 = 0
    cdef long k 
    for k in range(len(x1)):
        arg_x1_x2 += (x1[k]-x2[k])**2
        arg_y1_y2 += (y1[k]-y2[k])**2
        arg_x1_y2 += (x1[k]-y2[k])**2
        arg_x2_y1 += (x2[k]-y1[k])**2
    return(exp(-arg_x1_x2/lam_sqd) + exp(-arg_y1_y2/lam_sqd) - exp(-arg_x1_y2/lam_sqd) - exp(-arg_x2_y1/lam_sqd))

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void h_gaussian_aggregated(const double[:] x1,
                                 const double[:] x2,
                                 const double[:] y1,
                                 const double[:] y2,
                                 const double[:] lam_sqd,
                                 double[:] h_values) noexcept nogil:
    """
    Computes the kernel function h((x1,y1),(x2,y2)) = k(x1,x1) + k(y2,y2) - k(x1,y2) - k(x2,y1), where k is the Gaussian kernel,
    for all the squared bandwidths in lam_sqd
    
    Args:
      x1: vector of size d
      x2: vector of size d
      y1: vector of size d
      y2: vector of size d
      lam_sqd: squared bandwidth
      h_values: used to store results
    """
    
    cdef double arg_x1_x2 = 0
    cdef double arg_y1_y2 = 0
    cdef double arg_x1_y2 = 0
    cdef double arg_x2_y1 = 0
    cdef long k 
    for k in range(len(x1)):
        arg_x1_x2 += (x1[k]-x2[k])**2
        arg_y1_y2 += (y1[k]-y2[k])**2
        arg_x1_y2 += (x1[k]-y2[k])**2
        arg_x2_y1 += (x2[k]-y1[k])**2
    for k in range(len(lam_sqd)):
        h_values[k] = exp(-arg_x1_x2/lam_sqd[k]) + exp(-arg_y1_y2/lam_sqd[k]) - exp(-arg_x1_y2/lam_sqd[k]) - exp(-arg_x2_y1/lam_sqd[k])

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void incomplete_sqMMD_gaussian(const double[:,:] X1,
                                     const double[:,:] X2,
                                     const long long[:] incomplete_list,
                                     const long long max_l,
                                     const double lam_sqd,
                                     const long seed,
                                     double[:] sqMMD_list,
                                     double[:] sqMMD_variances,
                                     double[:] sqMMD_times) noexcept nogil:
    """
    Computes the incomplete squared MMD estimator
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      max_l: largest number of pairs (largest/last element of incomplete_list)
      lam_sqd: squared bandwidth
      seed: random seed
      sqMMD_list: used to store results
      sqMMD_variances: used to store results
      sqMMD_times: used to store results
    """
    
    cdef long n_samples = X1.shape[0]
    cdef double total_value = 0
    cdef double total_value_sqd = 0
    cdef double h_value
    cdef long incomplete_index = 0
    cdef long D0, D1
    cdef long long j
    cdef timespec ts
    cdef double start, end
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    #Set seed for rand()
    srand(seed)
    
    for j in range(max_l):
  
        #Sample pair
        D0 = rand() % n_samples 
        D1 = (D0 + 1 + rand() % (n_samples-1))%n_samples 
        
        #Compute contribution of pair
        h_value = h_gaussian(X1[D0],X1[D1],X2[D0],X2[D1],lam_sqd)
        total_value += h_value
        total_value_sqd += h_value**2
        
        #If needed, record sqMMD value and time for corresponding incomplete size
        if j == incomplete_list[incomplete_index] - 1:
            sqMMD_list[incomplete_index] = total_value/((j+1)*1.0)
            sqMMD_variances[incomplete_index] = (total_value_sqd/((j+1)*1.0) - (total_value**2/((j+1)*1.0))/((j+1)*1.0))/((j+1)*1.0) 
            clock_gettime(CLOCK_REALTIME, &ts)
            end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
            sqMMD_times[incomplete_index] = end-start
            incomplete_index += 1
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void incomplete_sqMMD_gaussian_Rademacher_subdiagonals(const double[:,:] X1,
                                                             const double[:,:] X2,
                                                             const long long[:] incomplete_list,
                                                             const long B,
                                                             const long long max_l,
                                                             const double lam_sqd,
                                                             const long[:,:] epsilon,
                                                             double[:,:] sqMMD_matrix,
                                                             double[:] sqMMD_vector,
                                                             double[:] sqMMD_times) noexcept nogil:
    """
    Computes the incomplete squared MMD estimator for the Gaussian kernel from samples X1 and X2, 
    for different Rademacher vectors
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      B: number of Rademacher vectors
      max_l: largest number of pairs (largest/last element of incomplete_list)
      lam_sqd: squared bandwidth
      epsilon: used to store Rademacher vector
      sqMMD_list: used to store results
      sqMMD_variances: used to store results
      sqMMD_times: used to store results
    """
    
    cdef long n_samples = X1.shape[0]
    cdef double h_value
    cdef long incomplete_index = 0
    cdef long long j

    cdef long k, l, r
    cdef long D0, D1

    cdef long long eps_D0, eps_D1
    cdef timespec ts
    cdef double start, end
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    printf('Run ktc.incomplete_sqMMD_gaussian_Rademacher inside\n')
    
    D0 = 0
    D1 = 1
    r = 1
    for j in range(max_l):
        if r > n_samples - 1:
            printf('Error: r > n_samples - 1: r = %u, n_samples = %u, D0 = %u, D1 = %u \n', r, n_samples, D0, D1)
        
        #Compute contribution of pair
        h_value = h_gaussian(X1[D0],X1[D1],X2[D0],X2[D1],lam_sqd)
        
        #Update sqMMD vector
        for l in range(B):
            sqMMD_vector[l] += epsilon[D0,l]*epsilon[D1,l]*h_value/n_samples
        sqMMD_vector[B] += h_value/n_samples
        
        #If j is equal to some element of incomplete_list (minus 1), store sqMMD values and times    
        if j == incomplete_list[incomplete_index] - 1:
            printf('incomplete_index %u of %u: %u\n', incomplete_index+1, len(incomplete_list), incomplete_list[incomplete_index])
            for l in range(B+1):
                sqMMD_matrix[incomplete_index,l] = sqMMD_vector[l]
            clock_gettime(CLOCK_REALTIME, &ts)
            end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
            sqMMD_times[incomplete_index] = end-start
            incomplete_index += 1
            
        if D0 == n_samples - r - 1:
            r += 1
            D0 = 0
            D1 = r
        else:
            D0 += 1
            D1 += 1
    
    for j in range(incomplete_index,len(incomplete_list)):
        for l in range(B+1):
            sqMMD_matrix[j,l] = sqMMD_vector[l]
        sqMMD_times[j] = end-start
            
    printf('final incomplete_index: %u. len(incomplete_list): %u.\n', incomplete_index, len(incomplete_list))
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void incomplete_sqMMD_gaussian_Rademacher_aggregated(const double[:,:] X1,
                                                           const double[:,:] X2,
                                                           const long long[:] incomplete_list,
                                                           const long B,
                                                           const long long max_l,
                                                           const double[:] lam_sqd,
                                                           const long[:,:] epsilon,
                                                           double[:,:,:] sqMMD_matrix,
                                                           double[:,:] sqMMD_vector,
                                                           double[:] sqMMD_times,
                                                           double[:] h_values) noexcept nogil:
    """
    Computes the incomplete squared MMD estimator for the Gaussian kernel from samples X1 and X2, 
    for different Rademacher vectors and different bandwidths
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      incomplete_list: vector containing numbers of pairs, ordered
      max_l: largest number of pairs (largest/last element of incomplete_list)
      epsilon: used to store Rademacher vector
      lam_sqd: squared bandwidth
      epsilon: random seed
      sqMMD_list: used to store results
      sqMMD_variances: used to store results
      sqMMD_times: used to store results
    """
    
    cdef long n_samples = X1.shape[0]
    cdef double h_value
    cdef long incomplete_index = 0
    cdef long long j

    cdef long k, l, r, i
    cdef long D0, D1

    cdef long long eps_D0, eps_D1
    cdef timespec ts
    cdef double start, end
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    printf('Run ktc.incomplete_sqMMD_gaussian_Rademacher inside\n')
    
    D0 = 0
    D1 = 1
    r = 1
    for j in range(max_l):
        if r > n_samples - 1:
            printf('Error: r > n_samples - 1: r = %u, n_samples = %u, D0 = %u, D1 = %u \n', r, n_samples, D0, D1)
        
        #Compute contribution of pair
        h_gaussian_aggregated(X1[D0],X1[D1],X2[D0],X2[D1],lam_sqd,h_values)
        
        #Update sqMMD vector
        for i in range(len(lam_sqd)):
            for l in range(B):
                sqMMD_vector[i,l] += epsilon[D0,l]*epsilon[D1,l]*h_values[i]/n_samples
            sqMMD_vector[i,B] += h_values[i]/n_samples
        
        #If j is equal to some element of incomplete_list (minus 1), store sqMMD values and times    
        if j == incomplete_list[incomplete_index] - 1:
            printf('incomplete_index %u of %u: %u\n', incomplete_index+1, len(incomplete_list), incomplete_list[incomplete_index])
            for i in range(len(lam_sqd)):
                for l in range(B):
                    sqMMD_matrix[incomplete_index,i,l] = sqMMD_vector[i,l]
                sqMMD_matrix[incomplete_index,i,B] = sqMMD_vector[i,B]
            clock_gettime(CLOCK_REALTIME, &ts)
            end = ts.tv_sec + (ts.tv_nsec / 1000000000.)
            sqMMD_times[incomplete_index] = end-start
            incomplete_index += 1
            
        if D0 == n_samples - r - 1:
            r += 1
            D0 = 0
            D1 = r
        else:
            D0 += 1
            D1 += 1
    
    for j in range(incomplete_index,len(incomplete_list)):
        for i in range(len(lam_sqd)):
            for l in range(B):
                sqMMD_matrix[j,i,l] = sqMMD_vector[i,l]
            sqMMD_matrix[j,i,B] = sqMMD_vector[i,B]
        sqMMD_times[j] = end-start
            
    printf('final incomplete_index: %u. len(incomplete_list): %u.\n', incomplete_index, len(incomplete_list))
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void cross_mmd(const double[:,:] X1,
                     const double[:,:] X2,
                     const double lam_sqd,
                     const long n1a,
                     const long n1b,
                     const long n2a,
                     const long n2b,
                     double[:] U_x,
                     double[:] U_y,
                     double[:] results) noexcept nogil:
    """
    Computes the Cross MMD estimator from https://arxiv.org/pdf/2211.14908.pdf
    for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: bandwidth squared
      n1a: size of first split of X1
      n1b: size of first split of X2
      n2a: size of second split of X1
      n2b: size of second split of X2
      U_x: sum of kernel evaluations for the first split of X1 
      U_y: sum of kernel evaluations for the first split of X2
      results: used to store results; first component is the statistic, 
          second component is the unnormalized statistic, 
          third component is the variance estimate
    """
    cdef long k, l, r, i, j
    cdef double U_x_mean = 0
    cdef double U_x_2nd_moment = 0
    cdef double U_y_mean = 0
    cdef double U_y_2nd_moment = 0
    
    for i in range(n1a):
        U_x[i] = sum_gaussian_kernel_vector_matrix(X1, X1, i, n1a, lam_sqd)/n1b - sum_gaussian_kernel_vector_matrix(X1, X2, i, n2a, lam_sqd)/n2b
        U_x_mean += U_x[i]
        U_x_2nd_moment += U_x[i]**2  
    U_x_mean = U_x_mean/n1a
    U_x_2nd_moment = U_x_2nd_moment/n1a
    variance_x = U_x_2nd_moment - U_x_mean**2
        
    for i in range(n2a):
        U_y[i] = sum_gaussian_kernel_vector_matrix(X2, X1, i, n1a, lam_sqd)/n1b - sum_gaussian_kernel_vector_matrix(X2, X2, i, n2a, lam_sqd)/n2b
        U_y_mean += U_y[i]
        U_y_2nd_moment += U_y[i]**2 
    U_y_mean = U_y_mean/n2a
    U_y_2nd_moment = U_y_2nd_moment/n2a
    variance_y = U_y_2nd_moment - U_y_mean**2
    
    results[1] = U_x_mean - U_y_mean
    results[2] = variance_x/n1a + variance_y/n2a
    results[0] = results[1]/sqrt(results[2])
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void independence_test_WB(const double[:,:] X1,
                                const double[:,:] X2,
                                const long[:,:] epsilon,
                                const long n,
                                const long B,
                                const double lam_sqd_1,
                                const double lam_sqd_2,
                                long[:,:] pi, 
                                double[:,:] K1,
                                double[:,:] K2,
                                double[:] meanK1,
                                double[:] meanK2,
                                double[:] statistics) noexcept nogil:
    """
    Computes the Cross MMD estimator from https://arxiv.org/pdf/2211.14908.pdf ### Update
    for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n,d)
      X2: Matrix of size (n,d)
      epsilon: used to store a vector with iid components valued at 0/1
      B: number of Rademacher vectors
      lam_sqd_1: bandwidth squared for X1
      lam_sqd_2: bandwidth squared for X1
      n: size of X1 and X2
      K1: kernel matrix for X1
      K2: kernel matrix for X2
      pi: Matrix of size (n,B+1), stores the permutation used for each replicate
      meanK1: mean of the kernel matrix for X1
      meanK2: mean of the kernel matrix for X2
      results: used to store results
    """
    cdef long k, l, r, i, j
    
    gaussian_kernel_same(X1,lam_sqd_1,K1)
    gaussian_kernel_same(X2,lam_sqd_2,K2)
    
    cdef double mean1, mean2
    
    for i in range(n):
        mean1 = 0
        mean2 = 0
        for j in range(n):
            mean1 += K1[i,j]
            mean2 += K2[i,j]
        meanK1[i] = mean1/n
        meanK2[i] = mean2/n
        
    for i in range(n):
        for j in range(n):
            K1[i,j] = K1[i,j] - meanK1[i]
            K2[i,j] = K2[i,j] - meanK2[i]
    
    cdef long n_over_2 = int(n/2)
    cdef long i2, i3
    
    for i in range(n):
        i2 = i + n_over_2 if i < n_over_2 else i - n_over_2
        i3 = i if i < n_over_2 else i - n_over_2
        for k in range(B+1):
            pi[i,k] = epsilon[i3,k]*i + (1-epsilon[i3,k])*i2
    
    cdef double statistic
    for k in range(B+1):
        statistic = 0
        for i in range(n):
            for j in range(n):
                statistic += K1[i,j]*K2[pi[j,k],pi[i,k]]
        statistics[k] = statistic/n**2


@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void new_cheap_independence_test_WB(const double[:,:] X1,
                                      const double[:,:] X2,
                                      const long[:,:] pi,
                                      const long[:,:] M,
                                      const long n,
                                      const long B,
                                      const long s,
                                      const double lam_sqd_1,
                                      const double lam_sqd_2,
                                      double[:,:,:,:] V,
                                      double[:,:] K1,
                                      double[:,:] K2,
                                      double[:] meanK1,
                                      double[:] meanK2,
                                      double[:] statistics) noexcept nogil:
    """
    Computes the Cheap Independence estimator from https://arxiv.org/pdf/2211.14908.pdf ### Update
    for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n,d)
      X2: Matrix of size (n,d)
      pi: used to store a vector with iid components valued at 0/1
      M: Matrix of size (n,2), stores identity permutation in first row and flip permutation in second row
      B: number of Rademacher vectors
      n: size of X1 and X2
      s: number of bins
      lam_sqd_1: bandwidth squared for X1
      lam_sqd_2: bandwidth squared for X2
      V: Matrix of size (s,s,2,2), to store the sufficient statistics
      K1: kernel matrix for X1
      K2: kernel matrix for X2
      meanK1: Vector of size n; row means of X1 kernel
      meanK2: Vector of size n; row means of X2 kernel
      statistics: used to store the statistic for each replicate
    """
    cdef long p, q, r, t, i, j
    
    # Compute kernel matrices
    gaussian_kernel_same(X1,lam_sqd_1,K1)
    gaussian_kernel_same(X2,lam_sqd_2,K2)
    
    # Compute kernel matrix row means and overall means
    cdef double mean1, mean2, mu1, mu2
    mu1 = 0 
    mu2 = 0
    for i in range(n):
        mean1 = 0
        mean2 = 0
        # Compute row sum
        for j in range(n):
            mean1 += K1[i,j]
            mean2 += K2[i,j]
        # Store row mean
        meanK1[i] = mean1 / n
        meanK2[i] = mean2 / n
        mu1 += meanK1[i]
        mu2 += meanK2[i]
    mu1 /= n
    mu2 /= n
    
    # Mean center each kernel
    for i in range(n):
        mean1 = mu1 - meanK1[i]
        mean2 = mu2 - meanK2[i]
        for j in range(n):
            K1[i,j] += mean1 - meanK1[j]
            K2[i,j] += mean2 - meanK2[j]

    # Compute sufficient statistics
    cdef long im, jm, a, b, M_ap, m = n//s, s_over_2 = s//2, two_m = 2*m
    cdef double V_sum, sub_diag_statistic, diag_statistic 
    # Iterate over bins
    for i in range(s_over_2):
        im = i*two_m
        for j in range(i):
            jm = j*two_m
            # Iterate over swap options
            for p in range(2):
                for q in range(2):
                    V_sum = 0
                    for r in range(two_m):
                        # Iterate over elements of bin i and Swap(bin i), which are adjacent
                        a = im+r
                        M_ap = M[a,p]
                        for t in range(two_m):
                            # Iterate over elements of bin j and Swap(bin j), which are adjacent                        
                            b = jm+t
                            V_sum += K1[a,b]*K2[M_ap,M[b,q]]
                            
                    V[i,j,p,q] = V_sum 
        # Compute diagonal entries (j = i) with p = q
        for p in range(2):
            sub_diag_statistic = 0
            diag_statistic = 0
            # Iterate over elements of bin i and Swap(bin i), which are adjacent
            for r in range(two_m):
                a = im+r
                M_ap = M[a,p]
                # Sub-diagonal contributions (b < a)
                for t in range(r):
                    b = im+t
                    sub_diag_statistic += K1[a,b]*K2[M_ap,M[b,p]]
                # Diagonal contribution (b = a)
                diag_statistic += K1[a,a]*K2[M_ap,M_ap]
            
            V[i,i,p,p] = 2*sub_diag_statistic + diag_statistic
    
    # Compute permuted test statistics
    cdef long pi_ik
    cdef double n_sqd = n**2 
    for k in range(B+1):
        sub_diag_statistic = 0
        diag_statistic = 0
        for i in range(s_over_2):
            pi_ik = pi[i,k]
            # Sub-diagonal contribution (j < i)
            for j in range(i):
                sub_diag_statistic += V[i,j,pi_ik,pi[j,k]]
            # Diagonal contribution (j=i)
            diag_statistic += V[i,i,pi_ik,pi_ik]
        statistics[k] = (2*sub_diag_statistic + diag_statistic)/n_sqd
        
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void norecenter_cheap_independence_test_WB(const double[:,:] X1,
                                      const double[:,:] X2,
                                      const long[:,:] pi,
                                      const long[:,:] M,
                                      const long n,
                                      const long B,
                                      const long s,
                                      const double lam_sqd_1,
                                      const double lam_sqd_2,
                                      double[:,:,:,:] V,
                                      double[:,:] K1,
                                      double[:,:] K2,
                                      double[:] meanK1,
                                      double[:] meanK2,
                                      double[:] statistics) noexcept nogil:
    """
    Computes the Cheap Independence estimator from https://arxiv.org/pdf/2211.14908.pdf ### Update
    for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n,d)
      X2: Matrix of size (n,d)
      pi: used to store a vector with iid components valued at 0/1
      M: Matrix of size (n,2), stores identity permutation in first row and flip permutation in second row
      B: number of Rademacher vectors
      n: size of X1 and X2
      s: number of bins
      lam_sqd_1: bandwidth squared for X1
      lam_sqd_2: bandwidth squared for X2
      V: Matrix of size (s,s,2,2), to store the sufficient statistics
      K1: kernel matrix for X1
      K2: kernel matrix for X2
      meanK1: Vector of size n; row means of X1 kernel
      meanK2: Vector of size n; row means of X2 kernel
      statistics: used to store the statistic for each replicate
    """
    cdef long p, q, r, t, i, j
    
#     printf('independence_test_WB start \n')
    # Compute kernel matrices
    gaussian_kernel_same(X1,lam_sqd_1,K1)
    gaussian_kernel_same(X2,lam_sqd_2,K2)
    
#     printf('independence_test_WB middle 1 \n')
    
    # Compute kernel matrix row means
    cdef double mean1, mean2, C1, C2###
    C1 = 0 
    C2 = 0
    for i in range(n):
        mean1 = 0
        mean2 = 0
        # Compute row sum
        for j in range(n):
            mean1 += K1[i,j]
            mean2 += K2[i,j]
        # Store row mean
        meanK1[i] = mean1 / n
        meanK2[i] = mean2 / n
        C1 += meanK1[i]
        C2 += meanK2[i]
    C1 /= n
    C2 /= n
            
#     printf('independence_test_WB middle 2 \n')

    # Compute sufficient statistics
    cdef long im, jm, a, b, M_ap, m = n // s, s_over_2 = s//2, two_m = m*2
    cdef double V_sum, sub_diag_statistic, diag_statistic, mean_statistic
    # Iterate over bins
    for i in range(s_over_2):
        im = i*two_m
        for j in range(i):
            jm = j*two_m
            # Compute sub-diagonal entries (j < i)
            # V[i,j,0,0] = sum_{a in bin i, b in bin j} k(X1_a, X1_b) k(X2_a, X2_b)/m^2
            # V[i,j,0,1] = sum_{a in bin i, b in bin j} k(X1_a, X1_b) k(X2_a, X2_{swap(b)})/m^2
            # V[i,j,1,0] = sum_{a in bin i, b in bin j} k(X1_a, X1_b) k(X2_{swap(a)}, X2_b)/m^2
            # V[i,j,1,1] = sum_{a in bin i, b in bin j} k(X1_a, X1_b) k(X2_{swap(a)}, X2_{swap(b)})/m^2
            # Iterate over swap options
            for p in range(2):
                for q in range(2):
                    V_sum = 0
                    # Iterate over elements of bin i and Swap(bin i), which are adjacent
                    for r in range(two_m):
                        a = im+r
                        M_ap = M[a,p]
                        # Iterate over elements of bin j and Swap(bin j), which are adjacent
                        for t in range(two_m):
                            b = jm+t
                            # V_sum += K1[a,b] * K2[M[a,p],M[b,q]]
                            V_sum += K1[a,b]*K2[M_ap,M[b,q]]
                    V[i,j,p,q] = V_sum 
        # Compute diagonal entries (j = i) with p = q
        # V[i,i,0,0] = [-2n meanK1[i] * meanK2[i] + sum_{a, b in bin i} k(X1_a, X1_b) k(X2_a, X2_b)]/m^2 
        # V[i,i,1,1] = [-2n meanK1[i] * meanK2[swap(i)] + sum_{a, b in bin i} k(X1_a, X1_b) k(X2_swap(a), X2_swap(b))]/m^2
        for p in range(2):
            sub_diag_statistic = 0
            diag_statistic = 0
            mean_statistic = 0
            # Enumerate over elements of bin i
            for r in range(two_m):
                a = im+r
                M_ap = M[a,p]
                # Sub-diagonal contributions (b < a)
                for t in range(r):
                    b = im+t
                    # sub_diag_statistic += K1[a,b] * K2[M[a,p],M[b,p]]
                    sub_diag_statistic += K1[a,b] * K2[M_ap,M[b,p]]
                # Diagonal contribution (b = a)
                # diag_statistic += K1[a,a] * K2[M[a,p],M[a,p]]
                diag_statistic += K1[a,a] * K2[M_ap,M_ap]
                # Mean product contribution
                mean_statistic += meanK1[a] * meanK2[M_ap]
            V[i,i,p,p] = (2*(sub_diag_statistic - n*mean_statistic) + diag_statistic) 
        
    
#     for i in range(n_over_2):
#         printf('i: %u, epsilon[i,0]: %u, ', i, epsilon[i,0])
    
    # Compute permuted test statistics
    cdef long pi_ik
    cdef double n_sqd = n**2 
    for k in range(B+1):
        sub_diag_statistic = 0
        diag_statistic = 0
#         printf('statistic set to zero \n')
        for i in range(s_over_2):
#             i2 = i + n_over_2 if i < n_over_2 else i - n_over_2
#             i3 = i if i < n_over_2 else i - n_over_2
#             pi_i = epsilon[i3,k]*i + (1-epsilon[i3,k])*i2
#             printf('k: %u \n', k)
#             printf('epsilon[i%n_over_2,k]*i + (1-epsilon[i%n_over_2,k])*i2: %u \n', epsilon[i%n_over_2,k]*i + (1-epsilon[i%n_over_2,k])*i2)
#             printf('i: %u, i2: %u, pi_i: %u, i3: %u, epsilon[i3,k]: %u.\n', i, i2, pi_i, i3, epsilon[i3,k])
            pi_ik = pi[i,k]
            # Sub-diagonal contribution (j < i)
            for j in range(i):
#                 j2 = j + n_over_2 if j < n_over_2 else j - n_over_2
#                 j3 = j if j < n_over_2 else j - n_over_2
#                 pi_j = epsilon[j3,k]*j + (1-epsilon[j3,k])*j2
#                 printf('k: %u \n', k)
#                 printf('j: %u, j2: %u, pi_j: %u, j3: %u, k: %u, epsilon[j3,k]: %u.\n', j, j2, pi_j, j3, k, epsilon[j3,k])
#                 statistic += K1[i,j]*K2[pi[j,k],pi[i,k]]
                sub_diag_statistic += V[i,j,pi_ik,pi[j,k]]
            # Diagonal contribution (j=i)
            diag_statistic += V[i,i,pi_ik,pi_ik]
        statistics[k] = (2*sub_diag_statistic + diag_statistic)/n_sqd  + C1*C2###
#         printf('loop finished \n')
        
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void cheap_independence_test_WB(const double[:,:] X1,
                                      const double[:,:] X2,
                                      const long[:,:] epsilon,
                                      const long[:,:] M,
                                      const long n,
                                      const long B,
                                      const long s,
                                      const double lam_sqd_1,
                                      const double lam_sqd_2,
                                      long[:,:] pi,
                                      double[:,:,:,:] V,
                                      double[:,:] K1,
                                      double[:,:] K2,
                                      double[:] meanK1,
                                      double[:] meanK2,
                                      double[:] statistics) noexcept nogil:
    """
    Computes the Cheap Independence estimator from https://arxiv.org/pdf/2211.14908.pdf ### Update
    for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n,d)
      X2: Matrix of size (n,d)
      epsilon: used to store a vector with iid components valued at 0/1
      M: Matrix of size (n,2), stores identity permutation in first row and flip permutation in second row
      B: number of Rademacher vectors
      n: size of X1 and X2
      s: number of bins
      lam_sqd_1: bandwidth squared for X1
      lam_sqd_2: bandwidth squared for X1
      pi: Matrix of size (n,B+1), to store the permutation used for each replicate
      V: Matrix of size (m,m,2,2), to store the sufficient statistics
      K1: kernel matrix for X1
      K2: kernel matrix for X2
      meanK1: mean of the kernel matrix for X1
      meanK2: mean of the kernel matrix for X2
      statistics: used to store the statistic for each replicate
    """
    cdef long p, q, r, t, i, j
    
    gaussian_kernel_same(X1,lam_sqd_1,K1)
    gaussian_kernel_same(X2,lam_sqd_2,K2)
    
    cdef double mean1, mean2
    
    for i in range(n):
        mean1 = 0
        mean2 = 0
        for j in range(n):
            mean1 += K1[i,j]
            mean2 += K2[i,j]
        meanK1[i] = mean1/n
        meanK2[i] = mean2/n
        
    for i in range(n):
        for j in range(n):
            K1[i,j] = K1[i,j] - meanK1[i]
            K2[i,j] = K2[i,j] - meanK2[i]

    cdef long m = int(n/s)

    for p in range(2):
        for q in range(2):
            for i in range(s):
                for j in range(s):
                    V[i,j,p,q] = 0
                    for r in range(m):
                        for t in range(m):
                            V[i,j,p,q] += K1[i*m+r,j*m+t]*K2[M[j*m+t,q],M[i*m+r,p]]
                    V[i,j,p,q] = V[i,j,p,q]/m**2
    
    cdef long s_over_2 = int(s/2)
    cdef long i2, i3
    
    for i in range(s):
        i3 = i if i < s_over_2 else i - s_over_2
        for k in range(B+1):
            pi[i,k] = epsilon[i3,k]
    
    cdef double statistic
    for k in range(B+1):
        statistic = 0
        for i in range(s):
            for j in range(s):
                statistic += V[i,j,pi[i,k],pi[j,k]]
        statistics[k] = statistic/s**2

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void cross_independence_test(const double[:,:] X1,
                                   const double[:,:] X2,
                                   const long n,
                                   const double lam_sqd_1,
                                   const double lam_sqd_2,
                                   double[:,:] K,
                                   double[:,:] L,
                                   double[:] u_K,
                                   double[:] l_K,
                                   double[:] u_L,
                                   double[:] l_L,
                                   double[:,:] K_circ_L,
                                   double[:] KL_u,
                                   double[:] LK_u,
                                   double[:] K_circ_L_l,
                                   double[:] w,
                                   double[:] results) noexcept nogil:
    """
    Computes the Cheap Independence estimator from https://arxiv.org/pdf/2211.14908.pdf ### Update
    for the Gaussian kernel from samples X1 and X2
    
    Args:
      X1: Matrix of size (n,d)
      X2: Matrix of size (n,d)
      lam_sqd_1: bandwidth squared for first component
      lam_sqd_2: bandwidth squared for second component
      K: kernel matrix for X1
      L: kernel matrix for X2
      ...
      results: used to store the results (un-normalized HSIC and variance)
    """
    cdef long p, q, r, t, i, j, k
    
    cdef long n_over_2 = int(n/2)
    cdef long n_over_4 = int(n_over_2/2)
    
    cdef timespec ts
    cdef double start, end
    
    cdef double time
    
    cdef double mean1, mean2
    
    #Record start time
    clock_gettime(CLOCK_REALTIME, &ts)
    start = ts.tv_sec + (ts.tv_nsec / 1000000000.)
    
    printf('Point 0 \n')
    
    gaussian_kernel(X1[:n_over_2,:],X1[n_over_2:,:],lam_sqd_1,K)
    gaussian_kernel(X2[:n_over_2,:],X2[n_over_2:,:],lam_sqd_2,L)
    
    cdef double tr_KL = 0
    cdef double u_KL_u = 0
    cdef double l_KL_l = 0
    cdef double l_K_u = 0
    cdef double l_L_u = 0
    cdef double T1, T2, T3, T4
    cdef double s_n_sqd
    
    printf('Point 1 \n')
    
    for i in range(n_over_2):
        for j in range(n_over_2):
            tr_KL += 2*K[i,j]*L[i,j]
            u_K[i] += K[j,i]
            l_K[i] += K[i,j]
            u_L[i] += L[j,i]
            l_L[i] += L[i,j]
            K_circ_L[i,j] += K[i,j]*L[i,j]
    
    printf('Point 2 \n')
    
    for i in range(n_over_2):
        u_KL_u += u_K[i]*u_L[i]
        l_KL_l += l_K[i]*l_L[i]
        l_K_u += u_K[i]
        l_L_u += u_L[i]
        
    printf('Point 3 \n')
        
    T1 = 0.5*tr_KL/n_over_2**2
    T2 = (l_KL_l-0.5*tr_KL)/(n_over_2**2*(n_over_2-1))
    T3 = (u_KL_u-0.5*tr_KL)/(n_over_2**2*(n_over_2-1))
    T4 = (l_K_u*l_L_u-l_KL_l-u_KL_u+0.5*tr_KL)/(n_over_2*(n_over_2-1))**2
        
    results[0] = T1 - T2 - T3 + T4
    
    for i in range(n_over_2):
        for j in range(n_over_2):
            KL_u[i] += K[i,j]*u_L[j]
            LK_u[i] += L[i,j]*u_K[j]
            K_circ_L_l[i] += K_circ_L[i,j]
    
    cdef double term_1, term_2, term_3, term_4, term_5, term_6
    
    cdef double norm_term_1 = 0
    cdef double norm_term_2 = 0
    cdef double norm_term_3 = 0
    cdef double norm_term_4 = 0
    cdef double norm_term_5 = 0
    cdef double norm_term_6 = 0
    cdef double norm_w_sqd = 0
    for i in range(n_over_2):
        term_1 = n_over_2*K_circ_L_l[i]/(2*(n_over_2-1))
        term_2 = 0.5*tr_KL/(2*(n_over_2-1))
        term_3 = (KL_u[i] + LK_u[i])/(2*(n_over_2-1))
        term_4 = l_K[i]*l_L[i]/(2*(n_over_2-1))
        term_5 = l_KL_l/(2*n_over_2*(n_over_2-1))
        term_6 = 2*(l_K_u*l_L[i]+l_L_u*l_K[i])/(4*n_over_2*(n_over_2-1))
        w[i] = term_1 + term_2 - term_3 - term_4 - term_5 + term_6
        norm_term_1 += term_1**2
        norm_term_2 += term_2**2
        norm_term_3 += term_3**2
        norm_term_4 += term_4**2
        norm_term_5 += term_5**2
        norm_term_6 += term_6**2
    
    for i in range(n_over_2):
        norm_w_sqd += w[i]**2
        
    cdef double variance_term_1 = 4*(n_over_2-1)*norm_w_sqd/(n_over_2-2)**2/(n_over_2-1)**2
    cdef double variance_term_2 = 4*(n_over_2-1)*n_over_2*results[0]**2/(n_over_2-2)**2
    
    results[1] = variance_term_1 - variance_term_2
    printf('n_over_2: %u \n', n_over_2)
    printf('norm_w_sqd: %f \n', norm_w_sqd)
    printf('norm_term_1: %f \n', norm_term_1)
    printf('norm_term_2: %f \n', norm_term_2)
    printf('norm_term_3: %f \n', norm_term_3)
    printf('norm_term_4: %f \n', norm_term_4)
    printf('norm_term_5: %f \n', norm_term_5)
    printf('norm_term_6: %f \n', norm_term_6)
    printf('results[0]: %f \n', results[0])
    printf('results[0]**2: %f \n', results[0]**2)
    printf('variance_term_1: %f \n', variance_term_1)
    printf('variance_term_2: %f \n', variance_term_2)
    printf('results[1]: %f \n', results[1])
    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double inner_prod(const double[:,:] K,
                        const double[:,:] L,
                        const long i,
                        const long j,
                        const long n_over_2):
    cdef long k
    
    cdef double term0 = 0
    cdef double K_i_sum = 0
    cdef double L_j_sum = 0
    
    for k in range(n_over_2):
        term0 += K[i,k]*L[j,k]
        K_i_sum += K[i,k]
        L_j_sum += L[j,k]
        
    cdef double term1 = term0/n_over_2
    cdef double term2 = (K_i_sum*L_j_sum - term0)/(n_over_2*(n_over_2-1))
    return term1 - term2

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double hfunc(const double[:,:] K,
                   const double[:,:] L,
                   const long i,
                   const long j,
                   const long n_over_2):
    cdef long k
    
    if i==j:
        return 0 
        # the term \langle f, \phi(X_i) \psi(Y_i) \rangle
    term_ii = inner_prod(K, L, i, i, n_over_2)
    term_jj = inner_prod(K, L, j, j, n_over_2)
    term_ij = inner_prod(K, L, i, j, n_over_2)
    term_ji = inner_prod(K, L, j, i, n_over_2)
    
    h = 0.5*(term_ii + term_jj - term_ij -term_ji)
    return h


    
@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef void compute_H(const double[:,:] X1,
                     const double[:,:] X2,
                     const long n,
                     const double lam_sqd_1,
                     const double lam_sqd_2,
                     double[:,:] K,
                     double[:,:] L,
                     double[:,:] H):
    cdef long i, j
    
    cdef long n_over_2 = int(n/2)
    
    gaussian_kernel(X1[:n_over_2,:],X1[n_over_2:,:],lam_sqd_1,K)
    gaussian_kernel(X2[:n_over_2,:],X2[n_over_2:,:],lam_sqd_2,L)
    
    for i in range(n_over_2):
        for j in range(n_over_2):
            H[i,j] = hfunc(K,L,i,j,n_over_2)
    

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_1(const double[:,:] X1,
                             const double[:,:] X2,
                             const double lam_sqd) noexcept nogil:
    """
    Computes E_z[(E_{z'}h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    cdef double intermediate_sum
    for i in range(n_samples):
        intermediate_sum = 0
        for j in range(n_samples):
            intermediate_sum += h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
        total_sum += (intermediate_sum/n_samples)**2
    return(total_sum/n_samples)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_2(const double[:,:] X1,
                             const double[:,:] X2,
                             const double lam_sqd) noexcept nogil:
    """
    Computes E_{z,z'}[(h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    for i in range(n_samples):
        for j in range(n_samples):
            total_sum += (h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd))**2
    return(total_sum/n_samples**2)

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double expectation_h_3(const double[:,:] X1,
                             const double[:,:] X2,
                             const double lam_sqd) noexcept nogil:
    """
    Compute (E_{z,z'}[h(z,z')])^2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double total_sum = 0
    for i in range(n_samples):
        for j in range(n_samples):
            total_sum += h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
    return (total_sum/n_samples**2)**2

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sigma_1_sqd(const double[:,:] X1,
                         const double[:,:] X2,
                         const double lam_sqd) noexcept nogil:
    """
    Compute sigma_1^2 = E_z[(E_{z'}h(z,z'))^2] - (E_{z,z'}[h(z,z')])^2
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double first_term = 0
    cdef double second_term = 0
    cdef double intermediate_sum
    cdef double h_value
    for i in range(n_samples):
        intermediate_sum = 0
        for j in range(n_samples):
            h_value = h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
            intermediate_sum += h_value
            second_term += h_value
        first_term += (intermediate_sum/n_samples)**2
    return (first_term/n_samples) - (second_term/n_samples**2)**2

@cython.boundscheck(False) # turn off bounds-checking for this function
@cython.wraparound(False)  # turn off negative index wrapping for this function
@cython.initializedcheck(False) # turn off memoryview initialization checks for this function
@cython.cdivision(True) # Disable C-division checks for this function
cpdef double sigma_2_sqd(const double[:,:] X1,
                         const double[:,:] X2,
                         const double lam_sqd) noexcept nogil:
    """
    Compute sigma_2^2 = E_{z,z'}[h(z,z')^2] - E_z[(E_{z'}h(z,z'))^2]
    
    Args:
      X1: Matrix of size (n1,d)
      X2: Matrix of size (n2,d)
      lam_sqd: double, squared bandwidth of the kernel
    """
    cdef long n_samples = X1.shape[0]
    cdef long i, j
    cdef double first_term = 0
    cdef double second_term = 0
    cdef double intermediate_sum
    cdef double h_value
    for i in range(n_samples):
        intermediate_sum = 0
        for j in range(n_samples):
            h_value = h_gaussian(X1[i],X1[j],X2[i],X2[j],lam_sqd)
            intermediate_sum += h_value
            first_term += h_value**2
        second_term += (intermediate_sum/n_samples)**2
    return first_term/n_samples**2 - (second_term/n_samples)
