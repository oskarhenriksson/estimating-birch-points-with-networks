import numpy as np
from numpy.linalg import matrix_rank

### No scaling implemented yet  !!!! 

# import sys 
# import os
# file_dir_folder = os.path.dirname(os.path.realpath(__file__))
# sys.path.append(file_dir_folder)
# from independence_models import *


def IPS(A_exponent_matrix, u, err_tol=1e-4, scaling_c=None):
    '''
    Computes the MLE p of log-linear model M_{A} given: 
        - design matrix A (size d * m)
        - vector of counts u (np.array, size (m,))
        ! No scaling (scaling_c) implemented yet
    Convergence criteria:  if |Ap - Au| < err_tol
    
    If no err_tol given, default 1e-4.
    
    OUTPUT:  p (np.array,  size (m,))
    '''
    
    dim_d = A_exponent_matrix.shape[0];
    dim_m = A_exponent_matrix.shape[1];
    
    u = np.array(u);
    u = (u.flatten()).reshape(-1)
    if len(u) != dim_m:
        print('\n Error: mismatch dims of A and u ')
        
    if scaling_c is None: ### scaling NOT yet implemented
        scaling_c = np.ones(dim_m);
    
    u_sum = u.sum();
    A_colsum = A_exponent_matrix.sum(axis=0);
    if not np.all(A_colsum  == A_colsum [0]):
        print('\n Error: column sums of A not constant.')
    else:
        A_colsum = A_colsum[0];
    
    p_old =  scaling_c/sum(scaling_c);
    err = dim_m; 
    Au_norm = (A_exponent_matrix @ u)/u_sum; 
    while err > err_tol:
        Ap = A_exponent_matrix @ p_old

        p_new = np.ones(dim_m)/dim_m
        for jj in range(dim_m): 
            vect_temp = ((Au_norm/Ap)**(A_exponent_matrix[:,jj]/A_colsum)).prod() 
            p_new[jj] = vect_temp*p_old[jj]
        p_old = p_new.copy()
        err = np.linalg.norm( (A_exponent_matrix @ p_new) - Au_norm )
        
    MLE = p_new 
    
    return MLE
    




    
#     '''
#     Example usage with some A:
#         A =  np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]])
#         u =  np.random.randint(1,15, size=(1, A.shape[1] ));
#         MLE, err = IPS(A, u, err_tol=1e-6);
#         print(MLE)
    
    
#     Example usage with independence model:
#     Requires: 
        # import sys 
        # import os
        # file_dir_folder = os.path.dirname(os.path.realpath(__file__))
        # sys.path.append(file_dir_folder)
        # from independence_models import *
            # k = 4
            # d = 6
            # A = independence_model(k,d);
            # u =  np.random.randint(1,15, size=(1, A.shape[1] ));
            # u = np.array([[12,  8, 11,  9,  1, 14,  2,  2,  7,  5,  8,  1,  8,  6,  6, 10,
            #          3, 12,  4,  1, 14,  5,  6, 10]])
            # MLE_theory = independence_model_birch_point(k,d, u);

            # tic = time.time()
            # MLE_IPS, err = IPS(A, u, err_tol=u.shape[1]*1e-7);
            # toc = time.time() 

            # print(np.linalg.norm(MLE_IPS - MLE_theory)/u.shape[1])
            # print(f'runtime = {toc-tic} s')
#     '''
    
    # A =  np.array([[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1], [1, 0, 0, 1, 0, 0], [0, 1, 0, 0, 1, 0], [0, 0, 1, 0, 0, 1]])
    # u =  np.random.randint(1,15, size=(1, A.shape[1] ));
    # MLE, err = IPS(A, u, err_tol=1e-6);
    # print(MLE)





# k = 10
# d = 15
# A = independence_model(k,d);
# u =  np.random.randint(1,15, size=(1, A.shape[1] ));
# # u = np.array([[12,  8, 11,  9,  1, 14,  2,  2,  7,  5,  8,  1,  8,  6,  6, 10,
# #          3, 12,  4,  1, 14,  5,  6, 10]])
# MLE_theory = independence_model_birch_point(k,d, u);

# tic = time.time()
# MLE_IPS, err = IPS(A, u, err_tol=u.shape[1]*1e-4);
# toc = time.time() 

# print(np.linalg.norm(MLE_IPS - MLE_theory)/u.shape[1])
# print(f'IPS runtime = {toc-tic} s')
    
