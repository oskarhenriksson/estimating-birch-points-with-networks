import numpy as np
from scipy import linalg
from itertools import combinations

def independence_model(k:int, m:int) -> np.ndarray:
    """
    Computes the design matrix (denoted A in the paper) for the two-way independence model with state space [k]×[m]. 
    The matrix will have k+m rows and k*m columns.
    
    Conventions:
    - The k*m columns correspond to the states (1,1),...,(1,m), ...., (k,1),...,(k,m)
    - The k first rows correspond to marginal probabilities for each of the states [k]
    - The m last rows correspond to marginal probabilities for each of the states [m]    
    
    """
    first_part = linalg.block_diag( *[np.ones((1, m), dtype=int) for _ in range(k)] )
    second_part = np.hstack([np.eye(m, dtype=int) for _ in range(k)])
    return np.vstack([first_part, second_part])

def independence_model_kernel(k:int, m:int) -> np.ndarray:
    """
    Computes an integer matrix whose columns constitute a basis for the rational kernel 
    of the design matrix for the two-way independence model with states [k]×[m].
    
    The output matrix will have k*m rows and (k-1)*(m-1) columns.
    
    Convention: The ordering of the rows correspond to the ordering of the columns in `independence_model`
    
    """
    block = np.vstack([np.ones(m-1, dtype=int), -np.eye(m-1, dtype=int)])
    first_part = np.hstack([block for _ in range(k - 1)])
    second_part = linalg.block_diag( *[-block for _ in range(k - 1)] )
    return np.vstack([first_part, second_part])

def independence_model_markov_basis(k:int, m:int) -> np.ndarray:
    """
    Computes an integer matrix whose columns encode a Markov basis for the two-way independence model with states [k]×[m].
    
    The output matrix B will have k*m rows and binomial(k,2)*bimomial(m,2) columns.
    
    Each column b of B corresponds to a binomial of the form p^b-1, where p is the vector of probabilities.  
    
    Convention: The ordering of the rows of B correspond to the ordering of the columns in `independence_model`
    """
    
    # Function that flattens the indices (i,j) in [k]×[n] of the probability into an index in [k*m] 
    # This corresponds to the ordering of the columns of A.
    def flatten_index(i,j):
        return (i - 1) * m + j

    # Initialize an empty matrix B
    B = np.empty((k * m, 0), dtype=int)

    # Loop through the 2-by-2 minors of the k-by-m matrix of probabilities
    for columns in combinations(range(1, m + 1), 2):
        for rows in combinations(range(1, k + 1), 2):
            new_column = np.zeros(k * m, dtype=int)
            new_column[flatten_index(rows[0], columns[0]) - 1] = 1
            new_column[flatten_index(rows[1], columns[1]) - 1] = 1
            new_column[flatten_index(rows[0], columns[1]) - 1] = -1
            new_column[flatten_index(rows[1], columns[0]) - 1] = -1
            B = np.column_stack((B, new_column))

    return B

def independence_model_birch_point(k, m, data):
    """
    Computes the true Birch point given a k*m data vector.

    Convention: The ordering of the entries are assumed to correspond to the
    ordering of the columns of independence_model(k, m).
    """
    U = np.reshape(data, (k, m))
    column_sums = np.sum(U, axis=0)
    row_sums = np.sum(U, axis=1)
    mle_matrix = np.outer(row_sums, column_sums) / np.sum(data)**2
    return mle_matrix.flatten()