"""
LU Factorization with Partial Pivoting (PA = LU)

DESCRIPTION:
Solves a linear system Ax = b by computing the PA = LU decomposition (LU factorization and partial pivoting)
of the coefficient matrix A, followed by forward and backward substitution.

HYPOTHESIS:
- The coefficient matrix A must be a square and non-singular matrix (det(A) != 0).
- Partial pivoting is employed to mitigate the risk of division by zero and to restrict 
  the accumulation of floating-point round-off errors, ensuring numerical stability.

COMPUTATIONAL COMPLEXITY:
- Factorization (PA = LU): O(2/3 * N^3) FLOPs.
- Forward/Backward Substitution: O(N^2) FLOPs.
- Total Time Complexity: O(N^3).
- Auxiliary Space Complexity: O(N^2) to store the decomposed matrices.

BEST USE CASE:
Serves as a direct solver for moderate-sized, dense, non-symmetric linear systems. Highly efficient when resolving multiple 
systems that share the same coefficient matrix A but possess varying right-hand side vectors b.
"""

import numpy as np

def lu_solve_pivoting(A, b):
    # Ensure inputs are float arrays to prevent integer truncation errors
    U = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(U)
    
    L = np.eye(n, dtype=float)
    P = np.eye(n, dtype=float)
 
    # 1. FACTORIZATION STAGE (PA = LU)

    for k in range(n - 1):
        # Identify the pivot: row with the maximum absolute value in the current column
        pivot_row = np.argmax(np.abs(U[k:n, k])) + k
        
        # Row swapping if the maximum is not on the diagonal
        if pivot_row != k:
            # Swap rows in U (from column k onwards to save computations)
            U[[k, pivot_row], k:n] = U[[pivot_row, k], k:n]
            # Swap rows in the Permutation matrix P
            P[[k, pivot_row], :] = P[[pivot_row, k], :]
            # Swap rows in L (only for previously computed multipliers)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]
                
        # Gaussian Elimination step
        for i in range(k + 1, n):
            multiplier = U[i, k] / U[k, k]
            L[i, k] = multiplier
            U[i, k:n] -= multiplier * U[k, k:n]
            
    # 2. FORWARD SUBSTITUTION (L * y = P * b)

    Pb = np.dot(P, b)
    y = np.zeros(n, dtype=float)
    
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
        
    # 3. BACKWARD SUBSTITUTION (U * x = y)
  
    x = np.zeros(n, dtype=float)
    
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
    return P, L, U, x

# USAGE EXAMPLE:
# A_test = np.array([[2, 1, 1], [4, 3, 3], [8, 7, 9]])
# b_test = np.array([4, 10, 24])
# P, L, U, x = lu_solve_pivoting(A_test, b_test)
# print("Solution x:\n", x)
