"""
Thomas Algorithm (Tridiagonal Matrix Algorithm)

DESCRIPTION:
An exceptionally efficient, simplified variant of Gaussian elimination designed 
exclusively for tridiagonal systems of linear equations. It operates directly 
on 1D vectors representing the diagonals, and it doesn't need to store or 
process zero elements.

MATHEMATICAL HYPOTHESIS:
- The coefficient matrix must be tridiagonal.
- For guaranteed numerical stability without pivoting, the matrix should ideally 
  be strictly diagonally dominant (|b_i| > |a_i| + |c_i|) or symmetric positive 
  definite. If these conditions are violated, the algorithm may fail due to 
  division by zero.

COMPUTATIONAL COMPLEXITY:
- Time Complexity: O(N) FLOPs (specifically, approximately 8N operations).
- Auxiliary Space Complexity: O(N) to store the modified diagonal and RHS vectors,
  achieving massive memory savings compared to dense solvers.

BEST USE CASE:
Highly optimized for solving 1D boundary value problems discretized via finite 
differences, cubic spline interpolation, and implicit time-stepping PDE schemes.
"""

import numpy as np

def thomas_solve(a, b, c, d):
    """
    Parameters:
    a : array_like, size (N-1) - Lower diagonal (subdiagonal)
    b : array_like, size (N)   - Main diagonal
    c : array_like, size (N-1) - Upper diagonal (superdiagonal)
    d : array_like, size (N)   - Right-Hand Side vector
    
    Returns:
    x : ndarray, size (N)      - Solution vector
    """
    # Convert inputs to float arrays to avoid integer division issues
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    d = np.array(d, dtype=float)
    
    n = len(b)
    
    # Preallocate modified vectors to preserve original inputs
    c_prime = np.zeros(n - 1, dtype=float)
    d_prime = np.zeros(n, dtype=float)
    
    # 1. FACTORIZATION

    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]
    
    for i in range(1, n - 1):
        denominator = b[i] - a[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i-1] * d_prime[i-1]) / denominator
        
    # Handle the last row separately since c_prime doesn't exist for the last element
    denominator_last = b[n-1] - a[n-2] * c_prime[n-2]
    d_prime[n-1] = (d[n-1] - a[n-2] * d_prime[n-2]) / denominator_last
    
    # 2. BACKWARD SUBSTITUTION
  
    x = np.zeros(n, dtype=float)
    x[-1] = d_prime[-1]
    
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
    return x

# USAGE EXAMPLE:
# a_sub = np.array([-1, -1, -1])          # Subdiagonal
# b_main = np.array([2, 2, 2, 2])         # Main diagonal
# c_sup = np.array([-1, -1, -1])          # Superdiagonal
# d_rhs = np.array([1, 0, 0, 1])          # Right-Hand Side
#
# x_solution = thomas_solve(a_sub, b_main, c_sup, d_rhs)
# print("Solution x:\n", x_solution)
