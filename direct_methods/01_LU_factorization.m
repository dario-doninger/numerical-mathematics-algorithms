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

function [P, L, U, x] = lu_factorization(A, b)

n = size(A, 1);
    
    % Initialize matrices
    U = A;
    L = eye(n);
    P = eye(n);
    b = b(:); % Ensure b is a column vector
    
    % 1. FACTORIZATION STAGE (PA = LU)
    
    for k = 1:n-1
        % Identify the pivot: row with the max absolute value in the current column
        [~, max_idx] = max(abs(U(k:n, k)));
        pivot_row = max_idx + k - 1;
        
        % Row swapping if necessary
        if pivot_row ~= k
            % Swap rows in U
            U([k, pivot_row], k:n) = U([pivot_row, k], k:n);
            % Swap rows in P
            P([k, pivot_row], :) = P([pivot_row, k], :);
            % Swap rows in L (only previously computed multipliers)
            if k > 1
                L([k, pivot_row], 1:k-1) = L([pivot_row, k], 1:k-1);
            end
        end
        
        % Gaussian Elimination step
        for i = k+1:n
            multiplier = U(i, k) / U(k, k);
            L(i, k) = multiplier;
            U(i, k:n) = U(i, k:n) - multiplier * U(k, k:n);
        end
    end
    
    % 2. FORWARD SUBSTITUTION (L * y = P * b)

    Pb = P * b;
    y = zeros(n, 1);
    
    for i = 1:n
        y(i) = Pb(i) - L(i, 1:i-1) * y(1:i-1);
    end
    
    % 3. BACKWARD SUBSTITUTION (U * x = y)

    x = zeros(n, 1);
    
    for i = n:-1:1
        x(i) = (y(i) - U(i, i+1:n) * x(i+1:n)) / U(i, i);
    end
    
end

% USAGE EXAMPLE:
% A_test = [2, 1, 1; 4, 3, 3; 8, 7, 9];
% b_test = [4; 10; 24];
% [P, L, U, x] = lu_factorization(A_test, b_test);
% disp('Solution x:');
% disp(x);
