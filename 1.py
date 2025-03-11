import numpy as np

def doolittle_factorization(A):
    """
    Performs Doolittle factorization (LU decomposition) of matrix A.
    Returns L (lower triangular) and U (upper triangular) matrices.
    """
    n = len(A)
    L = np.eye(n)  # Initialize L as identity matrix
    U = np.zeros((n, n))  # Initialize U as zero matrix

    for i in range(n):
        # Compute U's elements
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        # Compute L's elements
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U

def forward_substitution(L, b):
    """
    Solves Ly = b using forward substitution.
    """
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y

def backward_substitution(U, y):
    """
    Solves Ux = y using backward substitution.
    """
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def solve_system(A, b):
    """
    Solves the system Ax = b using Doolittle factorization.
    """
    # Step 1: Perform LU decomposition
    L, U = doolittle_factorization(A)

    # Step 2: Solve Ly = b using forward substitution
    y = forward_substitution(L, b)

    # Step 3: Solve Ux = y using backward substitution
    x = backward_substitution(U, y)

    return x

# Example usage
if __name__ == "__main__":
    # Input matrix A and vector b
    A = np.array([[1, 1, 1],
                  [1, 2, 2],
                  [1, 2, 3]], dtype=float)
    b = np.array([5, 6, 8], dtype=float)

    # Solve the system Ax = b
    x = solve_system(A, b)

    # Print the solution
    print("Solution x:")
    print(x)
