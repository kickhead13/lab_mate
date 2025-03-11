import numpy as np

def cholesky_factorization(A):
    n = len(A)
    L = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1):
            if i == j:
                L[i, j] = np.sqrt(A[i, i] - np.sum(L[i, :j]**2))
            else:
                L[i, j] = (A[i, j] - np.sum(L[i, :j] * L[j, :j])) / L[j, j]
    return L

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

def backward_substitution(LT, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]

    return x

def solve_system_cholesky(A, b):
    L = cholesky_factorization(A)

    y = forward_substitution(L, b)

    x = backward_substitution(L.T, y)

    return x

if __name__ == "__main__":
    A = np.array([[4, -2, 2],
                  [-2, 2, -4],
                  [2, -4, 14]], dtype=float)
    b = np.array([0, -2, 10], dtype=float)

    x = solve_system_cholesky(A, b)

    print("Solution x:")
    print(x)
