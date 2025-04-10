import numpy as np

def doolittle_factorization(A):
    n = len(A)
    L = np.eye(n)  
    U = np.zeros((n, n))  

    for i in range(n):
        for j in range(i, n):
            U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

    return L, U

def forward_substitution(L, b):
    n = len(b)
    y = np.zeros(n)

    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    return y

def backward_substitution(U, y):
    n = len(y)
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

def solve_system(A, b):
    L, U = doolittle_factorization(A)
    y = forward_substitution(L, b)
    x = backward_substitution(U, y)
    return x

if __name__ == "__main__":
    A = np.array([[1, 1, 1],
                  [1, 2, 2],
                  [1, 2, 3]], dtype=float)
    b = np.array([5, 6, 8], dtype=float)

    x = solve_system(A, b)

    print("Solution x:")
    print(x)
