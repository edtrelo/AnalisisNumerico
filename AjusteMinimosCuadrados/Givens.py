import numpy as np
from Sustitucion import sustAtras

def _obtenerCyS(a, i, j):
    """Obtiene el coseno y el seno de la rotación de Givens."""
    norma = np.sqrt(a[i]**2 + a[j]**2)
    c = a[i]/norma
    s = a[j]/norma
    return c, s

def obtenerRotGivens(a, i, j):
    m = len(a)
    # G = I_m
    G = np.eye(m)
    c, s = _obtenerCyS(a, i, j)
    G[i, i] = c
    G[j, j] = c
    G[j, i] = -s
    G[i, j] = s
    return G

def byGivens(Matriz, vector):
    A = np.array(Matriz, dtype = np.float64)
    b = np.array(vector, dtype = np.float64)
    m, n = A.shape
    # recorremos columnas
    for j in range(n):
        #recorremos filas
        for i in range(j+1, m):
            a = A[:, j]
            # hacemos cero debajo
            if A[i, j] != 0:
                G = obtenerRotGivens(a, j, i)
                A = np.dot(G, A)
                b = np.dot(G, b)
    
    print(A)
    R, b = A[:n], b[:n]

    X = sustAtras(R, b)
    return X

A = [[1,1],
    [-1,0],
    [0,1],
    [1,0]]

A = np.array(A, dtype = np.float64)

b = np.array([-1, 2, -1, 1])
print(-4*np.sqrt(3)/15)

X = byGivens(A, b)
print(X)
