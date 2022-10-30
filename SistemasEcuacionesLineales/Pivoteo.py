import numpy as np

def pivoteoParcial(A, j, i = None):
    """Obtiene la fila p tal que A[p,j] es el elemento mayor en valor absoluto de la columna j.
    La búsqueda se hace a partir de la diagonal principal por default."""
    if i is None:
        i = j + 1
        
    mayor = max(A[i:, j], key = abs)
    p = list(A[i:, j]).index(mayor) + i

    return p

def pivoteoTotal(A, j):
    """Obtiene la fila p y la columna q tal que A'[p,q] es el elemento mayor en valor absoluto de la
    submatriz A' que se obtiene al considerar solo las filas y columnas j:n-1."""
    n, _ = A.shape
    pivotes = np.zeros(n - j, dtype = np.int)
    # obtenemos los elementos más grandes de cada columna
    for k in range(j, n):
        p = pivoteoParcial(A, k, j)
        pivotes[k-j] = p

        mayores = [A[p, j + i] for i, p in enumerate(pivotes)]
        mayor = max(mayores, key = abs)
        q = mayores.index(mayor) + j

        return pivotes[q - j], q