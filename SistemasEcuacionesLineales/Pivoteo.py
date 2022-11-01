# ALGORITMOS DE PIVOTEO.

import numpy as np

def pivoteoParcial(A, j, i = None):
    """Obtiene la fila p tal que A[p,j] es el elemento mayor en valor absoluto de la columna j.
    La búsqueda se hace a partir de la diagonal principal por default.
    
    Args:
        j(int):
            Columna sobre la que se hace el pivoteo.
        i(int o None):
            si i es None, i se toma como j + 1. Si i es otro entero, la búsqueda del
            pivote se hace sobre ese renglón.
                
    Returns:
        p(int):
            renglón que tiene al elemento de mayor valor absoluto en la columna j
            a partir del renglón i."""
    # si i es None, por default buscamos debajo de la diagonal.
    if i is None:
        i = j + 1
    # obtenemos el elemento mayor en valor absoluto de la columna
    mayor = max(A[i:, j], key = abs)
    # obtenemos el índice del máximo.
    p = list(A[i:, j]).index(mayor) + i
    return p

def pivoteoTotal(A, j):
    """Obtiene la fila p y la columna q tal que A'[p,q] es el elemento mayor en 
    valor absoluto de la submatriz A' que se obtiene al considerar solo las filas 
    y columnas j:n-1.
    
    Args:
        j(int):
            Fila y columna desde la que se inicia la búsqueda.
                
    Returns:
        p(int):
            fila donde está el pivote.
        q(int):
            columna donde está el pivote."""
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