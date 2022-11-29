import numpy as np
from Sust import sustAtras

def _obtenerCyS(a, i, j):
    """Obtiene el coseno y el seno de la rotación de Givens.
    
    Args:
        a(np.ndarray): es la columna donde quieren hacerse los ceros.
        i, j(int): los índices tales que aj se hace cero a través de ai.
        
    Returns:
        (c, s): tupla de floats."""
    norma = np.sqrt(a[i]**2 + a[j]**2)
    c = a[i]/norma
    s = a[j]/norma
    return c, s

def _obtenerRotGivens(a, i, j):
    """Obtiene la matriz G de Givens tal que Ga hace cero la j-ésima entrada de a 
    alterando también la entrada i-ésima de a."""
    m = len(a)
    # G = I_m
    G = np.eye(m)
    # obtenemos los valores de coseno y seno.
    c, s = _obtenerCyS(a, i, j)
    # colocamos c y s en su lugar correspondiente
    G[i, i] = c
    G[j, j] = c
    G[j, i] = -s
    G[i, j] = s
    return G

def byGivens(Matriz, vector):
    """Soluciona el sistema de mínimos cuadrados a través de la factorización QR con
    el método de rotaciones de Givens.
    
    Args:
        Matriz(np.ndarray): es la matriz de sistema Ax = b.
        vector(np.ndarray): es el vector resutlado del sistema Ax=b.
        
    Returns:
        (np.ndarray): la mejor aproximación a Ax=b.
        
    Raises:
        ValueError: si el sistema tiene menos ecuaciones que variables."""
    A = np.array(Matriz, dtype = np.float64)
    b = np.array(vector, dtype = np.float64)
    m, n = A.shape
    if m < n:
        raise ValueError("El sistema debe tener más ecuaciones que incógnitas.")
    # recorremos columnas
    for j in range(n):
        #recorremos filas
        for i in range(j+1, m):
            a = A[:, j]
            # hacemos cero debajo
            if A[i, j] != 0:
                G = _obtenerRotGivens(a, j, i)
                A = np.dot(G, A)
                b = np.dot(G, b)
    R, b = A[:n], b[:n]
    # resolvemos Rx = b con el subsistema cuadrado.
    X = sustAtras(R, b)
    return X
