import numpy as np
from Sust import sustAtras, sustDelante

def factCholesky(M):
    """Obtención de la factorización M=LL^t. Suponemos que M es definida positiva y simétrica.
    
    Args:
        M(np.ndarray):
            Matriz definida positiva y simétrica.
        
    Returns:
        L(np.ndarray):
            Matriz triangular inferior tal que A=LL^t.
    """
    A = np.array(M, dtype = np.float64)
    n, _ = A.shape
    L = np.zeros((n,n))
    # recorremos columnas
    for i in range(n):
        for k in range(n):
            # dejamos en cero a los elementos arriba de la diagonal.
            if i > k:
                pass
            # solo implemento las fórumlas que Iván nos presentó.
            # si i = k:
            # l_{kk} = ( a_{kk} - \sum_{j=0}^{k-1} (l_{kj}^2) )^(1/2)
            # si i < k:
            # l_{ki} = (a_{ki}-\sum_{j=0}^{i-1} l_{ij}*l_{kj}) / l_{ii}
            # recordando que k = 0,1...,n-1 e i = 0,1,...,n-1

            # establecemos los elementos debajo de la diagonal.
            elif k > i:
                suma = 0.0
                for j in range(i):
                    suma += L[i, j]*L[k, j]
                L[k, i] = (A[k, i] - suma)/L[i, i]
            # establecemos los elementos de la diagonal.
            else:
                suma = 0.0
                for j in range(k):
                    suma+= np.square(L[k, j])
                L[k, k] = np.sqrt(A[k, k] - suma)
    return L

def _resolverConCholesky(A, b):
    """Resuelve el sistema Ax=b usando la factorización de Cholesky de A, A=LL^t.
    Se asume que A es definida positiva y además simétrica.
    
    Args:
        A(np.ndarray):
            Matriz Cuadrada definida positiva y simñetrica.
        b(np.ndarray)
    
    Returns:
        X(np.ndarray):
            Solución a AX=b."""
    L = factCholesky(A)
    # tenemos ahora el sistema LL^tx = b
    # Sea y = L^tx
    # Resolvemos Ly=b por sustitución hacia adelante.
    Y = sustDelante(L, b)
    # Resolvemos L^tx=y por sustitución hacia atrás.
    X = sustAtras(L.T, Y)
    return X

def byNormales(Matriz, vector):
    """Soluciona el problema de minimos cuadrados por el método de ecuaciones normales."""
    A = np.array(Matriz, dtype = np.float64)
    b = np.array(vector, dtype = np.float64)
    # el sistema de ecuaciones normales es A^t*Ax=A^t*b 
    S = np.dot(A.T, A)
    v = np.dot(A.T, b)
    # resolvemos Sx = v con Cholesky
    X = _resolverConCholesky(S, v)
    return X
