import numpy as np
from Factorizacion import factCholesky, factCholeskyDiag
from Sustitucion import *

def resolverConCholesky(A, b):
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