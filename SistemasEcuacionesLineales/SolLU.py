# SOLUCIÓN DE S.L. A TRAVÉS DE FACTORIZACIÓN LU.

import numpy as np
from Sustitucion import *
from Factorizacion import *

def resolverConLU(A, b):
    """Resuelve el sistema Ax = b obteniendo la factorización LU de A. Resuelve
    Ly = b con sustitución hacia adelante, y después resuelve Ux = y.
    
    Args:
        A(np.ndarray):
            Matriz Cuadrada.
        b(np.ndarray)
    
    Returns:
        X(np.ndarray):
            Solución a AX=b."""
    L, U = factLU(A)
    # resuelve LY = b
    Y = sustDelante(L, b)
    # resuelve UX = Y
    X = sustAtras(U, Y)
    return X

def resolverConLUParcial(A, b):
    """Resuelve el sistema Ax=b con la factorización L'U de A obtenida con pivoteo 
    parcial. Resuelve PAx = PL'Ux = Pb, primero se soluciona PL'y=Pb con sustitución hacia
    adelante. Finalmente se obtiene Ux=y con sustitución hacia atrás.
    
    Args:
        A(np.ndarray):
            Matriz Cuadrada.
        b(np.ndarray)
    
    Returns:
        X(np.ndarray):
            Solución a AX=b."""
    # obtenemos la factorización.
    L, U, P = factLUpivpar(A)
    # obtenemos L=PL' tal que L es triangular inferior.
    Ltrin = np.dot(P, L)
    # permutamos b
    # Ax=b > L'U x = b -> PL'U x = Pb -> PL' y = Pb 
    bper = np.dot(P, b)
    # Resolvemos Ly=Pb
    Y = sustDelante(Ltrin, bper)
    X = sustAtras(U, Y)
    
    return X

def resolverConLUTotal():
    pass

A = [[2,4,3,5],
        [-4,-7,-5,-8],
        [6,8,2,9],
        [4,9,-2,14]]

b = [1,1,1,1]

A = np.array(A)
b = np.array(b)

X = np.linalg.solve(A, b)
print(X)

print(resolverConLU(A, b))
print(resolverConLUParcial(A, b))