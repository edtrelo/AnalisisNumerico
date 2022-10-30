import numpy as np
from Pivoteo import *

def factLU(M):
    """Obtiene la factorización LU (método estándar) de una matriz, donde L es una matriz triangular inferior
    y U una matriz tringular superior.
    
    M(list of lists):
        M representa una matriz cuadrada.
        
    returns:
        L(np.ndarray):
            matriz triangular inferior de la descomposición LU.
        U(np.ndarray):
            matriz triangular superior de la descomposción LU.
            
    raises:
        Genera una expeción si en algún momento la diagonal tiene un elemento igual a cero."""

    A = np.copy(M)
    n, _ = A.shape
    # inicializamos L como I_n
    L = np.identity(n)
    # para cada renglón, los componentes de L debajo de la diagonal principal serán los factores de
    # pivoteo para cada fila. Es decir a_ij / a_ii, para j > i.
    # A la vamos a pivotear para cada i hasta la penúltima columna.
    for i in range(n-1):
        # estamos en la i-ésima columna.
        # como no hacemos pivoteo, si algo en la diagonal es cero, entonces falla la factorización.
        if A[i,i] == 0:
            raise Exception("La factorización LU falla.")
        for j in range(i+1, n):
            # para cada renglón debajo de la diagonal principal, aplicamos el pivoteo
            fact = A[j,i] / A[i,i]
            # cambiamos el renglón Rj por Rj - f*Ri
            A[j] = A[j] - fact*A[i]
            # añadimos el factor a la identidad
            L[j, i] = fact
    # para este paso, A ya es U. Sabemos que M^-1 es precisamente como cosntruimos L. 
    return L, A

def factLUpivpar(M):
    # copiamos la matriz como un array de tipo np.float64
    A = np.array(M, dtype = np.float64)
    n, _ = A.shape

    L = np.identity(n)
    P = np.identity(n)

    for j in range(n-1):
        PjInv = np.identity(n)
        MjInv = np.identity(n)

        p = pivoteoParcial(A, j)
        if p != j:
            # intercambiamos renglones
            A[[j, p]] = A[[p, j]]
            PjInv[[j, p]] = PjInv[[p, j]]

        for i in range(j+1, n):
            
            fact = A[i, j] / A[j, j]
            MjInv[i, j] = fact
            A[i] = A[i] - fact*A[j]

        if p != j:
            MjInv[[j, p]] = MjInv[[p, j]]
        
        L = np.dot(L, MjInv)
        P = np.dot(PjInv, P)
        

    return L, A, P
    
def factLUpivtot():
    pass

def factCholesky():
    pass

if __name__ == "__main__":

    A = [[2,4,3,5],
        [-4,-7,-5,-8],
        [6,8,2,9],
        [4,9,-2,14]]

    L, U, P = factLUpivpar(A)

    print(L)
    print(U)
    print(np.dot(L, U))

