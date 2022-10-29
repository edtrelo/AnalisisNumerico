import numpy as np

def factLU(M):
    """Obtiene la factorización LU de una matriz, donde L es una matriz triangular inferior
    y U una matriz tringular superior."""
    A = np.array(M, dtype = np.float64)
    n, _ = A.shape
    # inicializamos L como I_n
    L = np.identity(n)
    # para cada renglón, los componentes de L debajo de la diagonal principal serán los factores de
    # pivoteo para cada fila. Es decir a_ij / a_ii, para j > i.
    # A la vamos a pivotear para cada i hasta la penúltima columna.
    for i in range(n-1):
        if A[i,i] == 0:
            raise Exception
        for j in range(i+1, n):
            fact = A[j,i] / A[i,i]
            # cambiamos el renglón Rj por Rj - f*Ri
            A[j] = A[j] - fact*A[i]
            # añadimos el factor a la identidad
            L[j, i] = fact
    # para este paso, A ya es U. Sabemos que M^-1 es precisamente como cosntruimos L. 
    return L, A

def factParcialLU():
    pass

def factTotalLU():
    pass

def factCholesky():
    pass

A = [[2,4,3,5],
    [-4,-7,-5,-8],
    [6,8,2,9],
    [4,9,-2,14]]

L, U = factLU(A)

print(L)
print(U)
