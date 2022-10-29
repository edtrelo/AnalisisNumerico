import numpy as np
from Factorizacion import *

class MatrizCuadrada:
    """
    
    Properties:
        determinant(float):
            El determinante de la matriz.
        size(int):
            El tamaño de la matriz.
        normaUno(float):
            Norma uno de la matriz."""

    def __init__(self, A, dtype = np.float64):
        self.A = None
        self.n = None

        try:
            self.A = np.array(A, dtype)
        except TypeError:
            raise TypeError("El argumento no representa un arreglo n dimensional.")

        # n es el tamaño de la matriz
        esCuad, n = self._verificarCuad()
        if esCuad:
            self.n = n

    def _verificarCuad(self):
        """Verifica si un numpy.ndarray es una matriz cuadrada."""
        n, m = self.A.shape
        if n != m:
            raise ValueError("La matriz no es cuadrada.")
        return True, n
        
    def interRenglones(self, i, j):
        """Intercambia los renglones i y j de la matriz."""
        self.A[[i, j]] = self.A[[j, i]]

    def obtenerPivoteParcial(self, j, i = None):
        """Obtiene la fila p tal que A[p,j] es el elemento mayor en valor absoluto de la columna j.
        La búsqueda se hace a partir de la diagonal principal por default."""
        if i is None:
            i = j + 1
        
        mayor = max(self.A[i:, j], key = abs)
        p = list(self.A[i:, j]).index(mayor) + i

        return p

    def obtenerPivoteTotal(self, j):
        """Obtiene la fila p y la columna q tal que A'[p,q] es el elemento mayor en valor absoluto de la
        submatriz A' que se obtiene al considerar solo las filas y columnas j:n-1."""

        pivotes = np.zeros(self.n - j, dtype = np.int)
        # obtenemos los elementos más grandes de cada columna
        for k in range(j, self.n):
            p = self.obtenerPivoteParcial(k, j)
            pivotes[k-j] = p

        mayores = [self.A[p, j + i] for i, p in enumerate(pivotes)]
        mayor = max(mayores, key = abs)
        q = mayores.index(mayor) + j

        return pivotes[q - j], q

    def subMatriz(self, i, j):
        """Crea un objeto MatrizCuadrada con la sub matriz obtenida al eliminar
        la fila i y la columna j de la matriz."""
        Aij = np.copy(self.A)
        Aij = np.delete(Aij, i, axis=0)
        Aij = np.delete(Aij, j, axis=1)
        return MatrizCuadrada(Aij)

    def _determinante(self):
        """Calcula el determinante de la matriz."""
        if self.n == 1:
            return self.A[0, 0]
        # determinante de una matriz de 2x2: ad - bc
        elif self.n == 2:
            return self.A[0, 0]*self.A[1, 1] - self.A[0, 1]*self.A[1, 0]
        # determinante de manera recursiva
        else:
            deter = 0.0
            for i in range(self.n):
                sign = (-1)**i
                # sobre la primer fila
                subM = self.subMatriz(0, i)
                deter += sign*self.A[0, i] * subM.determinante
            return deter
        
    @property
    def determinante(self):
        return self._determinante()

    @property
    def size(self):
        return self.n

    @property
    def normaUno(self):
        # sobre columnas
        sumas = np.zeros(self.n)
        for j in range(self.n):
            sum_colj = 0.0
            for i in range(self.n):
                sum_colj += abs(self.A[i, j])
            sumas[j] = sum_colj
        return max(sumas)

    @property
    def normaInf(self):
        # sobre filas
        sumas = np.zeros(self.n)
        for i in range(self.n):
            sum_rowi = 0.0
            for j in range(self.n):
                sum_rowi += abs(self.A[i, j])
            sumas[i] = sum_rowi
        return max(sumas)

    def _estimarNormaInv(self):
        c = np.choice([-1, 1], size = self.n)
        



    def _normaInv(self):
        pass

    def _cond(self):
        pass
        
    def factorizarLU(self, pivoteo = None):
        """el parámetro pivoteo puede ser lo siguiente:
            None -> no se hace pivoteo.
            'parcial' -> se hace pivoteo parcial.
            'total' -> se hace pivoteo total."""
        if pivoteo is None:
            L, U = factLU(self.A)
            return MatrizCuadrada(L), MatrizCuadrada(U)


A = [   [2,4,3,12],
        [-40,-7,-5,-8],
        [6,8,2,9],
        [4,9,-2,14]]

A = np.array(A)

Q = MatrizCuadrada(A)

for i in range(3):
    print(Q.obtenerPivoteTotal(i))

print(Q.determinante)
print(np.linalg.det(A))
print(np.linalg.norm(A, ord = 1))
print(Q.normaInf)




