from MatricesCuadradas import MatrizCuadrada
from SustitucionGaussiana import *
from Sustitucion import *
from SolCholesky import *
from SolLU import *
import numpy as np
from time import time

class SistemaLineal:

    def __init__(self, A, b):
        if type(A) == MatrizCuadrada:
            self.Mat = A
        else:
            self.Mat = MatrizCuadrada(A)
        if self.Mat.size != len(b):
            raise Exception("El sistema lineal no puede crearse. Los tamaños no coinciden.")
        self.vec = np.array(b, dtype = np.float64)
        
    def porGauss(self, pivoteo = None):
        """Resuelve el sistema Ax = b por medio de eliminación gaussiana a la matriz aumentada
        A|b, de manera convencional o con pivoteo parcial.
        
        La eliminación se hace sin pivoteo por default. Para elegir la opción de pivoteo parcial
        la key del argumento pivoteo debe ser 'parcial'.

        Returns:
            X(np.ndarray):
                Solución al sistema AX = b."""        
        if pivoteo is None:
            return elimGauss(self.Mat.A, self.vec)
        elif pivoteo == 'parcial':
            return elimGaussPar(self.Mat.A, self.vec)
        else:
            raise ValueError("El argumento de pivoteo no es válido.")

    def porLU(self, pivoteo = None, medir = False):
        """Resuelve el sistema Ax = b por medio de la factorización L,U (sin pivoteo, pivoteo
        parcial o pivoteo total) de la siguiente manera:
        
        Como A = LU, entonces LUx = b. Se hace Ux = y, y se resuelve primero el sistema
        Ly = b por medio de sustitución hacia adelante. Finalmente, se resuelve 
        Ux = y con sustituación hacia atrás.

        La factorización se hace sin pivoteo por default. Para elegir entre una de las dos opciones
        restantes la key para el parámetro debe ser la siguiente:

            -> 'parcial': para pivoteo parcial.
            -> 'total': para pivoteo total.

        Args:
            pivoteo(str or None):
                Tipo de pivoteo.
            medir(bool):
                True para regresar el tiempo que tardó el algoritmo.
        
        Returns:
            X(np.ndarray):
                Solución al sistema AX = b.
            (si medir es True):
                también regresa t(float), tiempo que tardó el algoritmo en segundos."""
        X = None
        start = time()
        if pivoteo is None:
            # obtenemos la factorización L, U estándar
            # recordar que son objetos de la clase MatrizCuadrada
            X = resolverConLU(self.Mat.A, self.vec)
        elif pivoteo == 'parcial':
            X = resolverConLUParcial(self.Mat.A, self.vec)
        elif pivoteo == 'total':
            X = resolverConLUTotal(self.Mat.A, self.vec)
        else:
            raise ValueError("El parámetro para el pivoteo no es una opción viable.")
        t = time() - start
        if medir:
            return X, t
        return X
    
    def porCholesky(self):
        """Resuelve el sistema Ax = b por medio de la factorización LL^t.
        
        Returns:
            X(np.ndarray):
                Solución al sistema AX = b."""
        
        if self.Mat.esCholesky():
            return resolverConCholesky(self.Mat.A, self.vec)
        else:
            raise Exception("La Matriz A no tiene factorización de Cholesky.")

        