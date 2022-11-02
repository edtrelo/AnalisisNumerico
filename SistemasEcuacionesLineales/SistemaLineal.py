from MatricesCuadradas import MatrizCuadrada
from SustitucionGaussiana import sustGauss
from Sustitucion import *
import SolCholesky
import SolLU
import numpy as np

class SistemaLineal:

    def __init__(self, A, b):
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

    def porLU(self, pivoteo = None):
        """Resuelve el sistema Ax = b por medio de la factorización L,U (sin pivoteo, pivoteo
        parcial o pivoteo total) de la siguiente manera:
        
        Como A = LU, entonces LUx = b. Se hace Ux = y, y se resuelve primero el sistema
        Ly = b por medio de sustitución hacia adelante. Finalmente, se resuelve 
        Ux = y con sustituación hacia atrás.

        La factorización se hace sin pivoteo por default. Para elegir entre una de las dos opciones
        restantes la key para el parámetro debe ser la siguiente:

            -> 'parcial': para pivoteo parcial.
            -> 'total': para pivoteo total.
        
        Returns:
            X(np.ndarray):
                Solución al sistema AX = b."""
        if pivoteo is None:
            # obtenemos la factorización L, U estándar
            # recordar que son objetos de la clase MatrizCuadrada
            return resolverConLU(self.Mat.A, self.vec)
        elif pivoteo == 'parcial':
            return resolverConLUParcial(self.Mat.A, self.vec)
        elif pivoteo == 'total':
            return resolverConLUTotal(self.Mat.A, self.vec)
        else:
            raise ValueError("El parámetro para el pivoteo no es una opción viable.")
    
    def porCholesky(self):
        """Resuelve el sistema Ax = b por medio de la factorización LL^t.
        
        Returns:
            X(np.ndarray):
                Solución al sistema AX = b."""
        
        if self.Mat.esCholesky():
            return resolverConCholesky(self.Mat.A, self.vec)
        else:
            raise Exception("La Matriz A no tiene factorización de Cholesky.")

        