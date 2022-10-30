from MatricesCuadradas import MatrizCuadrada
from SustitucionGaussiana import sustGauss
from Sustitucion import *
import numpy as np
from SolLU import *

class SistemaLineal:

    def __init__(self, A, b):
        self.Mat = MatrizCuadrada(A)
        self.vec = np.array(b, dtype = np.float64)

    def solve(self, met = None):
        if met is None:
            return self._eliminacionGaussiana()


    def _eliminacionGaussiana(self):
        X = sustGauss(self.Mat.A, self.vec)
        return X

    def _porLU(self, pivoteo = None):
        """Resuelve el sistema Ax = b por medio de la factorización L,U (sin pivoteo, pivoteo
        parcial o pivoteo total) de la siguiente manera:
        
        Como A = LU, entonces LUx = b. Se hace Ux = y, y se resuelve primero el sistema
        Ly = b por medio de sustitución hacia adelante. Finalmente, se resuelve 
        Ux = y con sustituación hacia atrás.

        La factorización se hace sin pivoteo por default. Para elegir entre una de las dos opciones
        restantes la key para el parámetro debe ser la siguiente:

            -> 'parcial': para pivoteo parcial.
            -> 'total': para pivoteo total.
        
        returns:
            X(np.ndarray):
                Solución al sistema AX = b.
            pivoteo(None o str):
                """
        if pivoteo is None:
            # obtenemos la factorización L, U estándar
            # recordar que son objetos de la clase MatrizCuadrada
            return resolverConLU(self.Mat, self.vec)
        elif pivoteo == 'parcial':
            pass
        elif pivoteo == 'total':
            pass
        else:
            raise ValueError("El parámetro para el pivoteo no es una opción viable.")
        
        


