from MatricesCuadradas import MatrizCuadrada
from SustitucionGaussiana import sustGauss
from Sustitucion import *
import numpy as np

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
        L, U = None, None

        if pivoteo is None:
            # obtenemos la factorización L, U estándar
            # recordar que son objetos de la clase MatrizCuadrada
            L, U = self.Mat.factorizarLU()
        elif pivoteo == 'parcial':
            pass
        elif pivoteo == 'total':
            pass
        else:
            raise ValueError("El parámetro para el pivoteo no es una opción viable.")
        # resuelve LY = b
        Y = sustDelante(L.A, self.vec)
        # resuelve UX = Y
        X = sustAtras(U.A,Y)
        return X
        


A = [   [2,4,3,12],
        [-40,-7,-5,-8],
        [6,8,2,9],
        [4,9,-2,14]] 

b = [1,1,1,1]

S = SistemaLineal(A, b)

X = S._byLUstandar()
print(X)

Y = np.linalg.solve(A, b)
print(Y)