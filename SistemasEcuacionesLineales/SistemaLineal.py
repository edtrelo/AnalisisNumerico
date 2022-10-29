from MatricesCuadradas import MatrizCuadrada
from SustitucionGaussiana import sustGauss
import numpy as np

class SistemaLineal:

    def __init__(self, A, b):
        self.Mat = MatrizCuadrad(A)
        self.vec = np.array(b, dtype = np.float64)

    def eliminacionGaussiana(self):
        X = sustGauss(self.Mat.A, self.vec)
        return x

    

A = [   [2,4,3,12],
        [-40,-7,-5,-8],
        [6,8,2,9],
        [4,9,-2,14]] 

b = [1,1,1,1]

S = SistemaLineal(A, b)




