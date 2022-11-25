import sys
import numpy as np
from Householder import *
# obtenemos el valor de el épsilon de máquina.
e = sys.float_info.epsilon

epsilon1 = e + 1e-31
print('e', e)
sqrtep = np.sqrt(e)

eps = np.random.uniform(low = sqrtep, high = e, size = 3)

print(eps)

for E in eps:
    A = [[1, 1, 1],
        [E, 0, 0],
        [0, E, 0],
        [0, 0, E]]

    b = [1,0,0,0]

    X = byHouseholder(A, b)
    print('La solución para epsilon = {} es {}'.format(E, X))