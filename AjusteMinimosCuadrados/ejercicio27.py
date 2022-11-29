# ejercicio 27
# edgar armando trejo lópez
import sys
import numpy as np
from Householder import *
from Normales import *
from Givens import *
from GramSchmidt import *
# obtenemos el valor de el épsilon de máquina.
e = sys.float_info.epsilon
# la raíz de épsilon de máquina
sqrtep = np.sqrt(e)
# definimos la función de epsilon
def matrizEps(E):
    """Regresa la matriz del problema para un valor determinado de e."""
    A = [[1, 1, 1],
        [E, 0, 0],
        [0, E, 0],
        [0, 0, E]]
    A = np.array(A, dtype = np.float64)
    return A
# definimos b
b = np.array([1,0,0,0], dtype = np.float64)
# para cada método buscamos un 3 numeros cercanos a epsilon y su raíz
# para obtener soluciones que existan al problema.

# método normales
print('Para el método de Ecuaciones Normales:')
contador = 0
while contador < 3:
    try:
        ep = np.random.uniform(e, sqrtep)
        A = matrizEps(ep)
        X = byNormales(A, b)
        contador += 1
        print('con \u03b5 = {}, la solución es X={}'.format(ep, X))
    except:
        pass
# por householder
print('\nPara el método de Householder:')
contador = 0
subcontador = 0
while contador < 3 and subcontador < 100:
    try:
        ep = np.random.uniform(e, sqrtep)
        A = matrizEps(ep)
        X = byHouseholder(A, b)
        contador += 1
        print('con \u03b5 = {}, la solución es X={}'.format(ep, X))
    except:
        subcontador += 1
if subcontador == 100:
    print('Tras 100 iteraciones para Householder, no hay solución.')

# por givens
print('\nPara el método de Givens:')
contador = 0
while contador < 3:
    try:
        ep = np.random.uniform(e, sqrtep)
        A = matrizEps(ep)
        X = byGivens(A, b)
        contador += 1
        print('con \u03b5 = {}, la solución es X={}'.format(ep, X))
    except:
        pass

# por gram
print('\nPara el método de Gram-Schmidt:')
contador = 0
subcontador = 0
while contador < 3 and subcontador < 100:
    try:
        ep = np.random.uniform(e, sqrtep)
        A = matrizEps(ep)
        X = byGram(A, b)
        contador += 1
        print('con \u03b5 = {}, la solución es X={}'.format(ep, X))
    except:
        subcontador += 1
if subcontador == 100:
    print('Tras 100 iteraciones para Householder, no hay solución.')