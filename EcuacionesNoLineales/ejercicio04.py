# ejercicio 04
# por: edgar armando trejo lópez
import numpy as np
from MetodosEcNL import biseccion
# definimos la función
f = lambda x: x**2 - 3
# resolvemos sabiendo que la raíz está entre cero y 3
tol = 1/10**4
x = biseccion(f, 0, 3, tol = tol)
# imprimimos el resultado
print('La aproximación a la raíz de 3 que da el algoritmo de bisección es ')
print('x = {}'.format(x))
# checamos el número que da numpy como raíz de 3 y el que obtuvimos.
print('\nRespecto al número que usa numpy, hay un error absoluto de')
print('e = {}'.format(abs(np.sqrt(3) - x)))