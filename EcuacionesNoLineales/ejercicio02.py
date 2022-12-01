# ejercicio 02
# por: edgar armando trejo lópez
import numpy as np
from MetodosEcNL import biseccion
# Usar el Método de Bisección (programado) para encontrar la solución de las siguientes funciones
# con una tolerancia de 10^-5
tol = 1/10**5
# inciso a) --------------------------------------------------------------------------------------
# definimos la función
f = lambda x: x - 2**(-x)
# obtenemos la solución por el método de la bisección
x = biseccion(f, 0, 1, tol = tol)
# imprimimos el resultado
print('a) El cero de f(x) = x - 2^(-x) usando el intervalo [0, 1] es x = {}\n'.format(x))
# inciso d) --------------------------------------------------------------------------------------
# definimos la función
f = lambda x: x*np.cos(x) - 2*x**2 + 3*x - 1
print('d) El cero de f(x) = x cos(x) - 2x^2 + 3x - 1')
# obtenemos la solución para el primer intervalo
x = biseccion(f, 0.2, 0.3, tol = tol)
# imprimimos el resultado
print('\tusando el intervalo [0.2, 0.3] es x = {}'.format(x))
# obtenemos la solución para el segundo intervalo
x = biseccion(f, 1.2, 1.3, tol = tol)
# imprimimos el resultado
print('\tusando el intervalo [1.2, 1.3] es x = {}'.format(x))