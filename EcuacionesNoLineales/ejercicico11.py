# ejercicio 11
# por: edgar armando trejo lópez
import numpy as np
from MetodosEcNL import secante
# definimos la tolerancia para estos ejercicios
tol = 1/10**5
print('Soluciones por el Método de la Secante:\n')
# inciso a) ---------------------------------------------------------------
# definimos la función
f = lambda x: np.exp(x) + 2**(-x) + 2*np.cos(x) - 6
# obtenemos la solución
# el punto inicial lo elegimos entre 1 y 2.
x = secante(f, 1, 2, tol = tol)
print('a) La raíz de f(x) = e^x + 2^(-x) + 2cos(x) - 6 en el intervalo [1, 2] es x = {}'.format(x))
# inciso c) ---------------------------------------------------------------
# definimos la función
f = lambda x: np.exp(x) - 3*x**2
print('\nc) La raíz de f(x) = e^x - 3x^2 ')
# obtenemos la solución
# el primer punto inicial lo elegimos entre 0 y 1.
x = secante(f, 0, 1, tol = tol)
print('\ten el intervalo [0, 1] es x = {}'.format(x))
# obtenemos la solución
# el segundo punto inicial lo elegimos entre 3 y 5.
x = secante(f, 3, 5, tol = tol)
print('\ten el intervalo [3, 5] es x = {}'.format(x))