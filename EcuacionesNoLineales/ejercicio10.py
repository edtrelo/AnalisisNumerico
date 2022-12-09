# ejercicio 10
# por: edgar armando trejo lópez
import numpy as np
from ejercicio01 import newton
# definimos la tolerancia para estos ejercicios
tol = 1/10**5
print('Soluciones por el Método de Newton:\n')
# inciso a) ---------------------------------------------------------------
# definimos la función
f = lambda x: np.exp(x) + 2**(-x) + 2*np.cos(x) - 6
# escribimos la derivada
df = lambda x: np.exp(x) - 2**(-x) * np.log(2) - 2*np.sin(x)
# obtenemos la solución
# el punto inicial lo elegimos entre 1 y 2. Tomamos el punto medio.
x = newton(f, df, 1.5, tol = tol)
print('a) La raíz de f(x) = e^x + 2^(-x) + 2cos(x) - 6 en el intervalo [1, 2] es x = {}'.format(x))
# inciso c) ---------------------------------------------------------------
# definimos la función
f = lambda x: np.exp(x) - 3*x**2
# esribimos la derivada
df = lambda x: np.exp(x) - 6*x
print('\nc) La raíz de f(x) = e^x - 3x^2 ')
# obtenemos la solución
# el primer punto inicial lo elegimos entre 0 y 1. Tomamos el punto medio.
x = newton(f, df, 0.5, tol = tol)
print('\ten el intervalo [0, 1] es x = {}'.format(x))
# obtenemos la solución
# el segundo punto inicial lo elegimos entre 3 y 5. Tomamos el punto medio.
x = newton(f, df, 4, tol = tol)
print('\ten el intervalo [3, 5] es x = {}'.format(x))