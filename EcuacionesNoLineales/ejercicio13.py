# ejercicio 13
# por: edgar armando trejo lópez
import numpy as np
from ejercicio01 import *
# NOTA: Por default, la tolerancia ya es 1/10**6

# definimos el polinomio
# los coeficientes van ordenados del grado mayor a menor
p = np.poly1d([230, 18, 9, -221, -9])
# definimos la derivada 
dp = np.poly1d([230*4, 18*3, 9*2, -221])

# a) intervalo [-1, 0] ----------------------------------------------------------- 
print('Para el intervalo [-1, 0], las raíces del polinomio están dadas por')
# encontramos las raíces usando los diferentes métodos.
# método de la falsa regla.
x = falsaRegla(p, -1, 0)
print('\tRegla Falsa: x = {}'.format(x))
# método de la secante.
x = secante(p, -1, 0)
print('\tSecante: x = {}'.format(x))
# método de newton.
# tomamos el punto inicial como el punto medio del intervalo.
x = newton(p, dp, -0,5)
print('\tNewton: x = {}'.format(x))

# a) intervalo [0, 1] ----------------------------------------------------------- 
print('\nPara el intervalo [0, 1], las raíces del polinomio están dadas por')
# encontramos las raíces usando los diferentes métodos.
# método de la falsa regla.
x = falsaRegla(p, 0, 1)
print('\tRegla Falsa: x = {}'.format(x))
# método de la secante.
# usando x0 = 0, x1 = 1, me daba una raíz en otro intervalo
x = secante(p, 0.1, 0.9)
print('\tSecante: x = {}'.format(x))
# método de newton.
# tomamos el punto inicial como el punto final del intervalo.
# usando el punto medio como x0, me daba una raíz en otro intervalo.
x = newton(p, dp, 1)
print('\tNewton: x = {}'.format(x))
