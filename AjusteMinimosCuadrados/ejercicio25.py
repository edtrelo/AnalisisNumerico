# Ejercicio 25
# Autor: Edgar Armando Trejo López
import numpy as np
from Householder import byHouseholder
import matplotlib.pyplot as plt
from Normales import byNormales
# establecemos los valores para n, m y epsilon.
n = 12
m = 21
e = 10**(-10)
# los coeficientes del polinomio son todos 1's
coeficientes = np.ones(n)
# creamos arrays vacíos para guardar aquí los t's y y's
t = np.zeros(m)
y = np.zeros(m)
# definimos cada t_i y cada y_i
for i in range(m):
    t[i] = i/(m-1)
    # y_i es la evaluación del polinomio en t_i
    y[i] = np.polyval(coeficientes, t[i]) 
# perturbamos y
# obtenemos un arreglo de tamaño m donde cada entrada es un aleatorio.
u = np.random.random(size = m)
# definimos la perturbación
yper = y + e*(2*u - 1)
# establecemos la M del sistema de ecuaciones que vamos a querer resolver
M = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        # la j-ésima columna tendrá las evaluaciones al exponente j-1
        M[i, j] = t[i]**j
# entonces x será [x1, x2, ..., xn]
# --------------- SOLUCIÓN POR NORMALES --------------------------
Xnorm = byNormales(M, yper)

plt.plot(t, yper, marker = 'o', linestyle = 'none')
plt.show()