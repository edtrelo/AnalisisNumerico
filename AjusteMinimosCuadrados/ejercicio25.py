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
print(Xnorm)
# -------------------- SOLUCIÓN POR QR ---------------------------
# --------------- SOLUCIÓN POR HOUSEHOLDER -----------------------
Xhouse = byHouseholder(M, yper)
print(Xhouse)

# inciso a) ---------------------------------------------------------------------------------
# ¿Para cuál de los métdos la solución es más sensible a la perturbación generada?
# Entre más pequeño sea el coseno del ángulo (más cercano el ángulo a pi/2) más
# sensible será la solución. Por la interpretación geométrica de los mínimos cuadrados
# cos(0) = |Ax|/|b| = |b_aprox|/|b|

cos_norm = np.linalg.norm(np.dot(M, Xnorm)) / np.linalg.norm(yper)
cos_house = np.linalg.norm(np.dot(M, Xhouse)) / np.linalg.norm(yper)

if cos_house < cos_norm:
    print('La solución del método de Householder es más sensible.')
else:
    print('La solución de las Ecuaciones Normales es más sensible.')

# inciso b) ----------------------------------------------------------------------------------
# ¿Cuál de los métodos está más próximo a tener la solución exacta dada por xi = 1?.

dist_norm = np.linalg.norm(coeficientes - Xnorm)
dist_house = np.linalg.norm(coeficientes - Xhouse)

if dist_norm < dist_house:
    print('Las Ecuaciones Normales da una solución más cercana a la real.')
else:
    print('El método de Householder da una solución más cercana a la real.')

# inciso c) -----------------------------------------------------------------------------------
# ¿La diferencia en las soluciones afecta en el ajuste de puntos (ti, yi) por el polinomio, por qué?.
