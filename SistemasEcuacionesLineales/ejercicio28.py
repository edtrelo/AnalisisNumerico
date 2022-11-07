# Ejercicio 28
# Autor: Edgar Armando Trejo López
import numpy as np
from MatricesCuadradas import *
from SistemasLineales import *
import matplotlib.pyplot as plt
# inciso a)
print("¿Qué sucede cuando se usa Eliminación Gaussina con pivoteo parcial?")
print("\n\tR. Todas las filas van a seguir en su lugar mientras se pivotie sobre la ")
print("\t(n-1)-ésima columna. Al llegar a la última columna, como siempre se estuvo ")
print("\tusando factores positivos, el pivote se va a encontrar en la última fila. ")
print("\tEn conclusión, solo se va a usar una matriz de permutación.")
# inciso b) e inciso c)
# creamos una función que nos genere la matriz del ejercicio con varios tamaños.
def crearMatrizEj28(n):
    """Crea la matriz del ejercicio 28 para una matriz de nxn.
    
    Returns:
        Matriz(Matriz Cuadrada)"""
    if n<2:
        return None
    else:
        Matriz = np.zeros((n,n))
        for i in range(n-1):
            # creamos la columna que contiene solo a los distintos a cero.
            # tenemos n-1 valores distintos a cero
            columna = np.full(n - i, -1)
            # el primero valor es 1
            columna[0] = 1
            # remplazamos la i-ésima columna por la columna que creamos.
            Matriz[i:, i] = columna
        # cambiamos la última columna por un arreglo de puros 1's.
        Matriz[:, n-1] = np.full(n, 1)
        return MatrizCuadrada(Matriz) #objeto Matriz Cuadrada.

# aqui guardaremos la condición para cada caso.
cond = np.zeros(5)
# vamos a usar matrices de tamaño: 5, 10, 15, 25, 30
tamaños = [5, 10, 15, 25, 30]
for i, n in enumerate(tamaños):
    print("\nPara la matriz A de tamaño n = {}".format(n))
    M = crearMatrizEj28(n)
    # obtenemos la (estimación de la) condición de M.
    cond[i] = M.cond
    # b es un vector aleatorio con entradas entre 0 y 100
    # b es de tamaño n
    b = np.random.randint(0, 11, size = n)
    # creamos el Sistema Lineal
    S = SistemaLineal(M, b)
    print("Dado el vector aleatorio\nb = {}".format(b))
    # obtemeos la solución del sistema por eliminación gaussiana con pivoteo paricial.
    X = S.porGauss(pivoteo = 'parcial')
    print("La solución del Sistema Lineal es: {}".format(X))
    # Este es el resultado del inciso C) del ejercicio.
    # obtemeos la factorización LU
    L, U, _, _ = M.factorizarLU(pivoteo = 'total')
    print('La factorización LU de A por pivoteo total es la siguiente:')
    print("L = {}, \nU = {}".format(L, U))

print('\nLos número de condición de la matriz de tamaño n son: ')
for i in range(5):
    print('n = {}: cond(A) = {}'.format(tamaños[i], cond[i]))
print('La condición va aumentando de forma lineal, como lo suguiere la gráfica siguiente: ')
# graficamos los resultados de la condición.
f, ax = plt.subplots()
ax.plot(tamaños, cond, marker = 'o')
ax.set_xlabel('Tamaño de la Matriz')
ax.set_ylabel('Número de Condición')
f.suptitle('Cond(A) vs A.size')
plt.show()
