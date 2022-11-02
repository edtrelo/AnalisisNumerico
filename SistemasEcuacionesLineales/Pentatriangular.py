# SOLUCIÓN A SISTEMAS CON MATRICES EN BANDA

import numpy as np

def solvePentaTrian(A, B, C, Y):
    """Resuelve el sistema Mx=Y, donde M es una matriz pentatriangular. B es la diagonal 
    principal de M, A es la diagonal debajo de la principal, C es la diagonal por encima 
    de la principal, y las entradas de las diagonales por encima y por debajo de A y C 
    son todas 1. M es de nxn. 
    
    Args:
        B(iterable de floats):
            Lista que representa la diagonal de M, de tamaño n.
        A,C(iterable de floats):
            Listas que representan las diagonales que están por debajo y por encima
            de la diagonal principal de M --- respectivamente ---. Ambas son de tamaño
            n-1.
        Y(iterable de floats):
            Lista de tamaño n al que se iguala Ax.
            
    Returns:
        X(np.ndarray):
            Solución al sistema."""
    # obtenemos la 'matriz' triangularizada.
    a, b, c, y = eliminacionPentaTrian(A, B, C, Y)
    # aplicamos la sustitución hacia atrás.
    X = sustPentaTrian(b, c, y)
    return X

def eliminacionPentaTrian(A, B, C, Y):
    """Imita la eliminación gaussiana para la matriz aumentada M|Y donde la diagonal
    principal de M es B, la diagonal inferior es A y la diagonal superior es C. Además
    las entradas debajo de A y encima de C son todos 1.

    M es una matriz de nxn.
    
    Args:
        B(iterable de floats):
            Lista que representa la diagonal de M, de tamaño n.
        A,C(iterable de floats):
            Listas que representan las diagonales que están por debajo y por encima
            de la diagonal principal de M --- respectivamente ---. Ambas son de tamaño
            n-1.
        Y(iterable de floats):
            Lista de tamaño n al que se iguala Ax.
            
    Returns:
        b(np.ndarray):
            Arreglo de tamaño n que representa la diagonal de la matriz aumentada
            después de aplicarle la eliminación gaussiana.
        c(np.ndarray):
            Arreglo de tamaño n-1 que representa la diagonal por encima de la 
            diagonal princiapal en M después de aplicarle la eliminación gaussiana.
        y(np.ndarray):
            Y después de aplicarle la eliminación gaussiana que se aplicó a M."""
    # transformamos las listas a arreglos de numpy
    a = np.array(A, dtype = np.float64)
    b = np.array(B, dtype = np.float64)
    c = np.array(C, dtype = np.float64)
    y = np.array(Y, dtype = np.float64)
    # obtenemos el tamaño de n.
    n = len(b)
    for i in range(n-1):
        # modificamos el primer renglón debajo para hacer ceros debajo de b[i]
        # calculamos el factor de primera fila
        fact = a[i]/b[i]
        b[i+1] = b[i+1] - fact*c[i]
        y[i+1] = y[i+1] - fact*y[i]
        if i < n-2:
            # en el último renglón ya no hay c para modificar.
            c[i+1] = c[i+1] - fact
            # modificamos el segundo renglón debajo para hacer ceros.
            # calculamos el factor de la segunda fila
            fact = 1 / b[i]
            a[i+1] = a[i+1] - fact*c[i]
            b[i+2] = b[i+2] - fact
            y[i+2] = y[i+2] - fact*y[i]
    # terminamos!
    return a,b,c,y

def sustPentaTrian(b, c, y):
    """Realiza sustitución hacia atrás de un sistema Ux=y, donde U es una 
    matriz pentatriangular llevada a su forma triangular superior trás un
    proceso de eliminación gaussiana, además y pasó por el mismo proceso."""
    # obtenemos el tamaño de la matriz.
    n = len(b)
    # incializamos el vector solución.
    X = np.zeros(n)
    # imitamos la sustitución hacia atrás.
    for i in range(n-1, -1, -1):
        # último renglón.
        if i == n-1:
            X[i] = y[i] / b[i]
        # penúltimo renglón.
        elif i == n-2:
            X[i] = (y[i] - c[i]*X[i+1]) / b[i]
        else:
            X[i] = (y[i] - c[i]*X[i+1] - X[i+2]) / b[i]
    return X
        

b = np.full(100, 6, dtype = np.float64)
b[0] = 9
b[98] = 5
b[99] = 1



c = np.full(99, -4,  dtype = np.float64)
c[98] = -2

a = np.copy(c)
Y = np.full(100, 1,  dtype = np.float64)

A = np.zeros((100,100))
for i in range(100):
    A[i,i] = b[i]

for i in range(99):
    A[i, i+1] = c[i]
    A[i+1, i] = a[i]

for i in range(98):
    A[i, i+2] = 1
    A[i+2, i] = 1

print(A)
        
from SolLU import *
Z = solvePentaTrian(a,b,c,Y)
X = resolverConLU(A, Y)






#print('X:', X)

from Factorizacion import factLU

print(Z)
print(X)

