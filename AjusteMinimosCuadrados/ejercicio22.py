# ejercicio 22

import numpy as np

def normaUno(A):
        """Calcula la norma 1 de la matriz.

        Returns:
            float"""
        # sobre columnas
        # aquí guardaremos las sumas de las columnas.
        m, n = A.shape
        sumas = np.zeros(n)
        for j in range(n):
            # inicializamos la suma en la columna j
            sum_colj = 0.0
            for i in range(m):
                # sumamos el valor absolto de A[i,j]
                sum_colj += abs(A[i, j])
            # agregamos la suma obtenida al array con los resultados.
            sumas[j] = sum_colj
        # el maximo de las sumas de las columnas es la norma Uno
        return max(sumas)

def cond(A):
    """Calcula la condición de una matriz rectangular."""

    def pseudoInversa(A):
        """Obtiene la pseudo inversa de la matriz A."""

        AtA = np.dot(A.T, A)
        invAtA = np.linalg.inv(AtA)
        pseudoInversa = np.dot(invAtA, A.T)

        return pseudoInversa

    return normaUno(A) * normaUno(pseudoInversa(A))


A = [[1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [-1, 1, 0],
    [-1, 0, 1],
    [0, -1, 1]]

A = np.array(A)

print(cond(A))
print(normaUno(A))

A = np.array([[1, 1],
    [-1, 0],
    [0, 1],
    [1, 0]])

b = np.array([-1, 2, -1, 1])

x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))

print(x)