import numpy as np
from Sustitucion import sustAtras

def primerValorDistinto(a, valor):
    """Obtiene el índice del primer elemento distinto a valor buscando en el 
    array a desde la posición index."""
    n = len(a)
    for i in range(n):
        if a[i] != valor:
            return i 
    raise ValueError("Todos los elementos del arreglo a partir del son {}.".format(valor))

def HousholderTransf(a, v):
    """Obtiene Ha:"""
    # Ha = a - 2v (v^t a / v^t v)
    # obtenemos la norma al cuadrado de v, que es v^t * v
    vtv = np.dot(v, v)
    # obtenemos el producto v^t * a
    vta = np.dot(v, a)
    Ha = a - 2*(vta/vtv)*v
    return Ha 

def construirV(a, k):
    # formamos el vector ek
    ek = np.zeros(len(a))
    ek[k] = 1
    # obtenemos alpha
    signo = 1
    # regla del signo
    if a[k] < 0:
        signo = -1
    alpha = -signo*np.linalg.norm(a)
    # definimos v
    v = a - alpha * ek
    return v

def byHouseholder(Matriz, Vector):
    """Resuelve el problema de Mínimos Cuadrados Ax = b, donde A = Matriz y b = Vector, por medio de 
    la factorización QR a través del método de transformaciones de Householder. """
    # creamos copias de los argumentos
    A = np.array(Matriz, dtype = np.float64)
    b = np.array(Vector, dtype = np.float64)
    # obtenemos la forma de la matriz
    m, n = A.shape
    # queremos que hayan más ecuaciones que variables.
    if m < n:
        raise ValueError("La matriz no tiene la forma adecuada.")
    for j in range(n):
        col = A[j:, j]
        # buscamos el primer elemento distinto de cero del subarray.
        try:
            k = j + primerValorDistinto(col, 0)
        except:
            # si alguna columna es cero, la saltamos.
            continue
        a = np.zeros(m)
        a[k:] = A[k:, j]
        # construimos v
        v = construirV(a, k)
        # aplicamos H a las columnas
        for i in range(j, n):
            Ha = HousholderTransf(A[:, i], v)
            A[:, i] = Ha
        # apilciamos H a b
        b = HousholderTransf(b, v)
    print(b)
    # para esta instancia habremos aplicados n matrices de Householder al sistema,
    # transformandolo de la siguiente manera: Ax=b -> Q^tAx=Qb -> R = Q^tb
    # Ahora basta resolver R = Q^tb por sustitución hacia atrás.
    # sabemos que las primeras n filas y n columnas de R son una matriz cuadrada
    # triangular superior, llamos R a esta submatriz y tomamos las primeras 
    # n entradas de Q^tb.
    R, b = A[:n], b[:n]
    # resolvemos por sustitución hacia atrás.
    X = sustAtras(R, b)
    return X

if __name__ == '__main__':
    a = np.array([0,0,1,0.0332, 0.7231, 0.6899])

    A = [[1,1],
    [-1,0],
    [0,1],
    [1,0]]

    A = np.array(A, dtype = np.float64)

    b = np.array([-1, 2, -1, 1])

    X = byHouseholder(A, b)
    print(X)
    print(2/np.sqrt(3))
