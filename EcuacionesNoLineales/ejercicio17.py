# ejercicio 17
# por: edgar armando trejo lópez
import numpy as np
from numpy import linalg as LA

def puntoFijoGeneral(G, x0, k = 100):
    """Resuelve el sistema F(x) = 0 por medio del método de punto fijo
    generalizado, donde G: R^n -> R^n es tal que F(x) = G(x) - x y x0 
    es un vector con una  solución aproximada al sistema. Se hacen k iteraciones.

    No se evaluan condiciones de convergencia.
    
    Args:
        G: lista de funciones de R^n -> R. 
        x0 (iterable): solución aproximada. Su longuitud debe ser igual a la de G.
        k: número máximo de iteraciones."""
    # obtenemos x1
    xk = _evaluar(G, x0)
    # hacemos k iteraciones
    for i in range(1, k):
        # evaluamos G en xk
        xk = _evaluar(G, xk)
    return xk

def _evaluar(G, x):
    """Evalua la función G en x, donde G: R^n -> R^n y x es elemento de R^n.
    
    Args:
        G: lista de funciones de R^n -> R. 
        x0 (iterable): vector de dimensión n."""
    return np.array([G[i](*x) for i in range(len(G))])

# habiendo 'despejado' x1, x2, x3 de las funciones f1, f2, f3 respectivamente
# definimos las g's de las que queremos obtener el punto fijo
g1 = lambda x1, x2, x3: np.sqrt(5 - x2**2 - x3**2 )
g2 = lambda x1, x2, x3: 1 - x1
g3 = lambda x1, x2, x3: 3 - x1
# definimos la función de R^3 -> R^3
G = (g1, g2, g3)
# establecemos el punto inicial
x0 = ((1+np.sqrt(3))/2, (1-np.sqrt(3))/2, np.sqrt(3))
# encontramos la solución
x = puntoFijoGeneral(G, x0)
print('La solución del método del punto fijo para el punto inicial x0 = {}'.format(x0))
print('Es X = {}'.format(x))
