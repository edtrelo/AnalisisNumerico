def normaP(v, p = 2):
    """Calcula la norma p del vector v. Por default, se calcula la 
    norma euclidiana.
    
    Args:
        v(np.ndarray)"""
    # calculamos la potencia p de los valores absolutos.
    vecp = np.absolute(v)**p
    # calculamos la suma de lo calculado arriba.
    suma = np.sum(vecp)
    # regresamos la raiz p-ésima de la suma
    return suma**(1/p)

def normaInf(v):
    """Calcula la norma infinito del vector v.
    
    Args:
        v(np.ndarray)"""
    return max(v, key = abs)