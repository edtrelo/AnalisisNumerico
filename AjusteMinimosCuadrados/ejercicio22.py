# ejercicio 22

import numpy as np

def cond(A):
    """Calcula la condición de una matriz rectangular respecto a la 
    norma 2 para matrices.
    
    Args:
        A(np.ndarray): Una Matriz

    Returns:
        float: la condición de la matriz.
    """
    def _pseudoInversa(A):
        """Obtiene la pseudo-inversa de la matriz A."""
        AtA = np.dot(A.T, A)
        # obtenemos la inversa de (A^tA)
        invAtA = np.linalg.inv(AtA)
        pseudoInversa = np.dot(invAtA, A.T)
        return pseudoInversa
    # obtenemos la pseudo inversa de A
    Amas = _pseudoInversa(A)
    return np.linalg.norm(A, ord = 2) * np.linalg.norm(Amas, ord = 2)

# inciso a) -------------------------------
# la matriz del sistema (1) es:
A = [[1,1],
    [-1,0],
    [0,1],
    [1,0]]
# la pasamos a un array de numpy
A = np.array(A)
# calculamos su condición:
condi = cond(A)
print('La condición de la matriz del sistema (1) es: {}'.format(condi))
if condi < 10**6:
    print('La matriz es bien condicionada.')
else:
    print('La matriz es mal condicionada.')
# inciso b) ---------------------------------
# A mi me tocó hacer los ejercicios 17, 18, 19. En cada uno, cuando los calcule a mano
# obtuve que xi = [-0.4, -0.8], por lo que cada bi va a ser igual.
X = np.array([-0.4, -0.8])
b_aprox = np.dot(A, X)
# recordemos que x * y = cos 0 |x| |y| donde 0 es el ángulo que forma x y y.
# -> cos 0  = (x*y)/(|x||y|) -> 1/cos 0 = (|x||y|)/(x*y)
b = np.array([-1, 2, -1, 1])
dot_product = np.dot(b, b_aprox)
norma_b = np.sqrt(np.dot(b, b))
norma_b_approx = np.sqrt(np.dot(b_aprox, b_aprox))
# calculamos 1/cos 0
reciproco_cos = (norma_b_approx*norma_b)/dot_product
print('\nPara cada método obtuve que b_i = {}'.format(b_aprox))
print('El recíproco del coseno del ángulo formado entre b y cada b_i obtenida es: {}'.format(reciproco_cos))