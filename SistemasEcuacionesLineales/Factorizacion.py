# FACTORIZACIONES DE MATRICES
# LU (estándar, pivoteo parcial, pivoteo total)
# Cholesky (A = LL^t)

# Nota: Para matrices de permutación donde solo dos de sus columnas/renglones cambiaron respecto a la
# identidad

import numpy as np
from Pivoteo import *

def factLU(M):
    """Obtiene la factorización LU (método estándar) de una matriz, donde L 
    es una matriz triangular inferior y U una matriz tringular superior.
    
    Args:
        M(np.ndarray):
            M representa una matriz cuadrada.
        
    returns:
        L(np.ndarray):
            matriz triangular inferior de la descomposición LU.
        U(np.ndarray):
            matriz triangular superior de la descomposción LU.
            
    raises:
        Genera una expeción si en algún momento la diagonal tiene un elemento igual a cero."""
    # copiamos la matriz del argumento.
    A = np.array(M, dtype = np.float64)
    # obtenemos el tamaño de la matriz, que asumimos cuadrada.
    n, _ = A.shape
    # inicializamos L como I_n.
    L = np.identity(n)
    # para cada renglón, los componentes de L debajo de la diagonal principal serán los factores de
    # pivoteo para cada fila. Es decir a_ij / a_ii, para j > i.
    # A la vamos a pivotear para cada i hasta la penúltima columna.
    for i in range(n-1):
        # estamos en la i-ésima columna.
        # como no hacemos pivoteo, si algo en la diagonal es cero, entonces falla la factorización.
        if A[i,i] == 0:
            raise Exception("La factorización LU falla.")
        for j in range(i+1, n):
            # para cada renglón debajo de la diagonal principal, aplicamos el pivoteo
            fact = A[j,i] / A[i,i]
            # cambiamos el renglón Rj por Rj - f*Ri
            A[j] = A[j] - fact*A[i]
            # añadimos el factor a la identidad
            L[j, i] = fact
    # para este paso, A ya es U. Sabemos que M^-1 es precisamente como cosntruimos L. 
    return L, A

def factLUpivpar(M):
    """Obtiene la factorización A=LU, donde U es una matriz triangular superior. También
    calcula P tal que PL es una matriz triangular inferior. Se realiza por medio
    de pivoteo parcial, es decir, el pivote es el elemento de mayor valor absoluto en cada
    columna que este de la diagonal hacia abajo.

    Args:
        M(np.ndarray):
            M representa una matriz cuadrada.
        
    returns:
        L(np.ndarray):
            matriz.
        U(np.ndarray):
            matriz triangular superior de la descomposción LU.
        P(np.ndarray):
            matriz de permutación tal que PL es triangular inferior.
            
    raises:
        Genera una expeción si en algún momento la diagonal tiene un elemento igual a cero.
    
    """
    # copiamos la matriz como un array de tipo np.float64.
    A = np.array(M, dtype = np.float64)
    # obtenemos el tamaño de la matriz.
    n, _ = A.shape
    # incializamos L y P como identidades.
    L = np.identity(n)
    P = np.identity(n)
    # para cada columna hasta la penúltima realizamos el pivoteo.
    for j in range(n-1):
        # Pj es la matriz de permutación para el acomodar al pivote.
        # PjInv es la inversa de Pj.
        PjInv = np.identity(n)
        # Mj es la matriz de eliminación de la columna j.
        # MjInv es la inversa de Mj.
        MjInv = np.identity(n)
        # Entontramos el pivote por medio del pivoteo parcial
        p = pivoteoParcial(A, j)
        # si el pivote es cero, no hay más que hacer.
        if A[p, j] == 0:
            raise Exception("La matriz es singular.")
        elif p != j:
            # intercambiamos renglones.
            A[[j, p]] = A[[p, j]]
            # la matriz de permutación cambia para reflejar el intercambio.
            PjInv[[j, p]] = PjInv[[p, j]]
        # modificamos los renglones debajo de la digonal.
        for i in range(j+1, n):
            # obtenemos el factor de eliminación.
            fact = A[i, j] / A[j, j]
            # Recordemos que la inversa de la matriz de eliminación guarda estos factores.
            MjInv[i, j] = fact
            # modificamos los renglones para hacer ceros debajo de la diagonal.
            A[i] = A[i] - fact*A[j]
        # una vez calculada la matriz inversa de eliminación, procedemos a aplicarle
        # la matriz de permutación.
        if p != j:
            MjInv[[j, p]] = MjInv[[p, j]]
        # realizamos las multipliaciones 
        # L = P_1^T * M1^-1 * ... * P_{k-1}^T * M_{k-1}^-1
        L = np.dot(L, MjInv)
        # P = P_{k-1} * ... * P_1
        P = np.dot(PjInv, P)
    return L, A, P
    
def factLUpivtot(M):
    """Obtiene una factorización A=LU. Además se obtienen P y Q tales que PL es triangular inferior
    y UQ es triangular superior. El algoritmo utiliza pivoteo total para realizar la eliminación.
    
    Args:
        M(np.ndarray):
            M representa una matriz cuadrada.
        
    returns:
        L(np.ndarray):
            matriz.
        U(np.ndarray):
            matriz.
        P(np.ndarray):
            matriz de permutación tal que PL es triangular inferior.
        Q(np.ndarray):
            matriz de permutación tal que UQ es triangular superior."""
    # hacemos una copia de la matriz. Esta es la que modificaremos y no la original.
    A = np.array(M, dtype = np.float64)
    n, _ = A.shape
    # Inicializamos P, L, Q donde:
    # P = P_{k-1} * ... * P_1
    # Q = Q_{n-1}^t * ... * Q_1^t, donde Q_j es la matriz de permutación de la columna.
    # L = P_1^t * M_1^-1 * ... * P_{n-1}^t * M_{n-1}^-1, donde M_j es la matriz de eliminación.
    P, L, Q = np.identity(n), np.identity(n), np.identity(n)
    # para cada columna hasta la penúltima realizamos el pivoteo.
    for j in range(n-1):
        # PjInv es la inversa de Pj.
        PjInv = np.identity(n)
        # MjInv es la inversa de Mj.
        MjInv = np.identity(n)
        # QjInv es la inversa de Qj.
        QjInv = np.identity(n)
        # Entontramos el pivote por medio del pivoteo total.
        p, q = pivoteoTotal(A, j)
        # si el pivote es cero, no hay más que hacer.
        if A[p, q] == 0:
            raise Exception("La matriz es singular.")
        # si el pivote está por debajo de la diagonal.
        elif p != j:
            # intercambiamos renglones.
            A[[j, p]] = A[[p, j]]
            # la matriz de permutación cambia para reflejar el intercambio.
            PjInv[[j, p]] = PjInv[[p, j]]
        # si el pivote está a la derecha de la columna en la que estamos.
        if q != j:
            # intercambiar columnas
            A[:, [q, j]] = A[:, [j, q]]
            # la matriz de permutación cambia para reflejar el cambio.
            QjInv[:, [q, j]] = QjInv[:, [j, q]]
        # modificamos los renglones debajo de la digonal.
        for i in range(j+1, n):
            # obtenemos el factor de eliminación.
            fact = A[i, j] / A[j, j]
            # Recordemos que la inversa de la matriz de eliminación guarda estos factores.
            MjInv[i, j] = fact
            # modificamos los renglones para hacer ceros debajo de la diagonal.
            A[i] = A[i] - fact*A[j]
        # una vez calculada la matriz inversa de eliminación, procedemos a aplicarle
        # la matriz de permutación.
        if p != j:
            # Aplicamos P^t * Mj^-1
            MjInv[[j, p]] = MjInv[[p, j]]
        # realizamos las multipliaciones 
        # L = P_1^T * M1^-1 * ... * P_{k-1}^T * M_{k-1}^-1
        L = np.dot(L, MjInv)
        # P = P_{k-1} * ... * P_1
        P = np.dot(PjInv, P)
        # Q = Q_{n-1}^t * ... * Q_1^t
        Q = np.dot(QjInv, Q)
    # finalmente, el algoritmo nos deja con A*Q_1*...Q_{n-1} = LU. Despejamos A para obtener la U adecuada.
    U = np.dot(A, Q)
    # listo!
    return L, U, P, Q

def factCholesky(A):
    """Obtención de la factorización A=LL^t. Suponemos que A es definida positiva y simétrica.
    
    Args:
        A(np.ndarray):
            Matriz definida positiva y simétrica.
        
    Returns:
        L(np.ndarray):
            Matriz triangular inferior tal que A=LL^t.
    """
    n, _ = A.shape
    L = np.zeros((n,n))
    # recorremos columnas
    for i in range(n):
        for k in range(n):
            # dejamos en cero a los elementos arriba de la diagonal.
            if i > k:
                pass
            # solo implemento las fórumlas que Iván nos presentó.
            # si i = k:
            # l_{kk} = ( a_{kk} - \sum_{j=0}^{k-1} (l_{kj}^2) )^(1/2)
            # si i < k:
            # l_{ki} = (a_{ki}-\sum_{j=0}^{i-1} l_{ij}*l_{kj}) / l_{ii}
            # recordando que k = 0,1...,n-1 e i = 0,1,...,n-1

            # establecemos los elementos debajo de la diagonal.
            elif k > i:
                suma = 0.0
                for j in range(i):
                    suma += L[i, j]*L[k, j]
                L[k, i] = (A[k, i] - suma)/L[i, i]
            # establecemos los elementos de la diagonal.
            else:
                suma = 0.0
                for j in range(k):
                    suma+= np.square(L[k, j])
                L[k, k] = np.sqrt(A[k, k] - suma)
    return L

def factCholeskyDiag(A):
    """Obtención de la factorización A=LDL^t. Suponemos que A es definida positiva y simétrica.
    
    Args:
        A(np.ndarray):
            Matriz definida positiva y simétrica.
        
    Returns:
        L(np.ndarray):
            Matriz triangular inferior tal que A=LDL^t.
        D(np.ndarray):
            Matriz diagonal.
    """
    # Dado A = LL^t la factorización de Cholesky, obtenemos una matriz diagonal tal que A = L'DL'^t
    # donde las entradas de D son los cuadrados de las diagonales de la L original. 
    # Así podemos descomponer a D como D^(1/2) tal que L=L'D^(1/2). Para obtener esta L', obtenemos la
    # inversa de D^(1/2) --- que consiste en los recíprocos de D^(1/2) --- y obtener L[D^(1/2)]^(-1).
    # Este producto nos dara las columnas de L divididas entre la respectiva diagonal de L.
    
    # Obtenemos la factorización de Cholesky.
    Lch = factCholesky(A)
    n, _ = A.shape
    # Inicializamos la L'
    L = np.zeros((n,n))
    # Incializamos D
    D = np.zeros((n,n))
    # llenamos cada entrada de la diagonal de D
    for i in range(n):
        # Por como la obtuvimos, las entradas son solo los cuadrados de la L.
        D[i,i] = np.square(Lch[i, i])
    # columnas
    for j in range(n):
            # renglones
        for i in range(n):
            # L' tendrá 1 en su diagonal.
            if i == j:
                L[i,i] = 1
            # la columna j es la misma columna J de L dividida entre la entrada de la diagonal
            # de dicha columna.
            elif i > j:
                L[i,j] = Lch[i, j] / Lch[j, j]
    return L, D




    

    

    

if __name__ == "__main__":

    A = [[2,-1,0],
        [-1,2,-1],
        [0,-1,2]]
    A = np.array(A, dtype=np.float64)
    L, D = factCholeskyDiag(A)
    print(np.dot(L, np.dot(D, L.T)))


   


    
