import numpy as np
from Factorizacion import *
from Pivoteo import *
from SistemaLineal import *
from SolLU import *
from Normas import *

class MatrizCuadrada:
    """
    Esta clase sirve para representar a una matriz cuadrada, con el uso de la clase
    np.ndarray almacenando la matriz.

    Creí necesaria hacerla para poder implementar de manera ordenada las funciones
    necesarias para el análisis y ejecución de soluciones a sistemas de ecuaciones
    lineales cuadradas.

    Properties:
        determinant(float):
            El determinante de la matriz.
        size(int):
            El tamaño de la matriz.
        normaUno(float):
            Norma uno de la matriz.
        normaInf(float):
            Norma infinito de la matriz."""

    def __init__(self, A):
        """La matriz A es guardada como un arreglo de numpy con dtype float64."""
        self.A = None
        self.n = None
        try:
            # intentamos convertir el argumento a un arreglo
            self.A = np.array(A, dtype = np.float64)
        except TypeError:
            raise TypeError("El argumento no representa un arreglo n dimensional.")
        # n es el tamaño de la matriz
        esCuad, n = self._verificarCuad()
        if esCuad:
            self.n = n

    def __str__(self):
        """Esto se va a usar cuando se use la función print() con un objeto de esta clase
        como argumento.
        
        Args:
            other(MatrizCuadrada)"""
        # solo usamos la representación que usa numpy para sus arreglos.
        return self.A.__str__()
    
    def __mul__(self, other):
        """Implementa la función A*B para la multipliación de matrices."""
        if type(other) != MatrizCuadrada:
            raise Exception("Los objetos de MatrizCuadrada solo pueden multiplicarse entre ellos.")
        return MatrizCuadrada(np.dot(self.A, other.A))

    def __rmul__(self, other):
        """Implementa la función B*A para la multipliación de matrices.
        
        Args:
            other(MatrizCuadrada)"""
        if type(other) != MatrizCuadrada:
            raise Exception("Los objetos de MatrizCuadrada solo pueden multiplicarse entre ellos.")
        return MatrizCuadrada(np.dot(other.A, self.A))

    def __getitem__(self, index):
        """Establece una manera de usar la sintaxis A[i,j] para nuestra MatrizCuadrada
        de la misma manera que se maneja en numpy."""
        return self.A[index]
    
    def __setitem__(self, index, value):
        """Establece una manera de usar la sintaxis A[i,j] = value de la misma 
        manera que lo hace numpy.
        
        Args:
            value(np.float64)"""
        self.A[index] = value

    def _verificarCuad(self):
        """Verifica si un numpy.ndarray es una matriz cuadrada."""
        n, m = self.A.shape
        if n != m:
            raise ValueError("La matriz no es cuadrada.")
        return True, n
        
    def interRenglones(self, i, j):
        """Intercambia los renglones i y j de la matriz.
        
        Args:
            i,j (int):
                Enteros dentro del rango de los índices de la matriz."""
        self.A[[i, j]] = self.A[[j, i]]

    def obtenerPivoteParcial(self, j, i = None):
        """Obtiene la fila p tal que A[p,j] es el elemento mayor en valor absoluto 
        de la columna j. La búsqueda se hace a partir de la diagonal principal por default.
        
        Args:
            j(int):
                Columna sobre la que se hace el pivoteo.
            i(int o None):
                si i es None, i se toma como j + 1. Si i es otro entero, la búsqueda del
                pivote se hace sobre ese renglón.
                
        Returns:
            p(int):
                renglón que tiene al elemento de mayor valor absoluto en la columna j
                a partir del renglón i."""
        return pivoteoParcial(self.A, j, i)

    def obtenerPivoteTotal(self, j):
        """Obtiene la fila p y la columna q tal que A'[p,q] es el elemento mayor en 
        valor absoluto de la submatriz A' que se obtiene al considerar solo las filas 
        y columnas j:n-1.
        
        Args:
            j(int):
                Fila y columna desde la que se inicia la búsqueda.
                
        Returns:
            p(int):
                fila donde está el pivote.
            q(int):
                columna donde está el pivote."""
        pivoteoTotal(self.A, j)

    def subMatriz(self, i, j):
        """Crea un objeto MatrizCuadrada con la sub matriz obtenida al eliminar
        la fila i y la columna j de la matriz.
        
        Args:
            i, j (int):
                Fila y columna que va a elimnarse de la matriz.

        Returns:
            Aij(Matriz Cuadrada):
                Submatriz Aij."""
        Aij = np.copy(self.A)
        # eliminamos la fila i.
        Aij = np.delete(Aij, i, axis=0)
        # eliminamos la fila j.
        Aij = np.delete(Aij, j, axis=1)
        # cramos el objeto que vamos a regresar.
        return MatrizCuadrada(Aij)

    def _determinante(self):
        """Calcula el determinante de la matriz.
        
        Returns:
            deter(float)"""
        # caso de una matriz de 1x1: un número real.
        if self.n == 1:
            return self.A[0, 0]
        # determinante de una matriz de 2x2: ad - bc
        elif self.n == 2:
            return self.A[0, 0]*self.A[1, 1] - self.A[0, 1]*self.A[1, 0]
        # determinante de manera recursiva
        else:
            deter = 0.0
            for i in range(self.n):
                # calculamos el signo
                sign = (-1)**i
                # sobre la primer fila hacemos la expanción
                subM = self.subMatriz(0, i)
                deter += sign * self.A[0, i] * subM.determinante
            return deter
        
    @property
    def determinante(self):
        return self._determinante()

    @property
    def size(self):
        return self.n

    @property
    def normaUno(self):
        # sobre columnas
        # aquí guardaremos las sumas de las columnas.
        sumas = np.zeros(self.n)
        for j in range(self.n):
            # inicializamos la suma en la columna j
            sum_colj = 0.0
            for i in range(self.n):
                # sumamos el valor absolto de A[i,j]
                sum_colj += abs(self.A[i, j])
            # agregamos la suma obtenida al array con los resultados.
            sumas[j] = sum_colj
        # el maximo de las sumas de las columnas es la norma Uno
        return max(sumas)

    @property
    def normaInf(self):
        # sobre filas
        sumas = np.zeros(self.n)
        for i in range(self.n):
            sum_rowi = 0.0
            for j in range(self.n):
                sum_rowi += abs(self.A[i, j])
            sumas[i] = sum_rowi
        return max(sumas)

    @staticmethod
    def _normaUnoVect(v):
        """Calcula la norma 1 del vector v."""
        return normaP(v, p = 1)

    @staticmethod
    def _normaInfVect(v):
        """Calcula la norma inf del vector v."""
        return normaInf(v)

    @property
    def T(self):
        """La transpuesta de la matriz."""
        trans = self.A.T
        return MatrizCuadrada(trans)

    def _estimarNormaInv(self):
        """Estimación de la norma inversa usando la Norma 1 de vectores."""
        c = np.random.choice([1, -1], size = self.n)

        # calculamos la transpuesta de A
        At = MatrizCuadrada(self.A.T)
        # Resolvemos AtY = c
        Y = resolverConLU(At.A, c)
        # Resolvemos AZ = Y
        Z = resolverConLU(self.A, Y)

        normY = MatrizCuadrada._normaUnoVect(Y)
        normZ = MatrizCuadrada._normaUnoVect(Z)

        return normZ / normY

    @property
    def cond(self, metodo = None):
        """La condición de la matriz A. Si metodo es None, se regresa una estimación.
        Si method es 'exacto', se calcula la inversa de A y su norma."""
        if metodo is None:
            return self.normaUno * self._estimarNormaInv()
        elif metodo == 'exacto':
            try:
                Ainv = np.linalg.inv(self.A)
                Ainv = MatrizCuadrada(Ainv)
                return A.normaUno * Ainv.normaUno
            except np.LinAlgError:
                # si la matriz es singular, regresamos infinito como condición.
                return np.inf
        else:
            raise Exception("El parámetro para 'metodo' no es válido.")
        
    def factorizarLU(self, pivoteo = None):
        """
        Factoriza la matriz en su manera LU. 
        
        Args:
            pivoteo(str o None):
                El parámetro pivoteo puede ser lo siguiente:
                    -> None: no se hace pivoteo
                    -> 'parcial': se hace pivoteo parcial
                    -> 'total': se hace pivoteo total.
            
        Returns:
            (si pivoteo es None)
            L, U (MatrizCuadrada)
                L: matriz triangular inferior.
                U: matriz triangular superior.
                tal que A = LU.

            (si pivoteo es 'parcial')
            L, U, P (Matriz Cuadrada):
                U: matriz triangualar superior.
                P: matriz de permutacion tal que P*L es triangular inferior.
                L: matriz tal que A=LU"""
        if pivoteo is None:
            L, U = factLU(self.A)
            return MatrizCuadrada(L), MatrizCuadrada(U)
        elif pivoteo == 'parcial':
            L, U, P = factLUpivpar(self.A)
            return MatrizCuadrada(L), MatrizCuadrada(U), MatrizCuadrada(P)
        elif pivoteo == 'total':
            L, U, P, Q = factLUpivtot(self.A)
            return MatrizCuadrada(L), MatrizCuadrada(U), MatrizCuadrada(P), MatrizCuadrada(Q)
        else:
            raise Exception("El parámetro para pivote no es aceptable.")

    def factrizarCholesky(self, tipo = None):
        if self.esCholesky()
            if tipo is None:
                L = factCholesky(self.A)
                return MatrizCuadrada(L)
            elif tipo == "diagonal":
                L, D = factCholeskyDiag(self.A)
                return MatrizCuadrada(L), MatrizCuadrada(D)
            else:
                raise Exception("El argumento tipo no es aceptable.")
        else:
            raise Exception("La Martriz no acepta factorización de Cholesky.")

    def _defPos(self):
        """Verifica si la Matriz Cuadrada es definida positiva.
        
        Returns:
            bool"""
        for i in range(self.size):
            # obtenemos la submatriz
            subM = MatrizCuadrada(self.A[:i+1, :i+1])
            subDet = subM.determinante
            # si alguna submatriz es no positiva, la matriz
            # no es definida positiva
            if subDet <= 0:
                return False
        # todas los determinantes son positivos.
        return True

    def esCholesky(self):
        """verifica si la Matriz Cuadrada tiene factorización de Cholesky.
        
        Returns:
            bool:
                True si A es definida postiva y simétrica."""
        if self.A == self.A.T:
            if self._defPos():
                return True
            else:
                return False
        else:
            return False

if __name__ == "__main__":

    A = [[14,2,2],
        [2,1,-2],
        [2,-2,10]]

    A = MatrizCuadrada(A)

    print(A._defPos())