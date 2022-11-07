import numpy as np
from Sustitucion import sustAtras
from Pivoteo import pivoteoParcial

def elimGauss(M, b = None):
  """Resulve el Sistema de Ecuaciones Mx = b mediante el 
  método de Gauss y la sustitución hacia atrás.
  
  Args:
    M(list of lists):
      Si M es una matriz nxn, entonces se le considera
      como la matriz que contiene solo a los coeficientes
      de las variables del sistema. 
      
      Si M es una matriz nxn+1, entonces se le considera como la matriz aumentada del sistema.
      
    b(list or None):
      si b es None, es porque M es la matriz aumentada. En otro caso, b es el vector independiente.
      
  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si el sistema tiene infinitas
    soluciones o no tiene solución."""

  A = np.array(M, dtype=np.float64)  #nota: si dejo aqui sin argumento dtype, redondea todo a enteros.
  n, _ = A.shape

  # si v es None, entonces asumimos que A ya es la matriz
  # aumentada.
  if b is not None:
    # ahora A es de nx(n+1)
    A = _crearMatrizAumentada(A, b)

  for i in range(0, n-1):
    # Recorremos las primeras n-1 columnas, pues queremos
    # pivotear de forma que la última fila tenga n-1
    # ceros.
    if A[i, i] == 0:
      raise Exception("Se ha detenido el algoritmo.")
    for j in range(i+1, n):
      # Procedemos a hacer cero debajo del pivote en la
      # columna actual.
      m = A[j, i]/A[i, i]  # m es el valor por el que hay
      # que multiplicar la fila pivote para hacer ceros
      A[j] = A[j] - m*A[i]
  if A[n-1, n-1] == 0:
    # Si resulta que al haber terminado el pivoteado, el
    # coeficiente correspondiente a la variable x_n es 
    # cero entonces no tenemos solución.
    raise Exception("No existe Solución Única.")
  # creamos el vector solución
  # creamos el vector solución
  # Nuestra matriz original (con las operaciones elementales aplicadas)
  # se encuentra eliminando la última columna de A.
  # Nuestro vector original (con las operaciones elementales aplicadas) 
  # se encuentra en la última columna de A.
  X = sustAtras(A[:, :n], A[:, n])

  return X

def elimGaussPar(M, b = None):
  """Resulve el Sistema de Ecuaciones Mx = b mediante el 
  método de Gauss y la sustitución hacia atrás. Además, realiza un pivoteo parcial.
  
  Args:
    M(list of lists):
      Si M es una matriz nxn, entonces se le considera
      como la matriz que contiene solo a los coeficientes
      de las variables del sistema. 
      
      Si M es una matriz nxn+1, entonces se le considera como la matriz aumentada del sistema.
      
    b(list or None):
      si b es None, es porque M es la matriz aumentada. En otro caso, b es el vector independiente.
      
  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si el sistema tiene infinitas
    soluciones o no tiene solución."""

  A = np.array(M, dtype=np.float64)  #nota: si dejo aqui sin argumento dtype, redondea todo a enteros.
  n, _ = A.shape

  # si v es None, entonces asumimos que A ya es la matriz
  # aumentada.
  if b is not None:
    # ahora A es de nx(n+1)
    A = _crearMatrizAumentada(A, b)
  for i in range(0, n-1):
    # Recorremos las primeras n-1 columnas, pues queremos
    # pivotear de forma que la última fila tenga n-1
    # ceros.
    p = pivoteoParcial(A, i)
    if A[p, i] == 0:
      raise Exception("El sistema no tiene solución.")
    if p != i:
      # si el pivote no está en la diagonal, cambiamos su
      # renglón correspondiente para que sí lo este.
      _intercambiarRenglones(A, i, p)
    for j in range(i+1, n):
      # Procedemos a hacer cero debajo del pivote en la
      # columna actual.
      m = A[j, i]/A[i, i]  # m es el valor por el que hay
      # que multiplicar la fila pivote para hacer ceros
      A[j] = A[j] - m*A[i]

  if A[n-1, n-1] == 0:
    # Si resulta que al haber terminado el pivoteado, el
    # coeficiente correspondiente a la variable x_n es 
    # cero entonces no tenemos solución.
    raise Exception("No existe Solución Única.")
  # creamos el vector solución
  # Nuestra matriz original (con las operaciones elementales aplicadas)
  # se encuentra eliminando la última columna de A.
  # Nuestro vector original (con las operaciones elementales aplicadas) 
  # se encuentra en la última columna de A.
  X = sustAtras(A[:, :n], A[:, n])

  return X

def _intercambiarRenglones(A, i, j):
    """Intercambia los renglones i y j."""
    A[[i, j]] = A[[j, i]]

def _crearMatrizAumentada(M, v):
    """Crea una matriz aumentada con el vector no
    homogéneo."""
    n, _ = M.shape
    b = np.array(v, dtype = np.float64)
    b = b.reshape(n,1)
    return np.append(M, b, axis=1)
    