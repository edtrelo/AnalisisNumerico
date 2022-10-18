import numpy as np

def resolverSistema(A, b=None):
  """Resulve el Sistema de Ecuaciones mediante el 
  método de Gauss y la sustitución hacia atrás.
  
  Args:
    A(list of lists):
      Si A es una matriz nxn, entonces se le considera
      como la matriz que contiene solo a los coeficientes
      de las variables del sistema. 
      
      Si A es una matriz nxn+1, entonces se le considera como la matriz aumentada del sistema.
      
    b(list or None):
      si b es None, es porque A es la matriz aumentada. En otro caso, b es el vector independiente.
      
  Returns:
    x(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si el sistema tiene infinitas
    soluciones o no tiene solución."""

  A = np.array(A)
  n, _ = A.shape

  def crearMatrizAumentada(A, b):
    """Crea una matriz aumentada con el vector no
    homogéneo."""
    nonlocal n
    b = np.array(b)
    b = b.reshape(n,1)
    return np.append(A, b, axis=1)

  def obtenerPivote(A, i = 0):
    """Obtiene el renglón p para el cual A[p,i] no es
    cero buscando desde el i renglón."""
    nonlocal n
    for p in range(i, n-1):
      if A[p, i] != 0:
        return p
  
    raise Exception("No existe Solución Única.")

  def intercambiarRenglones(A, i, j):
    """Intercambia los renglones i y j."""
    A[[i, j]] = A[[j, i]]

  # si b es None, entonces asumimos que A ya es la matriz
  # aumentada.
  if b is not None:
    A = crearMatrizAumentada(A, b)

  for i in range(0, n-1):
    # Recorremos las primeras n-1 columnas, pues queremos
    # pivotear de forma que la última fila tenga n-1
    # ceros.
    p = obtenerPivote(A, i)
    if p != i:
      # si el pivote no está en la diagonal, cambiamos su
      # renglón correspondiente para que sí lo este.
      intercambiarRenglones(A, i, p)
    for j in range(i+1, n):
      # Procedemos a hacer cero debajo del pivote en la
      # columna actual.
      m = A[j,i]/A[i,i]  # m es el valor por el que hay
      # que multiplicar la fila pivote para hacer ceros
      A[j] = A[j]-m*A[i]

    print(A,"\n")

  if A[n-1, n-1] == 0:
    # Si resulta que al haber terminado el pivoteado, el
    # coeficiente correspondiente a la variable x_n es 
    # cero entonces no tenemos solución.
    raise Exception("No existe Solución Única.")
  # creamos el vector solución
  x = np.zeros(n)
  # comenzamos con la sustituación hacia atrás.
  x[n-1] = A[n-1, n]/A[n-1, n-1]
  # recorremos desde atrás el resto de variables que nos
  # hacen falta. 
  for i in range(n-2, -1, -1): #el segundo argumento es non-inclusive.
    suma = 0
    for j in range(i+1, n):
      suma += A[i,j]*x[j]
    x[i] =  (A[i, n] - suma)/A[i,i]

  return x
