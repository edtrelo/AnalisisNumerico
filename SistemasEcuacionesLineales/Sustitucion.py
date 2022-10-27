import numpy as np

def sustDelante(L, v):
  """Dada una matriz L diagonal inferior y v un vector, se resuelve el sistema Lx = v
  por medio de la sustitución hacia delante.
  
  Args:
    L(list of lists):
      Se asume que L representa una matriz diagonal inferior de nxn.
      
    v(list):
      Vector.
    
  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si en la diagonal de L hay un cero, i.e., L es singular."""

  # hacemos copias de las matrices.
  A = np.array(L, dtype=np.float64)
  b = np.array(v, dtype=np.float64)
  n, _ = A.shape
  # creamos el vector solución
  X = np.zeros(n)

  for j in range(n):
    # si la matriz tiene una entrada cero en la diagonal, 
    # entonces no es invertible y el sistema no tiene solución única.
    if A[j, j] == 0:
      raise Exception("La matriz es singular. El sistema no tiene solución")
    # efectuamos la división
    X[j] = b[j] / A[j, j]
    for i in range(j+1, n):
      # actualizamos los valores de b, sabiendo ya el valor de xj en la ecuación.
      b[i] = b[i] - A[i, j] * X[j]

  return X
