import numpy as np

def sustDelante(L, v):
  """Dada una matriz L triangular inferior y v un vector, se resuelve el sistema Lx = v
  por medio de la sustitución hacia delante.
  
  Args:
    L(list of lists):
      Se asume que L representa una matriz triangular inferior de nxn.
      
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


def sustAtras(U, v):
  """Dada una matriz U triangular superior y v un vector, se resuelve el sistema Ux = v
  por medio de la sustitución hacia delante.
  
  Args:
    U(list of lists):
      Se asume que U representa una matriz triangular superior de nxn.
      
    v(list):
      Vector.
    
  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si en la diagonal de U hay un cero, i.e., U es singular."""

  # hacemos copias de las matrices.
  A = np.array(U, dtype=np.float64)
  b = np.array(v, dtype=np.float64)
  n, _ = A.shape
  # creamos el vector solución
  X = np.zeros(n)

  for j in range(n-1, -1, -1): # vamos de atrás para delante. 
    # Recoradar que el segundo argumento de range es no inclusivo.

    # si la matriz tiene una entrada cero en la diagonal, 
    # entonces no es invertible y el sistema no tiene solución única.
    if A[j, j] == 0:
      raise Exception("La matriz es singular. El sistema no tiene solución")
    # efectuamos la división
    X[j] = b[j] / A[j, j]
    for i in range(0, j):
      # actualizamos los valores de b, sabiendo ya el valor de xj en la ecuación.
      b[i] = b[i] - A[i, j] * X[j]

  return X
