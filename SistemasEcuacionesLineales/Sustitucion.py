import numpy as np

# ¿Por qué se trabaja con una matriz aumentada en estos algoritmos? Pues en el algoritmo de
# sustGauss, se trabaja con una matriz aumentada, entonces decidí que estos algoritmos también
# lo hagan de la misma manera para poder usarlos sin mayor ajuste en tal función.

def sustDelante(A, v):
  """Dada una matriz L triangular inferior y b un vector, se resuelve el sistema Lx = b
  por medio de la sustitución hacia delante.
  
  Args:
    L(np.ndarray):
      Matriz triangular inferior.

    b(np.ndarray):
      vector.

  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si en la diagonal de L hay un cero, i.e., L es singular."""

  L, b = np.copy(A), np.copy(v)
  n, _ = L.shape
  # creamos el vector solución
  X = np.zeros(n)
  # hay que notar que b[k] = A[k, n], donde b es el vector independiente.
  # b es la última columna pues.
  for j in range(n):
    # si la matriz tiene una entrada cero en la diagonal, 
    # entonces no es invertible y el sistema no tiene solución única.
    if L[j, j] == 0:
      raise Exception("La matriz es singular. El sistema no tiene solución")
    # efectuamos la división
    X[j] = b[j] / L[j, j]
    for i in range(j+1, n):
      # actualizamos los valores de b, sabiendo ya el valor de xj en la ecuación.
      b[i] = b[i] - L[i, j] * X[j]
  return X

def sustAtras(A, v):
  """Dada una matriz U triangular superior y b un vector, se resuelve el sistema Ux = b
  por medio de la sustitución hacia delante.
  
  Args:
    U(np.ndarray):
      Matriz triangular superior.

    b(np.ndarray):
      vector
    
  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si en la diagonal de U hay un cero, i.e., U es singular."""

  U, b = np.copy(A), np.copy(v)
  n, _ = U.shape
  # creamos el vector solución
  X = np.zeros(n)
  for j in range(n-1, -1, -1): # vamos de atrás para delante. 
    # Recoradar que el segundo argumento de range es no inclusivo.

    # si la matriz tiene una entrada cero en la diagonal, 
    # entonces no es invertible y el sistema no tiene solución única.
    # hay que notar que b[k] = A[k, n], donde b es el vector independiente.
    if U[j, j] == 0:
      raise Exception("La matriz es singular. El sistema no tiene solución")
    # efectuamos la división
    X[j] = b[j] / U[j, j]
    for i in range(0, j):
      # actualizamos los valores de b, sabiendo ya el valor de xj en la ecuación.
      b[i] =  b[i] - U[i, j] * X[j]
  return X