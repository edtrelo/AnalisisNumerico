import numpy as np

# ¿Por qué se trabaja con una matriz aumentada en estos algoritmos? Pues en el algoritmo de
# sustGauss, se trabaja con una matriz aumentada, entonces decidí que estos algoritmos también
# lo hagan de la misma manera para poder usarlos sin mayor ajuste en tal función.

def sustDelante(L, v = None):
  """Dada una matriz L triangular inferior y v un vector, se resuelve el sistema Lx = v
  por medio de la sustitución hacia delante. El argumento que se espera es la matriz
  aumentada L = L|v de nx(n+1).

  Si v no es None, entonces se crea una matriz aumentada.
  
  Args:
    L(np.ndarray):
      Matriz triangular inferior aumentada con v.

    v(np.ndarray o None)_
      si v es None, la matriz L se considera aumentada. De otra manera,
      se crea la matriz aumentada L|v agregando a v como última columna.

  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si en la diagonal de L hay un cero, i.e., L es singular."""

  n, _ = A.shape
  # creamos el vector solución
  X = np.zeros(n)

  # hay que notar que b[k] = A[k, n], donde b es el vector independiente.
  # b es la última columna pues.
  for j in range(n):
    # si la matriz tiene una entrada cero en la diagonal, 
    # entonces no es invertible y el sistema no tiene solución única.
    if A[j, j] == 0:
      raise Exception("La matriz es singular. El sistema no tiene solución")
    # efectuamos la división
    X[j] = A[j, n] / A[j, j]
    for i in range(j+1, n):
      # actualizamos los valores de b, sabiendo ya el valor de xj en la ecuación.
      A[i, n] = A[i, n] - A[i, j] * X[j]

  return X


def sustAtras(U, v = None):
  """Dada una matriz U triangular superior y v un vector, se resuelve el sistema Ux = v
  por medio de la sustitución hacia delante. El argumento que se espera es la matriz
  aumentada U = U|v de nx(n+1).

  Si v no es None, entonces se crea una matriz aumentada.
  
  Args:
    Uv(np.ndarray):
      Matriz triangular superior aumentada con v.

    v(np.ndarray o None)_
      si v es None, la matriz U se considera aumentada. De otra manera,
      se crea la matriz aumentada U|v agregando a v como última columna.

    
  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si en la diagonal de U hay un cero, i.e., U es singular."""

  n, _ = A.shape
  # creamos el vector solución
  X = np.zeros(n)

  for j in range(n-1, -1, -1): # vamos de atrás para delante. 
    # Recoradar que el segundo argumento de range es no inclusivo.

    # si la matriz tiene una entrada cero en la diagonal, 
    # entonces no es invertible y el sistema no tiene solución única.

    # hay que notar que b[k] = A[k, n], donde b es el vector independiente.
    if A[j, j] == 0:
      raise Exception("La matriz es singular. El sistema no tiene solución")
    # efectuamos la división
    X[j] = A[j, n] / A[j, j]
    for i in range(0, j):
      # actualizamos los valores de b, sabiendo ya el valor de xj en la ecuación.
      b[i] =  b[i] - A[i, j] * X[j]

  return X