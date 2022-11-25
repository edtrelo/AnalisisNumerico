import numpy as np

def sustAtras(A, v):
  """Dada una matriz A triangular superior y v un vector, se resuelve el sistema Ax = v
  por medio de la sustitución hacia delante.
  
  Args:
    A(np.ndarray):
      Matriz triangular superior.

    v(np.ndarray):
      vector
    
  Returns:
    X(np.ndarray):
      vector solución del sistema de ecuaciones.
      
  Raises:
    Genera una excepción si en la diagonal de A hay un cero, i.e., A es singular."""

  U, b = np.copy(A), np.copy(v)
  n, _ = U.shape
  # creamos el vector solución
  X = np.zeros(n)
  for j in range(n-1, -1, -1): # vamos de atrás para delante. 
    # Recoradar que el segundo argumento de range es no inclusivo.

    # si la matriz tiene una entrada cero en la diagonal, 
    # entonces no es invertible y el sistema no tiene solución única.
    if U[j, j] == 0:
      raise Exception("La matriz es singular. El sistema no tiene solución")
    # efectuamos la división
    X[j] = b[j] / U[j, j]
    for i in range(0, j):
      # actualizamos los valores de b, sabiendo ya el valor de xj en la ecuación.
      b[i] =  b[i] - U[i, j] * X[j]
  return X
