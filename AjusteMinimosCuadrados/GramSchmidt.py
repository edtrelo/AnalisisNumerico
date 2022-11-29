import numpy as np
from numpy import linalg as LA
from Sust import *

def factGramSchmidt(A):
  """Calcula la factorización QR por medio del método de ortogonalización de 
  Gram-Schmidt. Este código es el de la ayudantía.
  
  Args:
    A(np.ndarray): Matriz."""
  # creamos una copia de A
  Q = np.copy(A)
  m, n = A.shape
  # creamos a R como una matriz del mismo tamaño de columnas que A.
  R = np.zeros((n, n))

  vj = A[:,0]
  R[0,0] = LA.norm(vj)
  Q[:,0] = vj/R[0,0]

  for j in range(1, n):
    for i in range(j-1):
      # subtraemos de las siguientes columnas.
      vj = A[:,j] - np.dot(Q[:,i].T, A[:,j])*Q[:,i]
      # la diagonal de R es la norma de nuestro vj.
      R[j, j] = LA.norm(vj)
      # normalizamos la columna actual.
      Q[:, i] = vj/R[j,j]

  return Q,R

def byGram(Matriz, vector):
  """resuelve el problema de mínimos cuadrados por gram-schmidt."""
  A = np.array(Matriz, dtype = np.float64)
  b = np.array(vector, dtype = np.float64)
  m, n = A.shape
  Q, R = factGramSchmidt(A)
  # Ax = QRx = b-> Rx = Q^t b
  Qtb = np.dot(Q.T, b)
  # ahora queremos la parte cuadrada del sistema
  R = R[:n]
  Qtb = Qtb[:n]
  # solucionamos por sustitucion hacia atrás
  # resolvemos Rx = b con el subsistema cuadrado.
  X = sustAtras(R, b)
  return X
