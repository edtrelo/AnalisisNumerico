# Ejercicio 32
# Autor: Edgar Armando Trejo López

# definimos las matrices
from MatricesCuadradas import *
A = [[2, -1, 0],
    [-1, 2, -1],
    [0, -1, 2]]
A = MatrizCuadrada(A)
B = [[4, 1, 1, 1],
    [1, 3, -1, 1],
    [1, -1, 2, 0],
    [1, 1, 0, 2]]
B = MatrizCuadrada(B)
C = [[4, 1, -1, 0],
    [1, 3, -1, 0],
    [-1, -1, 5, 2],
    [0, 0, 2, 4]]
C = MatrizCuadrada(C)
D = [[6, 2, 1, -1],
    [2, 4, 1, 0],
    [1, 1, 4, -1],
    [-1, 0, -1, 3]]
D = MatrizCuadrada(D)

Matrices = [A, B, C, D]
nombres = ['A', 'B', 'C', 'D']
# ------------------------------------------------- INICIO ------------------------------------------------#
# inciso a)
# factorización A = LLt
for i, M in enumerate(Matrices):
    L = M.factorizarCholesky()
    print("\nPara {}, su factorización LL^t es:".format(nombres[i]))
    print("L es \n{}".format(L))
    print("L^t es \n{}".format(L.T))
    print("El producto LL^t nos da \n{}".format(L*L.T))

#inciso b)
for i, M in enumerate(Matrices):
    L, D= M.factorizarCholesky(tipo = 'diagonal')
    print("\nPara {}, su factorización LL^t es:".format(nombres[i]))
    print("L es \n{}".format(L))
    print("D es \n{}".format(D))
    print("L^t es \n{}".format(L.T))
    print("El producto LDL^t nos da \n{}".format(L*D*L.T))