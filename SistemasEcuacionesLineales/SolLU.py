from Sustitucion import *

def resolverConLU(A, b):
    L, U = A.factorizarLU()
    # resuelve LY = b
    Y = sustDelante(L.A, b)
    # resuelve UX = Y
    X = sustAtras(U.A, Y)
    return X

def resolverConLUParcial():
    pass

def resolverConLUTotal():
    pass