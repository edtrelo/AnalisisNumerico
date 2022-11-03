# Ejercicio 26
# Autor: Edgar Armando Trejo López
from SistemasLineales import *
from collections import namedtuple
# Lista que guardará tuplas con las matrices y vectores (A,b) de los sistemas.
Sistemas = [None, None, None]
SL = namedtuple('SL', ' A b')
# Escibimos los sistemas y los agregamos a la lista.
# sistema a)
S = SL( [
    [4, -1, 3],
    [-8, 4, -7],
    [12, 1, 8]
    ], 
    [-8, 19, -19])
# agregamos el primer sistema.
Sistemas[0] = S
# sistema b)
S = SL( [
    [1, 4, -2, 1],
    [-2, -4, -3, 1],
    [1, 16, -17, 9],
    [2, 4, -9, -3]
    ], 
    [3.5, -2.5, 15, 10.5])
# agregamos el segundo sistema.
Sistemas[1] = S
# sistema c)
S = SL( [
    [4, 5, -1, 2, 3],
    [12, 13, 0, 10, 3],
    [-8, -8, 5, -11, 4],
    [16, 18, -7, 20, 4],
    [-4, -9, 31, -31, -1]
    ], 
    [34, 93, -33, 131, -58])
# agregamos el tercer sistema.
Sistemas[2] = S
incisos = ['a)', 'b)', 'c)']
for i in range(3):
    A = Sistemas[i].A
    b = Sistemas[i].b
    # Establecemos el Sistema Lineal.
    S = SistemaLineal(A, b)
    # Obtenemos la solución que nos da numpy.
    SR = np.linalg.solve(A, b)
    # solución por LU estándar.
    X, t = S.porLU(medir = True)
    # calculamos el error cuadrático medio de la solución.
    ECM = ((SR - X)**2).mean()
    # Imprimimos los resultados.
    print('\nRESULTADOS PARA EL SISTEMA LINEAL DEL INCISO {}'.format(incisos[i]))
    print("\nPara la solución dada por Factorización LU estándar: ")
    print("\tSe comete un error cuadrático medio de: {}".format(ECM))
    print("\tLa solución se obtuvo en: {} segundos.".format(t))
    # solución por LU con pivoteo parcial.
    X, t = S.porLU(pivoteo = 'parcial', medir = True)
    # calculamos el error cuadrático medio de la solución.
    ECM = ((SR - X)**2).mean()
    # Imprimimos los resultados.
    print("\nPara la solución dada por Factorización LU con pivoteo parcial: ")
    print("\tSe comete un error cuadrático medio de: {}".format(ECM))
    print("\tLa solución se obtuvo en: {} segundos.".format(t))
    # solución por LU con pivoteo total.
    X, t = S.porLU(pivoteo = 'total', medir = True)
    # calculamos el error cuadrático medio de la solución.
    ECM = ((SR - X)**2).mean()
    # Imprimimos los resultados.
    print("\nPara la solución dada por Factorización LU con pivoteo total: ")
    print("\tSe comete un error cuadrático medio de: {}".format(ECM))
    print("\tLa solución se obtuvo en: {} segundos.".format(t))



