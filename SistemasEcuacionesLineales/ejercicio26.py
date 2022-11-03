# Ejercicio 26
# Autor: Edgar Armando Trejo López
from SistemasLineales import *
# Lista que guardará tuplas con las matrices y vectores (A,b) de los sistemas.
Sistemas = [None, None, None]
# Escibimos los sistemas y los agregamos a la lista.
# sistema a)
S = ( [
    [4, -1, 3],
    [-8, 4, -7],
    [12, 1, 8]
    ], 
    [-8, 19, -19])
# agregamos el primer sistema.
Sistemas[0] = S
# sistema b)
S = ( [
    [1, 4, -2, 1],
    [-2, -4, -3, 1],
    [1, 16, -17, 9],
    [2, 4, -9, -3]
    ], 
    [3.5, -2.5, 15, 10.5])
# agregamos el segundo sistema.
Sistemas[1] = S








# Establecemos el Sistema Lineal.
S = SistemaLineal(A, b)
# Obtenemos la solución que nos da numpy.
SR = np.linalg.solve(A, b)
# solución por LU estándar.
X, t = S.porLU(medir = True)
# calculamos el error cuadrático medio de la solución.
ECM = ((SR - X)**2).mean()
# Imprimimos los resultados.
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



