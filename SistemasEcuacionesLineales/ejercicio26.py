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
# preparamos un array para imprimir el inciso según el número de ejrcicio.
incisos = ['a)', 'b)', 'c)']
# en este array guardaremos los tiempos.
# cada entrada tendrá el promedio de los tiempos de cada método.
tiempos = np.zeros(3)
# en este array guardaremos los errores cuadráticos medios.
# cada entrada tendrá el promedio de los e.c.m. de cada método.
ecm = np.zeros(3)
for i in range(3):
    A = Sistemas[i].A
    b = Sistemas[i].b
    # Establecemos el Sistema Lineal.
    S = SistemaLineal(A, b)
    # Obtenemos la solución que nos da numpy.
    SR = np.linalg.solve(A, b)
    # solución por LU estándar.
    X, t = S.porLU(medir = True)
    tiempos[0] += t
    # calculamos el error cuadrático medio de la solución.
    ECM = ((SR - X)**2).mean()
    ecm[0] += ECM
    # Imprimimos los resultados.
    print('\nRESULTADOS PARA EL SISTEMA LINEAL DEL INCISO {}'.format(incisos[i]))
    print("\n\tLa solución (real) de numpy es : {}".format(SR))
    print("\n\tPara la solución dada por Factorización LU estándar: ")
    print("\tLa solución que arroja es: {}".format(X))
    print("\tSe comete un error cuadrático medio de: {}".format(ECM))
    print("\tLa solución se obtuvo en: {} segundos.".format(t))
    # solución por LU con pivoteo parcial.
    X, t = S.porLU(pivoteo = 'parcial', medir = True)
    tiempos[1] += t
    # calculamos el error cuadrático medio de la solución.
    ECM = ((SR - X)**2).mean()
    ecm[1] += ECM
    # Imprimimos los resultados.
    print("\n\tPara la solución dada por Factorización LU con pivoteo parcial: ")
    print("\tLa solución que arroja es: {}".format(X))
    print("\tSe comete un error cuadrático medio de: {}".format(ECM))
    print("\tLa solución se obtuvo en: {} segundos.".format(t))
    # solución por LU con pivoteo total.
    X, t = S.porLU(pivoteo = 'total', medir = True)
    tiempos[2] += t
    # calculamos el error cuadrático medio de la solución.
    ECM = ((SR - X)**2).mean()
    ecm[1] += ECM
    # Imprimimos los resultados.
    print("\n\tPara la solución dada por Factorización LU con pivoteo total: ")
    print("\tLa solución que arroja es: {}".format(X))
    print("\tSe comete un error cuadrático medio de: {}".format(ECM))
    print("\tLa solución se obtuvo en: {} segundos.".format(t))

tiempos = tiempos / 3 #obtenemos el promedio.
ecm = ecm / 3 #obtenemos el promedio.
# usaremos este array para indicar el método y sus resultado.
metodos = ['estándar', 'parcial', 'total']
# imprimimos los promedios de nuestros resultados.
print("\nRESULTADOS EN GENERAL PARA LOS MÉTODOS DE FACTORIZACIÓN:")
for i in range(3):
    print("\n\tPara la factorización con pivoteo {}:".format(metodos[i]))
    print("\tEl promedio del tiempo de ejecución fue de {} segundos.".format(tiempos[i]))

    print("\tEl promedio del Error Cuadrático Medio cometido fue de {}.".format(ecm[i]))
