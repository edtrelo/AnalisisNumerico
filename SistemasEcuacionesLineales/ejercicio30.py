# Ejercicio 30
# Autor: Edgar Armando Trejo López
import numpy as np
import SistemasLineales as SL 
import Pentatriangular as PT
import Sustitucion as St
from time import time
# creamos la matriz del problema
# creamos la diagonal, por el momento con solo 6's.
d = np.full(100, 6, dtype = np.float64)
# la primer  entrada de la diagonal es 9.
d[0] = 9
# la penúltima entrada de la diagonal es 5.
d[98] = 5
# la última entrada de la diagonal es 1.
d[99] = 1
# sea dsup la diagonal que va por encima de la diagonal principal.
# dsup es de tamaño n-1. La llenamos de puros -4's.
dsup = np.full(99, -4,  dtype = np.float64)
# la última posición de c es un -2.
dsup[98] = -2
# sea dinf la diagonal por debajo de la diagonal principal.
# dinf es igual a dsup.
dinf = np.copy(dsup)
# creamos la matriz A con puros 0's y de tamaño 100 x 100.
A = np.zeros((100,100))
# hacemos que la diagonal de A sea d.
for i in range(100):
    A[i,i] = d[i]
# cambiamos los valores de las diagonales superiores e inferiores de A.
for i in range(99):
    A[i, i+1] = dsup[i]
    A[i+1, i] = dinf[i]
# las diagonales encima y debajo de dsup y dinf, respectivamente, solo tienen 1's.
for i in range(98):
    A[i, i+2] = 1
    A[i+2, i] = 1
# hemos terminado de formar A.
print('A = {}'.format(A))
# creamos el vector b
b = np.full(100, 1,  dtype = np.float64)
# aqui guardaremos los tiempos de ejeccución de las rutinas.
tiempos = [0, 0, 0]
# aqui guardaremos los errores cuadraticos medios respecto a la solución que da numpy.
ecm = [0, 0, 0]
# la solución de numpy es
Xreal = np.linalg.solve(A, b)
# ---------------------------------------------- INICIO ------------------------------------------------#
# inciso a)
# establecemos el sistema lineal
Sistema = SL.SistemaLineal(A, b)
# resolvemos por LU
Xlu, t = Sistema.porLU(medir = True)
print("\nLa solución obtenida mediante la factorización LU es:\n{}".format(Xlu))
# agregamos los datos que nos arroja esta solución.
tiempos[0] = t
ecm[0] = ((Xreal - Xlu)**2).mean()

# inciso b)
# usamos la función solvePentaTrian del archivo PT.
# ya considera que las diagonales encima y debajo de la dsup y dinf son 1's.
# el primer argumento es la diagonal inferior, el segundo la diagonal principal,
# el tercero la diagonal superior y el cuarto es b
start = time()
Xbanda = PT.solvePentaTrian(dinf, d, dsup, b)
t = time() - start
print("\nLa solución obtenida mediante la la rutina tipo banda es:\n{}".format(Xbanda))
# agregamos los datos que nos arroja esta solución.
tiempos[1] = t
ecm[1] = ((Xreal - Xbanda)**2).mean()

# inciso c)
# formamos a R
R = np.zeros((100, 100), dtype = np.float64)
# veamos que la diagonal de R es puros 1's y un 2 al inicio.
dR = np.full(100, 1, dtype = np.float64)
dR[0] = 2
# Empezamos a llenar R
# llenamos la diagonal
for i in range(100):
    R[i, i] = dR[i]
# llenamos la diagonal superior 
for i in range(99):
    R[i, i+1] = -2
for i in range(98):
    R[i, i+2] = 1
print('\nR es la matriz \n{}'.format(R))
# veamos que A es RR^T
RRt = np.dot(R, R.T)
print("\n¿Es verdad que A = RR^T?: {}".format(np.array_equal(A, RRt)))

def solveRRt(Y, medir = True):
    """Resuelve Ax=Y, donde A es la matriz que se creó al inicio del script y Y
    es un vector cualquiera. La solución se hace mediante la factorización A=RRt
    donde R es triangular superior.
    
    Returns:
        np.ndarray
        
        si medir == True:
            regresa el tiempo que tardó la solución en segundos."""
    # usamos la R de afuera
    global R
    start = time()
    # sea z = R^tx, entonces solucionamos Rz=Y por sustituación hacia atrás
    Z = St.sustAtras(R, Y)
    # solucionamos ahora R^tx = z, donde R^t es triangular inferior, por sustitucion ahcia adelante.
    X = St.sustDelante(R.T, Z)
    t = time() - start
    if medir:
        return X, t
    else:
        return X

Xfact, t = solveRRt(b, medir = True)
print("La solución mediante la factorización RR^t de A es:\n{}".format(Xfact))
# agregamos los datos que nos arroja esta solución.
tiempos[2] = t
ecm[2] = ((Xreal - Xfact)**2).mean()
# nombres de los métodos
met = ["por LU", "por Banda", "por A = RRt"]
# calculamos qué método fue el que tardó mayor tiempo.
tiempoMayor = max(tiempos)
ind = tiempos.index(tiempoMayor)
print("\nEl método más tardado fue el de solución {},".format(met[ind]))
print("tardó {} segundos.".format(tiempos[ind]))
# calculamos qué método fue el que tardó menor tiempo.
tiempoMenor = min(tiempos)
ind = tiempos.index(tiempoMenor)
print("\nEl método menos tardado fue el de solución {},".format(met[ind]))
print("tardó {} segundos.".format(tiempos[ind]))
# calculamos qué método que está más cerca de la solución.
menorECM = min(ecm)
ind = ecm.index(menorECM)
print("\nEl método más acertado fue el de solución {}.".format(met[ind]))