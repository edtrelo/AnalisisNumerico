# Ejercicio 29
# Autor: Edgar Armando Trejo López
from SistemasLineales import *
import numpy as np
from Normas import normaP
import numpy as np
import matplotlib.pyplot as plt
# creamos una matriz llena de unos.
# no hace falta que especifiquemos que A sea de tipo float
# pues al ponerla de argumento del sistema lineal
# desde ahí de hace el cambio de tipo de dato.
# en otro caso, A[0,0] = epsilon no dará que A[0,0] = 0 :((
A = np.full((2,2), 1)
# creamos el vector b (sin el épsilon aún.)
b = np.array([1, 2])
# establecemos el sistema lineal.
S = SistemaLineal(A, b)
# aquí guardaremos los errores absolutos respecto a la solución original.
eabs = np.zeros(10, dtype = np.float64)
solreal = np.array([1, 1])
# k = 1,...,10
for k in range(1, 11):
    epsilon = 10**(-2*k) 
    # editamos la entrada A_{00} del sistema:
    S.Mat[0, 0] = epsilon
    # editamos la primer entrada de b del sistema:
    S.vec[0] = 1 + epsilon
    # resolvemos el sistema por eliminación gaussiana estándar.
    print("\nPara k = {}, el valor de epsilon es  \u03B5 = {}".format(k, epsilon))
    X = S.porGauss()
    print("La solución del sistema es: X = {}".format(X))
    eabs[k-1] = normaP(solreal - X)/normaP(solreal)

print("\nPara ver cómo se comporta la solución del nuevo sistema, calculamos su error absoluto ")
print("respecto a la solución original x = {}. La gráfica siguiente sugiere que conforme k va aumentando".format(solreal))
print("(y por tanto \u03B5 disminuye), el error absoluto aumenta. Es decir, la solución se aleja")
print("de la dada por el problema original.")

f, ax = plt.subplots()
ax.plot(range(1, 11), eabs, marker = 'o', color = 'k')
f.suptitle(r'$k$ vs Error absoluto para el Sistema con $A_{00}=10^{-2k}$.')
ax.set_xlabel(r"$k$, tal que $\epsilon = 10^{-2k}$")
ax.set_ylabel(r"Error absoluto de $X$ respecto a $x = [1,\ 1]^T$")
plt.show()


