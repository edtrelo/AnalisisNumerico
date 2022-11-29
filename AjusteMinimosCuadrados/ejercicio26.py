#ejercicio 26
#edgar armando trejo lópez

from GramSchmidt import *
import numpy as np
from time import time
import matplotlib.pyplot as plt
from Householder import *
from Normales import factCholesky

def matrizHilbert(n):
    """Crea la matriz de Hilbert de tamaño n.
    
    Args:
        n(int): tamaño de la matriz.
        
    Returns:
        np.ndarray: matriz de tamaño nxn."""
    # definimos un espacio para la matriz de hilbert
    H = np.zeros((n, n))
    for i in range(1, n+1):
        for j in range(1, n+1):
            H[i - 1, j - 1] = 1 / (i + j - 1)
    return H

# inciso a) ----------------------------------
medidaOrtGram = np.zeros(13-2)
tiempoGram = np.zeros(13-2)

medidaOrtDoble = np.zeros(13-2)
tiempoDoble = np.zeros(13-2)

for n in range(2, 13):
    H = matrizHilbert(n)
    # Gram-Schmidt clásico
    start = time()
    Q, _ = factGramSchmidt(H)
    matrizMedida = np.eye(n) - np.dot(Q.T, Q)
    # uso la norma 2
    norma = np.linalg.norm(matrizMedida, ord = 2)
    medidaOrtGram[n-2] = - np.log10(norma)
    tiempoGram[n-2] = time() - start
    # Doble Gram-Schmidt
    start = time()
    Q, _ = factGramSchmidt(H)
    QQ, _ = factGramSchmidt(Q)
    matrizMedida = np.eye(n) - np.dot(QQ.T, QQ)
    # uso la norma 2
    norma = np.linalg.norm(matrizMedida, ord = 2)
    medidaOrtDoble[n-2] = - np.log10(norma)
    tiempoDoble[n-2] = time() - start

f, ax = plt.subplots(1, 2)

ax[0].plot(range(2, 13), medidaOrtGram, label = 'Gram-Schmidt Clásico')
ax[0].plot(range(2, 13), medidaOrtDoble, label = 'Doble Gram-Schmidt')
ax[0].legend()
ax[0].set_title('Medida de Ortogonalidad')
ax[0].set_xlabel('Tamaño de la Matriz H (n)')
ax[0].set_ylabel('Medida de ortogonalidad')

ax[1].plot(range(2, 13), tiempoGram, label = 'Tiempos para\nGram-Schmidt')
ax[1].plot(range(2, 13), tiempoDoble, label = 'Tiempos para\nGram-Schmidt Doble')
ax[1].legend()
ax[1].set_title('Tiempo de Ejecución')
ax[1].set_xlabel('Tamaño de la Matriz H (n)')
ax[1].set_ylabel('tiempo (s)')

f.set_size_inches(14, 6)
#plt.show()

print('Resultados del inciso a:\n')
print('La medida de ortogonalidad de los dos métodos fluctúan entre sí.Gram-Schmidt')
print('por lo general tiene menor exactitud aunque de tamaño a tamaño sube y baja ')
print('respecto al doble Gram-Schmidt. El doble Gram-Schmidt tiene una tendencia más')
print('monótona y termina teniendo mejor exactitud que el clásico.')
print('\nPara los tamaños debajo de n = 6, los tiempos también cambian de estar uno arriba')
print('del otro. A partir de n = 6, el tiempo de aplicar el doble Gram-Schmidt sobrepasa')
print('al original dando saltos como de escalón.')
print('\nComo el doble Gram-Schmidt necesita obtener dos Matrices, utiliza el doble de tamaño')
print('que el clásico.')

# inciso b) -------------------------------------------------------------------------------------------
medidaOrtH = np.zeros(13-2)
tiempoH = np.zeros(13-2)

for n in range(2, 13):
    H = matrizHilbert(n)
    # Gram-Schmidt clásico
    start = time()
    Q, _ = factHouseholder(H)
    matrizMedida = np.eye(n) - np.dot(Q.T, Q)
    # uso la norma 2
    norma = np.linalg.norm(matrizMedida, ord = 2)
    medidaOrtH[n-2] = - np.log10(norma)
    tiempoH[n-2] = time() - start


f, ax = plt.subplots(1, 2)

ax[0].plot(range(2, 13), medidaOrtGram, label = 'Gram-Schmidt Clásico')
ax[0].plot(range(2, 13), medidaOrtDoble, label = 'Doble Gram-Schmidt')
ax[0].plot(range(2, 13), medidaOrtH, label = 'Householder')
ax[0].legend()
ax[0].set_title('Medida de Ortogonalidad')
ax[0].set_xlabel('Tamaño de la Matriz H (n)')
ax[0].set_ylabel('Medida de ortogonalidad')

ax[1].plot(range(2, 13), tiempoGram, label = 'Tiempos para\nGram-Schmidt')
ax[1].plot(range(2, 13), tiempoDoble, label = 'Tiempos para\nGram-Schmidt Doble')
ax[1].plot(range(2, 13), tiempoH, label = 'Tiempos para\nHouseholder')
ax[1].legend()
ax[1].set_title('Tiempo de Ejecución')
ax[1].set_xlabel('Tamaño de la Matriz H (n)')
ax[1].set_ylabel('tiempo (s)')

f.set_size_inches(14, 6)

print('\nResultados del inciso b:\n')
print('Householder tiene mucho mejor presición pues presenta una tendencia  tener')
print('15 dígitos de presición que se mantienen más o menos constantes para cada n.')
print('Los tiempos de ejcución de Householder son similares para los otros dos métodos')
print('Aunque varían los intervalos de n donde cierto método se comporta mejor que otro.')
print('Householder requiere de dos matrices para guardar la factorización y una para')
print('realizarla aunque si usaramos la idea de aplicar H a cada columna nos ahorramos espacio.')

# inciso c) --------------------------------------------------------------------------------------
# usando la ortogonalización por ecuaciones normales
medidaOrtNorm = np.zeros(13-2)
tiempoNorm = np.zeros(13-2)

for n in range(2, 13):
    H = matrizHilbert(n)
    start = time()
    # obtemos la L de la factorización de cholesky de H^tH
    L = factCholesky(np.dot(H.T, H))
    Q = np.dot(H, L.T)
    matrizMedida = np.eye(n) - np.dot(Q.T, Q)
    # uso la norma 2
    norma = np.linalg.norm(matrizMedida, ord = 2)
    medidaOrtNorm[n-2] = - np.log10(norma)
    tiempoNorm[n-2] = time() - start

f, ax = plt.subplots(1, 2)

ax[0].plot(range(2, 13), medidaOrtGram, label = 'Gram-Schmidt Clásico')
ax[0].plot(range(2, 13), medidaOrtNorm, label = 'Ec. Normales')
ax[0].legend()
ax[0].set_title('Medida de Ortogonalidad')
ax[0].set_xlabel('Tamaño de la Matriz H (n)')
ax[0].set_ylabel('Medida de ortogonalidad')

ax[1].plot(range(2, 13), tiempoGram, label = 'Tiempos para\nGram-Schmidt')
ax[1].plot(range(2, 13), tiempoNorm, label = 'Tiempos para\nEc. Normales')
ax[1].legend()
ax[1].set_title('Tiempo de Ejecución')
ax[1].set_xlabel('Tamaño de la Matriz H (n)')
ax[1].set_ylabel('tiempo (s)')
f.set_size_inches(14, 6)

print('\nResultados del inciso c:\n')
print('El método con ecuaciones normales resulta en menor presición. Además')
print('Podemos notar que para n mayor a 10, el tiempo de ejecución ya tarda más')
print('que para el método de gram-schmidt.')

plt.show()