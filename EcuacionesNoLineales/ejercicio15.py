# ejercicio 15
# por: edgar armando trejo lópez
import numpy as np

def _parcial(f, x, respecto_a = 1, h = 0.0001):
    """Obtiene la parcial de una función f de R^3->R con respecto a la
    variable especificada: x1, x2 o x3.
    
    Args:
        f(function): función de tres variables a una variable.
        x(iterable): vector en donde se desea evaluar la derivada.
        respecto_a(int): variable respecto a la que se quiere derivar.
        h(float): la h de la definición de derivada parcial.
        
    Returns:
        float:
            La evaluación de la derivada parcial de f respecto a la 
            variable especificada en el vector dado."""
    # verificamos que los argumentos sean adecuados
    if respecto_a not in [1, 2, 3]:
        raise Exception('La variable respecto a la que se deriva debe ser xj, j = 1,2,3')
    elif len(x) != 3:
        raise Exception("La longuitud del vector debe ser 3.")
    # hacemos una copia de x para poder modificarlo
    X = np.array(x, dtype = np.float64)
    # a la entrada a derivar le sumamos h
    X[respecto_a - 1] += h
    # obtenemos la derivada parcial
    return (f(*X)-f(*x))/h

def _jacobiana(F, x):
    """Obtiene la matriz jacobiana de la función F evaluada en x. Se asume que 
    F es de R^3->R^3 y que len(x) es 3.
    
    Args:
        F: lista de funciones de R^3 a R.
        x: vector en R^3"""
    J = np.zeros((3, 3))
    for i, f in enumerate(F):
        for j in [1, 2, 3]:
            J[i, j-1] = _parcial(f, x, respecto_a = j)
    return J

def Newton(F, aprox, k = 100):
    """Resuelve el sistema F(x) = 0 por medio del método de Newton
    generalizado, donde F: R^3 -> R^3 y aprox es un vector con una 
    solución aproximada al sistema. Se hacen k iteraciones.

    No se evaluan condiciones de convergencia.
    
    Args:
        F: lista de funciones de R^3 -> R. El número de elementos de F es 3.
        aprox (iterable): solución aproximada. Su longuitud debe ser 3
        k: número máximo de iteraciones."""
    # hacemos k iteraciones 
    for i in range(k):
        # calculamos la jacobiana evaluada en xk
        Jk = _jacobiana(F, aprox)
        # calculamos la función evaluada en xk
        Fk = np.array([F[i](*aprox) for i in range(3)], dtype = np.float64)
        # actualizamos el valor de xk
        aprox -= np.dot(np.linalg.inv(Jk), Fk)

    return aprox
    
# establecemos las ecuaciones del sistema, todas deben estar igualdas a cero
f1 = lambda I, phi, delta: I*np.cos(phi) - 2/3
f2 = lambda I, phi, delta: np.cos(delta) + 0.91*I*np.sin(delta + phi) - 1.22
f3 = lambda I, phi, delta: 0.76*I*np.cos(phi + delta) - np.sin(delta)
# F es la función cuyas entradas son todas las funciones del sistema
F = (f1, f2, f3)
# Resolvemos por medio del método de Newton Generalizado

# a) ------------------------------------------------------
aprox = (1, 0.1, 0.1)
X = Newton(F, aprox)
print('Para la aproximación I = {0[0]}, \u03d5 = {0[1]}, \u03b4 = {0[2]}'.format(aprox))
print('La solución aproximada es X = {}'.format(X))
# b) -------------------------------------------------------
aprox = (1, 1, 1)
X = Newton(F, aprox)
print('\nPara la aproximación I = {0[0]}, \u03d5 = {0[1]}, \u03b4 = {0[2]}'.format(aprox))
print('La solución aproximada es X = {}'.format(X))
# concluciones 
print('\nAdmisibilidad:')
print('\tEn la primer aproximación tenemos que')
print('\tI es positivo y los ángulos están ambos entre 0 y 2\u03c0.')
print('\n\tEn la segunda itercion I es negativo y además \u03d5 < 0.')
print('\n\tLa primer solución es admisible y la segunda no lo es.')