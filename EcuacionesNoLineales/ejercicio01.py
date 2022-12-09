# ejercicio 01
# por: edgar armando trejo lópez

import numpy as np

def biseccion(f, a, b, tol = 1/10**6, maxiter = 100):
    """Obtiene algún cero de la función f usando el método de la bisección en el
    intervalo [a, b].
    
    La toleracia para la longuitud del intervalo está dada por default como 1/10^6 y el 
    número máximo de iteraciones, por default, es 100.
    
    Args:
        f(function)
        a, b (float): extremos del intervalo de búsqueda, tal que a<b.
        tol(float): tolerancia de la separación entre dos soluciones para iteraciones
            consecutivas.
        maxiter(int): número máximo de iteraciones
        
    Returns:
        x(float): solución (aproximada) a f(x)=0."""
    def sign(x):
        """Obtiene el signo de x.
        
        Args:
            x(float)
        
        Returns:
            int:
                1 si x >= 0, -1 en otro caso."""
        if x >= 0:
            return 1
        else:
            return -1
    # si el signo de f(a) y f(b) es el mismo, no podemos asegurar que encontremos la solución.
    if sign(f(a)) == sign(f(b)):
        print("El signo de la evaluación en los extremos debería ser diferente.")
        return
    # usaremos k para llevar cuenta de las iteraciones.
    k = 0
    while abs(b - a) > tol and k < maxiter:
        # calculamos el punto medio del intervalo
        m = a + (b-a)/2
        # varificamos para qué extremo f(m) tiene el mismo signo.
        if sign(f(a)) == sign(f(m)):
            a = m
        else:
            b = m
        # aumentamos el contador de iteraciones.
        k += 1
    return m

def puntoFijo(g, dg, x0, tol = 1/10**6, maxiter = 100):
    """
    Solucióna la ecuación f(x)=0, tal que f(x) = g(x) - x, por medio del método
    del punto fijo, es decir, encuentra x tal que g(x) = x.
    
    Args:
        g(function): tal que f(x) = g(x) - x
        dg(function): derivada de g.
        x0(float): punto incial.
        tol(float): tolerancia de la separación entre dos soluciones para iteraciones
            consecutivas.
        maxiter(int): número máximo de iteraciones
        
    Returns:
        x(float): solución (aproximada) a f(x) = g(x)-x = 0."""
    # calculamos la evalución de la derivada en el punto inicial
    if abs(dg(x0)) >= 1:
        raise Exception("El método no va a converger a la solución.")
    # calculamos x1
    xk = g(x0)
    # inciamos el contador de iteraciones.
    k = 0
    while abs(xk - x0) > tol and k < maxiter:
        # verificamos que no exista la condición de no convergencia.
        if abs(dg(xk)) >= 1:
            raise Exception("El método no va a converger a la solución.")
        # calculamos el siguiente punto
        xk, x0 = g(xk), xk
        k += 1
    return xk

def newton(f, df, x0, tol = 1/10**6, maxiter = 100):
    """
    Solucióna la ecuación f(x)=0 por medio del método de Newton dado el punto
    inicial x0.
    
    Args:
        f(function)
        df(function): derivada de f.
        x0(float): punto incial.
        tol(float): tolerancia de la separación entre dos soluciones para iteraciones
            consecutivas.
        maxiter(int): número máximo de iteraciones
        
    Returns:
        x(float): solución a f(x) = 0"""
    k = 0
    # checamos si en el punto inicial la derivada no
    # evalua en cero
    try:
        # obtenemos el primer punto.
        xk = x0 - f(x0)/df(x0)
        # queremos tener un número máximo de iteraciones y
        # si dos iteraciones resultan muy pegadas, entonces
        # terminamos el algoritmo.
        while abs(xk - x0) > tol and k < maxiter:
            # checamos si la derivada no evalua a cero
            if df(xk) == 0:
                # si es el caso, no regresamos nada
                return
            else:
                # evaluamos el siguiente punto
                # actualizamos el punto anterior
                xk, x0 = xk - f(xk)/df(xk), xk
                # aumentamos en 1 el contador de iteraciones
                k += 1
        # o pasamos el número máximo de iteraciones o dos 
        # puntos ya estaban muy juntos.
        return xk
    except:
        # el punto inicial evalua a cero la derivada.
        return

def secante(f, x0, x1, tol = 1/10**6, maxiter = 100):
    """
    Solucióna la ecuación f(x)=0 por medio del método de la secante dados los 
    puntos iniciales x0 y x1.
    
    Args:
        f(function)
        x0, x1(float): puntos inciales.
        tol(float): tolerancia de la separación entre dos soluciones para iteraciones
            consecutivas.
        maxiter(int): número máximo de iteraciones
        
    Returns:
        x(float): solución a f(x) = 0"""
    k = 0
    # checamos si en el punto inicial el denominador no
    # evalua en cero
    try:
        # obtenemos el primer punto.
        xk = x1 - f(x1)*(x1 - x0)/(f(x1)- f(x0))
        # queremos tener un número máximo de iteraciones y
        # si dos iteraciones resultan muy pegadas, entonces
        # terminamos el algoritmo.
        while abs(xk - x1) > tol and k < maxiter:
            # checamos si el denominador evalua cero
            if f(xk) - f(x1) == 0:
                # si es el caso, no regresamos nada
                return
            else:
                # evaluamos el siguiente punto
                # actualizamos el punto anterior
                xk, x1 = xk - f(xk)*(xk - x1)/(f(xk) - f(x1)), xk
                # aumentamos en 1 el contador de iteraciones
                k += 1
        # o pasamos el número máximo de iteraciones o dos 
        # puntos ya estaban muy juntos.
        return xk
    except:
        # el punto inicial evalua a cero la derivada.
        return

def falsaRegla(f, a, b, tol = 1/10**6, maxiter = 100):
    """Soluciona la ecuación no lineal f(x)=0 por medio del método de la
    regla falsa en el intervalo a, b.
    
    Args:
        f(function)
        a, b (float): extremos del intervalo de búsqueda, tal que a<b.
        tol(float): tolerancia de la separación entre dos soluciones para iteraciones
            consecutivas.
        maxiter(int): número máximo de iteraciones
        
    Returns:
        x(float): solución (aproximada) a f(x)=0."""
    def sign(x):
        """Obtiene el signo de x.
        
        Args:
            x(float)
        
        Returns:
            int:
                1 si x >= 0, -1 en otro caso."""
        if x >= 0:
            return 1
        else:
            return -1
    # si el signo de f(a) y f(b) es el mismo, no podemos asegurar que encontremos la solución.
    if sign(f(a)) == sign(f(b)):
        print("El signo de la evaluación en los extremos debería ser diferente.")
        return
    # usaremos k para llevar cuenta de las iteraciones.
    k = 0
    while abs(b - a) > tol and k < maxiter:
        # calculamos el punto donde la secante cruza el eje x
        c = b - f(b)*(b - a)/(f(b) - f(a))
        # varificamos para qué extremo f(m) tiene el mismo signo.
        if sign(f(a)) == sign(f(c)):
            a = c
        else:
            b = c
        # aumentamos el contador de iteraciones.
        k += 1
    return c
