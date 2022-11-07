## Sistemas de Ecuaciones Lineales

Dada un sistema de ecuaciones $A\overline{x} = \overline{b}$, donde $A\in \mathcal M_{n},\ \overline{x},\ \overline{b}\in \mathbb R^{n}$ buscamos maneras eficientes de encontrar tal solución.

Para realizar tal tarea, nos apoyamos en dos clases de matrices que en un sistema de ecuaciones nos arrojan una solución de manera directa: las matrices triangulares inferiores y superiores. Los algoritmos en esta sección buscan transformar a $A$ en alguna combinación de este tipo de matrices. Los algortimos de *sustitución hacia atrás* y *sustitución hacia adelante* nos arrojan la solución cuando las matrices son triangulares superiores e inferiores, respectivamente. En el script [Sustitucion.py](https://github.com/edtrelo/AnalisisNumerico/blob/main/SistemasEcuacionesLineales/Sustitucion.py) está la implementación de tales algoritmos.

Cuando se resuelven sistemas de ecuaciones, las divisiones que se hacen para hacer ceros debajo de la diagonal (el proceso de triangularización) nos puede llevar a propagar errores por redondeo si los resultados de algunas divisiones son mayores a 1. El procedimiento de pivoteo parcial ubica la fila con el candidato a pivote que provocaría el menor valor absoluto para la división. El pivoteo total encuentra la fila y columna que puede usarse con tal fin. Ambos procedimientos están en [Pivoteo.py](https://github.com/edtrelo/AnalisisNumerico/blob/main/SistemasEcuacionesLineales/Pivoteo.py).

La *eliminación Gaussiana* aplica operaciones elementales a la matriz aumentada $A|\overline{b}$ para llevar al sistema original a uno equivalente que puede solucionarse con *sustitución hacia atrás* y que conserva la misma solución que el sistema original. Un procedimiento de pivoteo puede usarse para evitar una propagación del error por redondeo. Dos versiones del algoritmo, con pivoteo parcial y sin él, están en [SustitucionGaussiana.py](https://github.com/edtrelo/AnalisisNumerico/blob/main/SistemasEcuacionesLineales/SustitucionGaussiana.py).

Si $A$ es invertible, podemos expresar a $A$ por medio del producto de dos matrices, $L$ y $U$ matrices triangulares inferiores y superiores. La factorización LU también puede obtenerse por medio de pivoteos: 
- Si el pivoteo es parcial, la factorización nos arroja $U$, matriz triangular superior, a $L$, no necesariamente triangular, y a $P$, matriz de permutación. Estas matrices son tales que $A=LU$ y $PL$ es una matriz triangular inferior.
- Si el pivoteo es total, la factorización nos arroja $U$ y $L$, no necesariamente triangulares, y a $P$ y $Q$, matrices de permutación. Estas matrices son tales que $A=LU$ y además $PL$ es triangular inferior$ y $UQ$ es triangular superior.

El segundo tipo de factorización es la de *Cholesky*. Esta factorización es de la forma $A=LL^T$, donde $L$ es una matriz triangular inferior. También existe una factorización del estilo $A=LDL^T$ donde $L$ es también triangular inferior y $D$ es una matriz diagonal. 

Las implementaciones de estos cinco tipos de factorización (tres para $LU$ y dos para Cholesky) están en [Factorizacion.py](https://github.com/edtrelo/AnalisisNumerico/blob/main/SistemasEcuacionesLineales/Factorizacion.py).

Una vez obtenidas las factorizaciones, en [SolLU.py](https://github.com/edtrelo/AnalisisNumerico/blob/main/SistemasEcuacionesLineales/SolLU.py) y [SolCholesky.py](https://github.com/edtrelo/AnalisisNumerico/blob/main/SistemasEcuacionesLineales/SolCholesky.py) se usan los algoritmos de *sustitución hacia atrás* y *sustitución hacia adelante* junto con algunos cambios de variables para resolver sistemas lineales de la forma $A\overline{x} = \overline{b}$.

En [MatricesCuadradas.py](https://github.com/edtrelo/AnalisisNumerico/blob/main/SistemasEcuacionesLineales/MatricesCuadradas.py) se encuentra la clase `MatricesCuadradas` que implementa las propiedades de las matrices cuadradas: normas, el determinante, la condición, la transpuesta, el producto entre matrices y entre matriz y vector, las operaciones elementales y las distintas factorizaciones. Además, es una forma de asegurar que en los sistemas lineales que se van a usar $A$ sea cuadrada.
