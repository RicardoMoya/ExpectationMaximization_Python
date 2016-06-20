# Expectation-Maximization

El Expectation-maximization (EM) es un método estadístico de Clustering similar al K-means, pero con un enfoque probabilístico. Este método asume que todos los objetos (del data set) han sido generados a partir de ‘k’ distribuciones de probabilidad de las cuales desconocemos a priori sus parámetros.

Para saber más sobre el K-means, ir al siguiente tutorial:

http://jarroba.com/expectation-maximization-python-scikit-learn-ejemplos/

## Pseudocódigo

A continuación se muestra el Pseudocódigo del Expectation-maximization:

![alt jarroba](http://jarroba.com/wp-content/uploads/2016/06/EM_pseudocodigo_jarroba.png)

## Diagrama de Clases

A continuación se muestra el diagrama de clases para la implementación del Expectation-maximization, en el que se ven involucradas las clases Point (Point.py) y Cluster (Cluster.py). En el script EM.py (que no es una clase aunque así se representa en el diagrama de clases) está el método Main que ejecuta el EM.
 
![alt jarroba](http://jarroba.com/wp-content/uploads/2016/06/EM_ClassDiagram_jarroba.png)

En el script EM_scikit.py se muestra una solución del EM utilizando la librería scikit-learn, por tanto no es una implementación propia (o desde cero) de este algoritmo.

## Prerrequisitos

El código que se encuentra en este repositorio hace uso de las librerías de numpy, matplotlib, scipy y scikit-learn. Para descargar e instalar (o actualizar a la última versión con la opción -U) estas librerías con el sistema de gestión de paquetes pip, se deben ejecutar los siguiente comandos:

```ssh
$ pip install -U numpy
$ pip install -U matplotlib
$ pip install -U scipy
$ pip install -U scikit-learn
```

## Resultados esperados de los data set

El orden de los clusters no tiene porque coincidir con los propuestos, pero los centroides si que deben de tener valores muy similares a los indicados:

![alt jarroba](http://jarroba.com/wp-content/uploads/2016/05/DataSet_info_clusters_jarroba.png)

### Resultados:

![alt jarroba](http://jarroba.com/wp-content/uploads/2016/06/Cluster3C.png)
![alt jarroba](http://jarroba.com/wp-content/uploads/2016/06/Cluster3C2.png)
![alt jarroba](http://jarroba.com/wp-content/uploads/2016/06/Cluster5C.png)
![alt jarroba](http://jarroba.com/wp-content/uploads/2016/06/Cluster7C.png)


Para más detalles del proyecto vista la web de jarroba.com:

![alt jarroba](http://jarroba.com/wp-content/themes/jarrobav6/static/img/logojarroba.png)
