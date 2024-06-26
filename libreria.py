from google.colab import drive
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.metrics import confusion_matrix, auc
import scipy.stats as stats

"""
mimodulo:
-------------------------

Este módulo proporciona clases para analizar un conjunto de datos.

Clases:
    ResumenNumerico: Implementa un resumen de datos numéricos para una lista de datos.
    GeneradoraDeDatos: Implementa clases para generar datos de diferentes distribuciones.
    ResumenGrafico: Implementa clases para generar histogramas y curvas de densidad.
    Regresion: Implementa una regresión lineal múltiple
    RegresionLineal: Implementa una regresión lineal.
    RegresionLogistica: Implementa una regresión logística.
    PruebaChiCuadrado: Implementa un test de chi-cuadrado.
"""

class ResumenNumerico:
  """
  Clase para realizar un resumen numérico de una lista de datos.

  Atributos:
    datos: Lista de datos numéricos
  """

  def __init__(self, datos: np.ndarray):
    """
    Constructor de la clase ResumenNumerico

    Args:
      datos (np.array): Lista de datos numéricos
    """
    self.datos = datos

  def calculo_de_media(self) -> float:
    """
    Calcula la media de los datos

    Returns:
      float: Media de los datos
    """
    media = np.mean(self.datos)
    return media

  def calculo_de_mediana(self, datos = None) -> float:
    """
    Calcula la mediana de los datos

    Args:
      datos (np.array, opcional): Lista de datos numéricos. Defaults None.

    Returns:
      float: Mediana de los datos
    """
    if datos is None:
      datos = self.datos
      mediana = np.median(self.datos)
      return mediana

  def calculo_de_varianza(self) -> float:
    """
    Calcula la varianza de los datos

    Returns:
      float: Varianza de los datos
    """
    media = self.calculo_de_media()
    suma_cuadrados = sum((x - media) ** 2 for x in self.datos)
    varianza = suma_cuadrados / len(self.datos)
    return varianza

  def calculo_de_desvio_estandar(self) -> float:
    """
    Calcula el desvío estándar de los datos

    Returns:
      float: Desvío estándar de los datos
    """
    varianza = self.calculo_de_varianza()
    desvio_estandar = np.sqrt(varianza)
    return desvio_estandar

  def calculo_de_cuartiles(self) -> list:
    """
    Calcula los cuartiles de los datos

    Returns:
      list: Lista con los cuartiles
    """
    q1 = np.percentile(self.datos, 25)
    q2 = np.percentile(self.datos, 50)
    q3 = np.percentile(self.datos, 75)
    return [q1, q2, q3]

  def generacion_resumen_numerico(self) -> dict:
    """
    Genera un diccionario con los resultados numéricos del resumen

    Returns:
      dict: Diccionario con los resultados numéricos
    """
    res_num = {
        'Media': self.calculo_de_media(),
        'Mediana': self.calculo_de_mediana(),
        'Varianza': self.calculo_de_varianza(),
        'Desvio': self.calculo_de_desvio_estandar(),
        'Cuartiles': self.calculo_de_cuartiles(),
        'Mínimo': min(self.datos),
        'Máximo': max(self.datos)
        }
    return res_num

  def muestra_resumen(self):
    """
    Muestra los resultados numéricos del resumen
    """
    res_num = self.generacion_resumen_numerico()
    for estad, valor in res_num.items():
      print(f"{estad}: {np.round(valor,3)}")
    return

class ResumenGrafico:
  """
  Clase para generar histogramas y curvas de densidad.

  Atributos:
    datos: Lista de datos numéricos
  """
  def __init__(self, datos):
    """
    Constructor de la clase ResumenGrafico

    Args:
      datos (np.array): Lista de datos numéricos
    """
    self.datos = np.array(datos)

  def generacion_histograma(self, h) -> tuple:
    """
    Genera el histograma de los datos

    Args:
      h (float): Ancho del bin

    Returns:
      tuple: Tupla con los bins y el histograma
    """
    # Calcula el valor mínimo y máximo de los datos
    val_min = min(self.datos)
    val_max = max(self.datos)

    # Calcula los bordes de los bins utilizando el ancho h
    bins = np.arange(val_min, val_max, h)

    # Si el valor máximo es mayor que el último bin, agrega un bin adicional
    if val_max > bins[-1]:
      bins = np.append(bins, bins[-1] + h)

    # Calcula la cantidad de bins
    m = len(bins)

    # Inicializa el histograma con ceros
    histo = [0] * (m - 1)  # El histograma tiene m-1 bins

    # Recorre los datos y asigna cada valor a un bin en el histograma
    for valor in self.datos:
      for i in range(len(bins) - 1):
          if valor == bins[0]:
             histo[0] += 1
             break
          elif bins[i] < valor <= bins[i + 1]:
              histo[i] += 1
              break

    # Normaliza el histograma
    for i in range(len(histo)):
          histo[i] /= (len(self.datos) * h)
    # Retorna los bordes de los bins y el histograma normalizado
    return bins, histo

  def evalua_histograma(self, x, h) -> np.array:
    """
    Evalúa un conjunto de datos en la función histograma

    Args:
      x (np.array): Conjunto de datos a evaluar
      h (float): Ancho del bin

    Returns:
      np.array: Vector de frecuencias

    """
    # Genera el histograma utilizando la función generacion_histograma
    bins, histo = self.generacion_histograma(h)

    # Inicializa una lista para almacenar los resultados
    res = [0] * len(x)

    # Itera sobre los valores de x
    for j in range(len(x)):
      # Si el valor de x es igual al valor mínimo de los datos, asigna la frecuencia del primer bin al resultado correspondiente
      if x[j] == min(self.datos):
          res[j] = histo[0]
      else:
        # Si el valor de x está dentro de un bin, asigna la frecuencia del bin correspondiente al resultado
          for i in range(len(bins) - 1):
            if bins[i] < x[j] <= bins[i + 1]:
                res[j] = histo[i]
                break
    return res

  def kernel_gaussiano(self, x):
    """
    Calcula el valor del kernel Gaussiano para un valor x

    Args:
      x (float): Valor para el que se calcula el kernel

    Returns:
      float: Valor del kernel Gaussiano
    """
    valor_kernel_gaussiano = (1/(np.sqrt(2*np.pi)))*np.e**(-1/2*x**2)
    return valor_kernel_gaussiano

  def kernel_uniforme(self, x):
    """
    Calcula el valor del kernel Uniforme para un valor x

    Args:
      x (float): Valor para el que se calcula el kernel

    Returns:
      float: Valor del kernel Uniforme
    """
    if(-1/2) < x < (1/2):
      valor_kernel_uniforme = 1
    else:
      valor_kernel_uniforme = 0
    return valor_kernel_uniforme

  def kernel_cuadratico(self, x):
    """
    Calcula el valor del kernel Cuadrático para un valor x

    Args:
      x (float): Valor para el que se calcula el kernel

    Returns:
      float: Valor del kernel Cuadrático
    """
    if -1 < x < 1:
      valor_kernel_cuadratico = (3/4)*(1 - x**2)
    else:
      valor_kernel_cuadratico = 0
    return valor_kernel_cuadratico

  def kernel_triangular(self, x):
    """
    Calcula el valor del kernel Triangular para un valor x

    Args:
      x (float): Valor para el que se calcula el kernel

    Returns:
      float: Valor del kernel Triangular
    """
    if -1 <= x <= 0:
      valor_kernel_triangular = 1 + x
    elif 0 <= x <= 1:
      valor_kernel_triangular = 1 - x
    else:
      valor_kernel_triangular = 0
    return valor_kernel_triangular

  def densidad_nucleo(self, x, h, kernel):
    """
    Calcula la densidad de un núcleo en un conjunto de datos
    Args:
      x (np.array): Conjunto de datos
      h (float): Ancho del núcleo
      kernel (str): Tipo de núcleo

    Returns:
      np.array: Vector de densidades

    """

    # Inicializa una lista para almacenar la densidad calculada para cada valor en x
    density = [0]*len(x)

    # Itera sobre los valores en x
    for i in range(len(x)):
      # Inicializa una variable para almacenar la suma de los núcleos ponderados
      suma = 0
      # Itera sobre los datos
      for j in range(len(self.datos)):
        # Calcula el valor normalizado de u
        u = (self.datos[j]-x[i])/h
        # Calcula el valor del núcleo correspondiente según el tipo de kernel
        if kernel == 'uniforme':
          suma += self.kernel_uniforme(u)
        if kernel == 'gaussiano':
          suma += self.kernel_gaussiano(u)
        if kernel == 'cuadratico':
          suma += self.kernel_cuadratico(u)
        if kernel == 'triangular':
          suma += self.kernel_triangular(u)

      # Calcula la densidad para el valor actual de x y lo almacena en la lista density
      density[i] = suma/(h*len(self.datos))

    # Retorna la lista de densidades calculadas
    return density

class GeneradoraDeDatos:
  """
  Clase para generar datos de diferentes distribuciones.

  Atributos:
    n: Número de datos a generar
  """
  def __init__(self, n):
    """
    Constructor de la clase GeneradoraDeDatos

    Args:
      n (int): Número de datos a generar
    """
    self.datos = None
    self.n = n

  def generar_datos_dist_unif(self, low, high):
    """
    Genera un arreglo de n datos con distribución uniforme entre low y high

    Args:
      low (float): Limite inferior del intervalo
      high (float): Limite superior del intervalo

    """
    self.datos = np.random.uniform(low, high, self.n)
    return

  def generar_datos_dist_norm(self, media, desvio):
    """
    Genera un arreglo de n datos con distribución normal con media y desvio

    Args:
      media (float): Media de la distribución normal
      desvio (float): Desviación estándar de la distribución normal
    """
    self.datos = np.random.normal(loc=media, scale=desvio, size=self.n)
    return

  def generar_datos_dist_BS(self):
    """
    Genera un arreglo de n datos con distribución Bart Simpson
    """
    # Genera una muestra aleatoria de tamaño n de una distribución uniforme entre 0 y 1
    u = np.random.uniform(size=(self.n))

    # Copia la muestra aleatoria generada
    y = u.copy()

    # Encuentra los índices donde los valores de u son mayores que 0.5
    ind = np.where(u > 0.5)[0]

    # Reemplaza los valores en y en los índices encontrados anteriormente con muestras de una distribución normal con media 0 y desviación estándar 1
    y[ind] = np.random.normal(0, 1, size=len(ind))

    # Divide el intervalo [0,1] en 5 subintervalos y reemplaza los valores en y dentro de cada subintervalo con muestras de una distribución normal
    for j in range(5):
        ind = np.where((u > j * 0.1) & (u <= (j+1) * 0.1))[0]
        y[ind] = np.random.normal(j/2 - 1, 1/10, size=len(ind))

    # Guarda los datos generados en self.datos
    self.datos = y
    return

  def mostrar_datos(self):
    """
    Muestra los datos generados
    """
    return self.datos

  def pdf_unif(self, x, l_inf, l_sup):
    """
    Calcula la curva teórica de la distribución uniforme

    Args:
      x (np.array): Arreglo de valores para los que se calcula la curva teórica
      l_inf (float): Limite inferior del intervalo
      l_sup (float): Limite superior del intervalo

    Returns:
      np.array: Curva teórica de la distribución uniforme
    """
    # Definimos los parámetros de la distribución uniforme
    a = l_inf  # Límite inferior del intervalo
    b = l_sup  # Límite superior del intervalo

    # Calculamos la función de densidad de probabilidad (PDF) teórica
    pdf_uniforme = np.full_like(x, 1 / (b - a))
    return pdf_uniforme

  # Curva teorica de la distribución normal
  def pdf_norm(self, x, media, desvio):
    """
    Calcula la curva teórica de la distribución normal
    Args:
      x (np.array): Arreglo de valores para los que se calcula la curva teórica
      media (float): Media de la distribución normal
      desvio (float): Desviación estándar de la distribución

    Returns:
      np.array: Curva teórica de la distribución normal
    """
    return norm.pdf(x, loc=media, scale=desvio)

  def f_BS(self, x):
    """
    Calcula la función de densidad de la distribución Bart Simpson

    Args:
      x (float): Valor para el que se calcula la función de densidad

    Returns:
      float: Función de densidad de la distribución Bart Simpson
    """
    x_norm = norm.pdf(x,loc=0,scale=1)
    x_sum = 0
    for j in range(5):
      x_sum += norm.pdf(x,(j/2)-1,1/10)
    return (1/2)*x_norm + (1/10)*x_sum

  def pdf_BS(self, x):
    """
    Calcula la curva teórica de la distribución Bart Simpson

    Args:
      x (np.array): Arreglo de valores para los que se calcula la curva teórica

    Returns:
      np.array: Curva teórica de la distribución Bart Simpson
    """
    res_BS = np.zeros(len(x))
    for i in range(len(x)):
      res_BS[i] = self.f_BS(x[i])
    return res_BS


class Regresion:
  """
  Clase para realizar regresiones.

  Atributos:
    X: Variable independiente
    y: Variable dependiente
  """
  def __init__(self, X, y):
    """
    Constructor de la clase Regresion

    Args:
      X (np.array): Variable independiente
      y (np.array): Variable dependiente
    """
    self.X = X
    self.y = y
    self.modelo = None

  def ajustar_modelo(self):
    """
    Ajusta el modelo a los datos.

    Returns:
      modelo: Modelo ajustado
    """
    X = sm.add_constant(self.X)
    self.modelo = sm.OLS(self.y, X).fit()
    return self.modelo

  def betas(self):
    """
    Calcula los betas del modelo ajustado.

    Returns:
      betas: Vector de betas
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      return self.modelo.params

  def p_valores(self):
    """
    Calcula los p valores del modelo ajustado.

    Returns:
      p_valores: Vector de p valores
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      return self.modelo.pvalues

  def t_observado(self):
    """
    Calcula los t observados del modelo ajustado.

    Returns:
      t_observado
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      return self.modelo.tvalues

  def error_estandar(self):
    """
    Calcula el error estándar del modelo ajustado.

    Returns:
      error_estandar: Error estándar del modelo ajustado
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      return self.modelo.bse

  def datos_modelo(self):
    """
    Calcula los datos del modelo ajustado.

    Returns:
      datos_modelo: Datos del modelo ajustado
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      return self.modelo.summary()

  def graficar_regresion(self):
    """
    Grafica cada variable independiente contra la variable dependiente.
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
        # itero por columnas
        for i in range(self.X.shape[1]):
            plt.figure()
            plt.scatter(self.X[:, i], self.y, label=f'x{i+1} vs y')
            plt.plot(self.X[:, i], self.modelo.params[0] + self.modelo.params[i] * self.X[:, i], color='red',
                     label=f"y = {round(self.modelo.params[0],3)} + {round(self.modelo.params[i+1],3)}*X{i+1}")
            plt.xlabel(f'x{i+1}')
            plt.ylabel('y')
            plt.title(f'x{i+1} vs y')
            plt.legend()
        plt.show()

  def modificar_predictora(self, X):
    """
    Modifica la variable independiente.

    Args:
      X (np.array): Nueva variable independiente
    """
    self.X = X

  def modificar_respuesta(self, y):
    """
    Modifica la variable dependiente.

    Args:
      y (np.array): Nueva variable dependiente
    """
    self.y = y
    return

  def R_cuadrado(self):
    """
    Calcula el coeficiente de determinación.

    Returns:
      R_cuadrado: Coeficiente de determinación
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      return self.modelo.rsquared

  # Calcular coeficiente de determinación
  def calcular_r2(self):
    """
    Calcula el coeficiente de determinación.

    Returns:
      r2: Coeficiente de determinación
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      r2 = self.modelo.rsquared
      r2_ajustado = self.modelo.rsquared_adj
      return r2, r2_ajustado

class RegresionLineal(Regresion):
  """
  Clase para realizar regresiones lineales.

  Atributos:
    x: Variable independiente
    y: Variable dependiente
  """
  def __init__(self, x, y):
    """
    Constructor de la clase RegresionLineal

    Args:
      x (np.array): Variable independiente
      y (np.array): Variable dependiente
    """
    super().__init__(x, y)

  def ajustar_modelo_minimos_cuadrados(self):
    """
    Ajusta el modelo utilizando el método de mínimos cuadrados.

    Returns:
      b0: Intercepto del modelo
      b1: Pendiente del modelo
    """
    b1 = sum((self.X - np.mean(self.X)) * (self.y - np.mean(self.y))) / sum((self.X - np.mean(self.X)) ** 2)
    b0 = np.mean(self.y) - b1 * np.mean(self.X)
    return b0, b1

  # Graficar la dispersión de puntos y la recta de mejor ajuste
  def graficar_regresion(self, nombre_respuesta='Variable Respuesta', nombre_predictora='Variable Predictora'):
    """
    Grafica la dispersión de puntos y la recta de mejor ajuste.

    Args:
      nombre_respuesta (str): Nombre de la variable respuesta
      nombre_predictora (str): Nombre de la variable predictora
    """
    b0, b1 = self.ajustar_modelo_minimos_cuadrados()
    y_pred = b0 + b1 * self.X
    plt.scatter(self.X, self.y)
    plt.plot(self.X, y_pred, color='red', label=f'y = {round(b0,3)} + {round(b1,3)}*x')
    plt.xlabel(nombre_predictora)
    plt.ylabel(nombre_respuesta)
    plt.show()

  # Calcular coeficiente de correlación
  def coeficiente_correlacion(self):
    """
    Calcula el coeficiente de correlación.

    Returns:
      coeficiente_correlacion: Coeficiente de correlación
    """
    coeficiente_correlacion = np.corrcoef(self.X, self.y)
    return coeficiente_correlacion[0, 1]

  # Realizar el análisis de los residuos
  def graficar_residuos(self):
    """
    Grafica los residuos.
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      y_pred = self.modelo.fittedvalues
      residuos = self.modelo.resid
      plt.scatter(y_pred, residuos)
      plt.xlabel('Valores predichos')
      plt.ylabel('Residuos')
      plt.axhline(y=0, color='r', linestyle='--')
      plt.show()
      sm.qqplot(residuos, line='s')
      plt.show()

  # Calcular intervalo de confianza
  def intervalo_confianza(self):
    """
    Calcula el intervalo de confianza.

    Returns:
      intervalo_confianza: Intervalo de confianza
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      return self.modelo.conf_int()

  # Calcular intervalo de predicción
  def intervalo_prediccion(self, nuevos_x):
    """
    Calcula el intervalo de predicción.

    Args:
      nuevos_x (np.array): Valores de la variable independiente para los que se desea calcular el intervalo de predicción

    Returns:
      intervalo_prediccion: Intervalo de predicción
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      nuevos_x = sm.add_constant(nuevos_x)
      predicciones = self.modelo.get_prediction(nuevos_x)
      intervalos = predicciones.conf_int(obs=True)
      return intervalos

class RegresionLogistica:
  """
  Clase para realizar regresiones logística.

  Atributos:
    x: Variable independiente
    y: Variable dependiente
  """
  def __init__(self, X, y):
    """
    Constructor de la clase RegresionLogistica

    Args:
      X (np.array): Variable independiente
      y (np.array): Variable dependiente
      x_train (np.array): Conjunto de entrenamiento de la variable independiente
      y_train (np.array): Conjunto de entrenamiento de la variable dependiente
      x_test (np.array): Conjunto de prueba de la variable independiente
      y_test (np.array): Conjunto de prueba de la variable dependiente
    """
    self.X = X
    self.y = y
    self.x_train = X
    self.y_train = y
    self.x_test = None
    self.y_test = None
    self.modelo = None

  def setear_datos_train_test(self, porcentaje=0.2, semilla=None):
    """
    Separa los datos en conjuntos de entrenamiento y prueba.

    Args:
      porcentaje (float): Porcentaje de datos para el conjunto de entrenamiento
      semilla (int): Semilla para la generación de números aleatorios
    """
    if semilla is not None:
        random.seed(semilla)

    # Elegimos el % de los datos que se separan para el test
    cuantos = int(len(self.X) * porcentaje)
    cuales = random.sample(range(len(self.X)), cuantos)
    self.x_train = self.X.drop(cuales)
    self.x_test = self.X.iloc[cuales]
    self.y_train = self.y.drop(cuales)
    self.y_test = self.y.iloc[cuales]
    return

  def ajustar_modelo(self):
    """
    Ajusta el modelo utilizando la librería statsmodels.

    Returns:
      modelo: Modelo ajustado
    """
    X = sm.add_constant(self.x_train)
    self.modelo = sm.Logit(self.y_train, X).fit()
    return self.modelo

# A partir de los datos de test, poder calcular matriz de confusión
  def calcular_matriz_confusion(self, umbral=0.5):
    """
    Calcula la matriz de confusión.

    Args:
      umbral (float): Umbral de clasificación

    Returns:
      mc: Matriz de confusión
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      y_pred_prob = self.modelo.predict(sm.add_constant(self.x_test))
      y_pred = [1 if x > umbral else 0 for x in y_pred_prob]
      mc = confusion_matrix(self.y_test, y_pred)
      return mc

#error total de mala clasificación
  def calcular_error_total(self, umbral=0.5)->float:
    """
    Calcula el error total.

    Args:
      umbral (float): Umbral de clasificación

    Returns:
      error_total: Error total
    """
    mc = self.calcular_matriz_confusion(umbral)
    error_total = mc[0,1] + mc[1,0]
    return error_total/len(self.y_test)

#sensibilidad y especificidad.
  def calcular_sensibilidad_especificidad(self, umbral=0.5)->tuple:
    """
    Calcula la sensibilidad y la especificidad.

    Args:
      umbral (float): Umbral de clasificación

    Returns:
      sensibilidad: Sensibilidad
      especificidad: Especificidad
    """
    mc = self.calcular_matriz_confusion(umbral)
    sensibilidad = mc[1,1]/(mc[1,1]+mc[1,0])
    especificidad = mc[0,0]/(mc[0,0]+mc[0,1])
    return sensibilidad, especificidad

#Predecir valores de respuesta ante nuevas entradas utilizando un umbral.
  def predecir_respuesta(self, nuevo_X, umbral=0.5):
    """
    Predice valores de respuesta ante nuevas entradas.

    Args:
      nuevo_X (np.array): Arreglo de valores de la variable independiente para las nuevas entradas
      umbral (float): Umbral de clasificación

    Returns:
      y_pred: Arreglo de valores de respuesta predichos
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      y_pred = self.modelo.predict(sm.add_constant(nuevo_X))
      y_pred = [1 if x > umbral else 0 for x in y_pred]
      return y_pred

#Realizar la curva ROC, calculo de AUC y clasificacion
  def graficar_curva_roc_y_clasificar(self):
    """
    Grafica la curva ROC y calcula el AUC y la clasificación.
    """
    if self.modelo is None:
      print("El modelo no ha sido ajustado aún.")
    else:
      p = np.linspace(0, 1, 100)
      y_pred = self.modelo.predict(sm.add_constant(self.x_test))
      sensibilidad_p = []
      especificidad_p = []
      for i in p:
        y_pred_i = 1 * (y_pred > i)
        a = np.sum((y_pred_i == 1) & (self.y_test == 1))
        b = np.sum((y_pred_i == 1) & (self.y_test == 0))
        c = np.sum((y_pred_i == 0) & (self.y_test == 1))
        d = np.sum((y_pred_i == 0) & (self.y_test == 0))
        sensibilidad_p.append(a / (a + c))
        especificidad_p.append(d / (b + d))

      plt.plot(1 - np.array(especificidad_p), sensibilidad_p)
      plt.title('Curva ROC')
      plt.ylabel('Sensibilidad')
      plt.xlabel('1 - Especificidad')
      plt.show()
      roc_auc = auc(1-np.array(especificidad_p), sensibilidad_p)
      print("AUC:", roc_auc)

      mensaje = ""
      if 0.90 < roc_auc <= 1.00:
        mensaje = "El clasificador es excelente"
      elif 0.80 < roc_auc <= 0.90:
        mensaje = "El clasificador es bueno"
      elif 0.70 < roc_auc <= 0.80:
        mensaje = "El clasificador es regular"
      elif 0.60 < roc_auc <= 0.70:
        mensaje = "El clasificador es pobre"
      elif 0.50 <= roc_auc <= 0.60:
        mensaje = "El clasificador es fallido"
      print(mensaje)

class PruebaChiCuadrado:
  """
  Clase para realizar pruebas de chi-cuadrado.

  Atributos:
    prob_esperadas: Probabilidades esperadas
    prob_observadas: Probabilidades observadas
    alfa: Nivel de significancia
    chi_cuadrado: Chi-cuadrado
    p_valor: P-valor
    grados_libertad: Grados de libertad
  """
  def __init__(self, prob_esperadas, prob_observadas, alfa):
    """
    Constructor de la clase PruebaChiCuadrado

    Args:
      prob_esperadas (np.ndarray): Probabilidades esperadas
      prob_observadas (np.ndarray): Probabilidades observadas
      alfa (float): Nivel de significancia
    """
    self.prob_esperadas = prob_esperadas
    self.prob_observadas = prob_observadas
    self.alfa = alfa
    self.chi_cuadrado = None
    self.p_valor = None
    self.grados_libertad = None

  def calcular_chi_cuadrado(self)->float:
    """
    Calcula el chi-cuadrado.

    Returns:
      chi_cuadrado (float): Chi-cuadrado
    """
    self.chi_cuadrado = np.sum((self.prob_observadas - self.prob_esperadas)**2 / self.prob_esperadas)
    return self.chi_cuadrado

  def set_grados_libertad(self, k)->None:
    """
    Establece los grados de libertad.

    Args:
      k (int): Grados de libertad
    """
    self.grados_libertad = k
    return

  def calcular_p_valor(self)->float:
    """
    Calcula el p-valor.

    Returns:
      p_valor(float): P-valor
    """
    self.p_valor = stats.chi2.sf(self.chi_cuadrado, self.grados_libertad)
    return self.p_valor

  def rechazar_hipotesis_nula(self)->bool:
    """
    Determina si se rechaza o no la hipótesis nula.

    Returns:
      bool: True si se rechaza la hipótesis nula, False en caso contrario
    """
    return self.p_valor < self.alfa