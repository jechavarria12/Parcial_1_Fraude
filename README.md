# Buo Ai – Detección de Fraude Financiero con Machine Learning

Este proyecto, desarrollado en el marco de la asignatura de IA Aplicada a la Economía, tiene como objetivo identificar transacciones fraudulentas mediante técnicas de machine learning. Se utiliza un dataset real obtenido de Hugging Face, y se aplican diversas técnicas de preprocesamiento, análisis descriptivo y modelado utilizando algoritmos como Regresión Ridge (L2), Regresión Lasso (L1) y k-Nearest Neighbors (kNN). Además, se implementa un balanceo de clases con SMOTE y se evalúan los modelos mediante métricas estándar y curvas ROC.

---

## Tabla de Contenidos

- [Introducción](#introducción)
- [Descripción del Proyecto](#descripción-del-proyecto)
- [Datos](#datos)
- [Preprocesamiento de Datos](#preprocesamiento-de-datos)
  - [Instalación de Librerías y Configuración del Entorno](#instalación-de-librerías-y-configuración-del-entorno)
  - [Carga y División de Datos](#carga-y-división-de-datos)
  - [Análisis Descriptivo y Exploratorio](#análisis-descriptivo-y-exploratorio)
  - [Limpieza y Creación de Variables Derivadas](#limpieza-y-creación-de-variables-derivadas)
  - [Estandarización de Variables](#estandarización-de-variables)
- [Modelado y Evaluación](#modelado-y-evaluación)
  - [Regresión Ridge NB (L2)](#regresión-ridge-nb-l2)
  - [Balanceo de Datos con SMOTE](#balanceo-de-datos-con-smote)
  - [Regresión Ridge B (L2)](#regresión-ridge-b-l2)
  - [Regresión Lasso (L1)](#regresión-lasso-l1)
  - [Modelo k-Nearest Neighbors (kNN)](#modelo-k-nearest-neighbors-knn)
  - [Curva ROC y Comparación de Modelos](#curva-roc-y-comparación-de-modelos)
- [Estructura del Código](#estructura-del-código)
- [Explicación para No Programadores](#explicación-para-no-programadores)
- [Explicación para Programadores](#explicación-para-programadores)
- [Conclusiones y Posibles Mejoras](#conclusiones-y-posibles-mejoras)
- [Licencia y Contacto](#licencia-y-contacto)

---

## Introducción

El fraude financiero es un problema crítico en el sector bancario y de pagos, y la detección temprana de transacciones anómalas puede prevenir pérdidas importantes. **Buo Ai** utiliza algoritmos de machine learning para analizar características de las transacciones y clasificar cada operación como fraudulenta o no fraudulenta, ayudando a aumentar la seguridad y la confianza en el sistema financiero.

---

## Descripción del Proyecto

El proyecto se compone de las siguientes fases:

- **Extracción y carga de datos:** Se trabaja con un dataset de transacciones financieras de Hugging Face que incluye más de 3 millones de registros y 14 variables.  
- **Preprocesamiento:** Se realizan tareas de limpieza, manejo de valores nulos, creación de variables derivadas y estandarización de variables.  
- **Análisis Descriptivo:** Se genera un reporte estadístico y se visualizan correlaciones y distribuciones, lo que permite entender la estructura de los datos.  
- **Modelado:** Se entrenan y evalúan diferentes modelos (Ridge, Lasso y kNN) con ajuste de hiperparámetros mediante Grid Search.  
- **Balanceo de Clases:** Dado el desbalance inherente en la variable objetivo, se aplica SMOTE para obtener un conjunto de entrenamiento balanceado.  
- **Evaluación:** Se calculan métricas como accuracy, precision, recall, F1-score y se generan matrices de confusión y curvas ROC para comparar el desempeño de cada modelo.

---

## Datos

- **Fuente:**  
  [credit_fraud_detection en Hugging Face](https://huggingface.co/datasets/rohan-chandrashekar/credit_fraud_detection)

- **Características:**  
  El dataset original cuenta con 14 columnas (por ejemplo, `amount`, `oldBalanceOrig`, `newBalanceOrig`, `isFraud`, etc.) y más de 3 millones de observaciones.

- **División Utilizada para el Proyecto:**  
  - **Train:** 80,000 registros  
  - **Test:** 20,000 registros
 
La base de datos se compone de 14 columnas, que son las diferentes variables y 3.713.576 de observaciones que ya están divididas en el grupo de entrenameinto, validación y prueba.

El conjunto de datos de detección de fraude crediticio disponible en [Hugging Face](https://huggingface.co/datasets/rohan-chandrashekar/credit_fraud_detection) contiene las siguientes 14 variables representadas en las columnas:

| **Columna**                  | **Descripción**                                                                                                                                                                                                 | **Tipo de Variable** |
|-------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------------|
| `amount`                      | Monto de la transacción.                                                                                                                                                                                        | Numérica (float64)   |
| `nameOrig`                    | Identificador único del cliente que inicia la transacción.                                                                                                                                                      | Categórica (int64)   |
| `oldBalanceOrig`              | Saldo de la cuenta del originador antes de la transacción.                                                                                                                                                      | Numérica (float64)   |
| `newBalanceOrig`              | Saldo de la cuenta del originador después de la transacción.                                                                                                                                                    | Numérica (float64)   |
| `nameDest`                    | Identificador único del cliente o entidad que recibe la transacción.                                                                                                                                            | Categórica (int64)   |
| `oldBalanceDest`              | Saldo de la cuenta del destinatario antes de la transacción.                                                                                                                                                    | Numérica (float64)   |
| `newBalanceDest`              | Saldo de la cuenta del destinatario después de la transacción.                                                                                                                                                  | Numérica (float64)   |
| `isFraud`                     | Indicador de si la transacción es fraudulenta (1) o no (0).                                                                                                                                                     | Binaria (int64)      |
| `isUnauthorizedOverdraft`     | Indicador de si la transacción resultó en un sobregiro no autorizado (1) o no (0).                                                                                                                              | Binaria (int64)      |
| `action__CASH_IN`             | Indicador de si la acción de la transacción es un depósito de efectivo (1) o no (0).                                                                                                                            | Binaria (int64)      |
| `action__CASH_OUT`            | Indicador de si la acción de la transacción es un retiro de efectivo (1) o no (0).                                                                                                                              | Binaria (int64)      |
| `action__DEBIT`               | Indicador de si la acción de la transacción es un débito (1) o no (0).                                                                                                                                           | Binaria (int64)      |
| `action__PAYMENT`             | Indicador de si la acción de la transacción es un pago (1) o no (0).                                                                                                                                            | Binaria (int64)      |
| `action__TRANSFER`            | Indicador de si la acción de la transacción es una transferencia (1) o no (0).                                                                                                                                   | Binaria (int64)      |


---

## Preprocesamiento de Datos

### Instalación de Librerías y Configuración del Entorno

El script instala múltiples librerías necesarias mediante `pip` (como pandas, numpy, scikit-learn, seaborn, matplotlib y otras). Además, se configura la semilla (SEED = 42) para asegurar la reproducibilidad.

### Carga y División de Datos

- Se carga el dataset directamente desde Hugging Face usando rutas específicas para train y test.
- Los datos se cargan en DataFrames de pandas y se imprime su tamaño y una vista previa.

### Análisis Descriptivo y Exploratorio

- Se utiliza **ydata_profiling** para generar reportes HTML que resumen las estadísticas de cada conjunto.
- Se construyen matrices de correlación utilizando seaborn para identificar relaciones entre variables.
- Se analiza el desbalance de la variable objetivo (`isFraud`) mediante gráficos de barras y se calculan porcentajes.

### Limpieza y Creación de Variables Derivadas

- Se eliminan observaciones con valores nulos en la variable `isFraud`.
- Se separan las variables de entrada (X) de la variable objetivo (y).
- Se crean nuevas variables derivadas para capturar información relevante (por ejemplo, `balance_change_Orig`).
- Se eliminan columnas redundantes, como `oldBalanceOrig`, `newBalanceOrig` y `newBalanceDest` (esta última se descarta al ser idéntica a otra variable).

### Estandarización de Variables

- Se definen tres tipos de variables:
  - **Continuas:** (por ejemplo, `amount`, `balance_change_Orig`, `oldBalanceDest`)
  - **Binarias:** (por ejemplo, `action__CASH_IN`, `action__CASH_OUT`, etc.)
  - **Identificadores:** (`nameOrig`, `nameDest`)
- Solo las variables continuas se estandarizan utilizando `StandardScaler`, mientras que las variables binarias e identificadoras se mantienen sin cambios para preservar su significado.

---

## Modelado y Evaluación

Se implementan y evalúan tres modelos principales:

### Regresión Ridge NB (L2) 

- Se utiliza `LogisticRegression` con penalización L2.
- Se realiza una búsqueda de hiperparámetros (valor de `C`) mediante `GridSearchCV`.
- Se evalúa el modelo en los conjuntos de entrenamiento y prueba calculando métricas como accuracy, precision, recall y F1-score.
- Se generan matrices de confusión para visualizar los resultados.

### Balanceo de Datos con SMOTE

- Debido al desbalance en la variable `isFraud`, se aplica SMOTE para balancear el conjunto de entrenamiento.
- Se entrena nuevamente el modelo Ridge con los datos balanceados y se evalúa su desempeño.

### Regresión Ridge B (L2)

- Se utiliza `LogisticRegression` con penalización L2.
- Se realiza una búsqueda de hiperparámetros (valor de `C`) mediante `GridSearchCV`.
- Se evalúa el modelo en los conjuntos de entrenamiento y prueba calculando métricas como accuracy, precision, recall y F1-score.
- Se generan matrices de confusión para visualizar los resultados.
- Se evidencia que empeoran los resultados con respecto a la primera regresión Ridge

### Regresión Lasso (L1)

- Se entrena un modelo de `LogisticRegression` con penalización L1 (Lasso) y se realiza un ajuste de hiperparámetros similar.
- Se calculan las mismas métricas y se presentan las matrices de confusión.

### Modelo k-Nearest Neighbors (kNN)

- Se implementa `KNeighborsClassifier` y se realiza Grid Search para optimizar el número de vecinos, el tipo de ponderación y la métrica de distancia.
- Se evalúa el modelo y se generan los gráficos correspondientes.

### Curva ROC y Comparación de Modelos

- Se obtiene la probabilidad de predicción de cada modelo (usando el método `predict_proba`).
- Se utiliza el área bajo la curva (AUC) para comparar el desempeño, y se traza un gráfico con diferentes colores para cada modelo.

---

## Estructura del Código


El proyecto se organiza de forma modular para mantener el código limpio y fácil de mantener.  
Una estructura sugerida es la siguiente:

```
Buo-Ai/
├── data/
│   ├── train.csv         # Conjunto de entrenamiento (2.6 millones de registros)
│   └── test.csv          # Conjunto de prueba (557,000 registros)
├── notebooks/            # Notebooks para exploración y análisis de datos
├── src/
│   ├── preprocessing.py  # Funciones de carga, limpieza, creación de variables y estandarización
│   ├── eda.py            # Scripts para análisis exploratorio y generación de reportes
│   ├── models.py         # Implementación y ajuste de modelos (Ridge, Lasso y kNN)
│   └── main.py           # Script principal que integra todo el pipeline
├── requirements.txt      # Dependencias del proyecto
└── README.md             # Este archivo
```




Cada módulo está debidamente comentado y documentado para facilitar tanto su comprensión como futuras mejoras.

---

## Explicación para No Programadores

Esta sección explica el proyecto de forma sencilla:

- **Objetivo:**  
  Detectar transacciones financieras sospechosas o fraudulentas para ayudar a prevenir pérdidas económicas y mejorar la seguridad de los sistemas de pago.

- **Proceso General:**  
  1. **Importación de Datos:**  
     Se obtiene un gran conjunto de datos reales que contiene información de transacciones, como montos, saldos y tipos de acción (depósito, retiro, transferencia, etc.).
  2. **Preparación de la Información:**  
     - Se limpian los datos, eliminando registros erróneos o duplicados.
     - Se crean nuevas variables que permiten identificar cambios en los saldos.
     - Se “normalizan” (estandarizan) los valores numéricos para que sean comparables.
  3. **Análisis y Modelado:**  
     Se usan algoritmos matemáticos que aprenden a identificar patrones de fraude a partir de los datos. Una vez entrenados, estos modelos pueden predecir si una transacción es fraudulenta.
  4. **Evaluación:**  
     Se comparan los resultados obtenidos con distintos modelos y se elige el que mejor identifica el fraude, utilizando gráficos y métricas para medir el desempeño.

- **Importancia:**  
  Este análisis es fundamental para que las instituciones financieras puedan detectar de forma temprana actividades sospechosas y evitar fraudes, protegiendo el dinero de los clientes.

---

## Explicación para Programadores

Para los usuarios con conocimientos técnicos, se detallan los aspectos claves:

- **Librerías y Herramientas:**  
  Se utilizan librerías estándar de Python (pandas, numpy, scikit-learn, seaborn, matplotlib) junto con herramientas como ydata_profiling y SMOTE (de imbalanced-learn).

- **Pipeline de Preprocesamiento:**  
  - **Carga y División:** Se importa el dataset directamente desde Hugging Face, aprovechando que ya viene dividido en train y test.
  - **Limpieza y Feature Engineering:**  
    - Se eliminan observaciones con valores nulos en la variable objetivo.
    - Se crea la variable `balance_change_Orig` y se eliminan columnas redundantes.
  - **Estandarización:**  
    - Solo se estandarizan las variables numéricas para garantizar una escala uniforme.
    - Se preservan las variables binarias e identificadoras.
    
- **Modelado:**  
  Se emplea `LogisticRegression` con penalizaciones L1 y L2, así como `KNeighborsClassifier`. Se utiliza GridSearchCV para ajustar hiperparámetros (por ejemplo, el parámetro `C` en los modelos de regresión y el número de vecinos en kNN).  
  Además, se aplica SMOTE para abordar el desbalance de clases en el conjunto de entrenamiento, lo que mejora la robustez del modelo.
  
- **Evaluación:**  
  Se calculan métricas como accuracy, precision, recall y F1-score. También se generan matrices de confusión y se traza la curva ROC para comparar el desempeño entre modelos.

---

## Conclusiones y Posibles Mejoras

El proyecto **Buo Ai** demuestra la viabilidad de utilizar técnicas de *machine learning* para la detección de fraude financiero. Aun así, existen diversas áreas en las que se puede seguir mejorando:

- **Optimización de Hiperparámetros:**  
  Ampliar la búsqueda (por ejemplo, usando GridSearchCV o RandomizedSearchCV) para cada modelo, a fin de mejorar la precisión y reducir los falsos positivos.

- **Modelos Avanzados:**  
  Evaluar algoritmos más complejos como Random Forest, Gradient Boosting o redes neuronales para aumentar la capacidad de detección de fraudes.

- **Interpretabilidad del Modelo:**  
  Implementar herramientas como [SHAP](https://github.com/slundberg/shap) o [LIME](https://github.com/marcotcr/lime) para comprender mejor las decisiones de los modelos y reforzar la confianza en sus predicciones.

- **Automatización y Despliegue (MLOps):**  
  Integrar el pipeline de entrenamiento y predicción en una plataforma de ML Ops que facilite el versionado, monitoreo y despliegue continuo del sistema.
---
## Licencia y Contacto

- **Licencia:**  
  Este proyecto se distribuye bajo la [MIT License](LICENSE).

- **Contacto:**  
  Para cualquier duda, sugerencia o colaboración, puedes escribir a [Info@BUOAI.com] o visitar nuestra pagina (https://buoAI.com.co).

