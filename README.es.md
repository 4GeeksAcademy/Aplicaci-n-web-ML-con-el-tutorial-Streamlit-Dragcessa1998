# Aplicacion web de ML con Streamlit

![Banner del proyecto](reports/figures/streamlit_banner.png)

**Idioma / Language:** Español | [English](README.md)

Este proyecto convierte el clasificador de especies Iris en una aplicacion web interactiva desarrollada con **Streamlit**. La interfaz permite introducir las medidas de una flor, consultar la especie estimada y revisar la probabilidad de cada clase.

## Objetivo

- Reutilizar un modelo de Machine Learning entrenado previamente.
- Construir una interfaz web con Streamlit.
- Mostrar metricas, probabilidades y datos del modelo de forma clara.
- Preparar el repositorio para despliegue en Streamlit Community Cloud y Render.

## Dataset

Se utiliza el **UCI Iris Dataset**, cargado con `sklearn.datasets.load_iris`.

Variables predictoras:

- `sepal_length_cm`
- `sepal_width_cm`
- `petal_length_cm`
- `petal_width_cm`

Variable objetivo:

- `species`

## Modelo

El pipeline de Machine Learning usa:

- `StandardScaler`
- `RandomForestClassifier`
- `GridSearchCV` para optimizar hiperparametros

Resultados principales:

- Accuracy optimizado: `0.933`
- F1 macro optimizado: `0.933`

El modelo entrenado se guarda en:

```text
models/iris_classifier.joblib
```

## Aplicacion Streamlit

La aplicacion principal esta en:

```text
src/app.py
```

La app incluye:

- Sliders para introducir medidas botanicas.
- Prediccion de especie Iris.
- Grafico de probabilidades por clase.
- Metricas del modelo.
- Importancia de variables.
- Vista del dataset.

## Estructura del proyecto

```text
.
├── .streamlit/config.toml
├── data/
│   ├── raw/iris.csv
│   └── processed/
│       ├── train.csv
│       └── test.csv
├── models/
│   ├── iris_classifier.joblib
│   └── iris_metrics.json
├── reports/figures/
│   ├── feature_importance.png
│   ├── petal_scatter.png
│   ├── species_distribution.png
│   └── streamlit_banner.png
├── src/
│   ├── app.py
│   ├── explore.ipynb
│   ├── train_model.py
│   └── utils.py
├── Procfile
├── render.yaml
└── requirements.txt
```

## Ejecutar localmente

Instala dependencias:

```bash
pip install -r requirements.txt
```

Entrena o regenera el modelo:

```bash
python src/train_model.py
```

Ejecuta la aplicacion:

```bash
streamlit run src/app.py
```

## Despliegue en Streamlit Community Cloud

1. Entra en [share.streamlit.io](https://share.streamlit.io/).
2. Conecta tu cuenta de GitHub.
3. Selecciona este repositorio.
4. Usa como archivo principal:

```text
src/app.py
```

5. Streamlit instalara las dependencias desde `requirements.txt`.

URL de Streamlit:

```text
Pendiente de pegar despues del despliegue.
```

## Despliegue en Render

Este repositorio incluye `render.yaml` y `Procfile`.

Configuracion esperada:

- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run src/app.py --server.port $PORT --server.address 0.0.0.0`

URL de Render:

```text
Pendiente de pegar despues del despliegue.
```

## Recursos externos

- UCI Iris Dataset via scikit-learn.
- Streamlit documentation.
- Render Web Services documentation.
- Streamlit Community Cloud.
