from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (  # noqa: E402
    FEATURE_COLUMNS,
    MODEL_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    load_iris_dataframe,
    load_model_artifact,
    predict_species,
    read_metrics,
)


FIELD_LABELS = {
    "sepal_length_cm": "Longitud del sepalo (cm)",
    "sepal_width_cm": "Ancho del sepalo (cm)",
    "petal_length_cm": "Longitud del petalo (cm)",
    "petal_width_cm": "Ancho del petalo (cm)",
}

DEFAULT_VALUES = {
    "sepal_length_cm": 5.8,
    "sepal_width_cm": 3.0,
    "petal_length_cm": 4.35,
    "petal_width_cm": 1.3,
}


st.set_page_config(
    page_title="Iris ML Streamlit App",
    page_icon="🌿",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.65rem; }
    .app-subtitle { color: #51615f; font-size: 1.05rem; margin-top: -0.5rem; }
    .result-card {
        border: 1px solid #d8e5e0;
        border-radius: 8px;
        padding: 1rem 1.1rem;
        background: #f6fbf8;
    }
    .small-note { color: #61706d; font-size: 0.88rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifact() -> dict:
    return load_model_artifact()


@st.cache_data
def get_metrics() -> dict:
    return read_metrics()


@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_iris_dataframe()


artifact = get_artifact()
metrics = get_metrics()
df = get_dataset()

st.title("Iris Species Predictor")
st.markdown(
    "<p class='app-subtitle'>Aplicacion web con Streamlit para usar un modelo de Machine Learning entrenado con el dataset Iris.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Medidas de entrada")
    st.caption("Ajusta las cuatro medidas botanicas y ejecuta la prediccion.")

    input_values = {}
    for feature in FEATURE_COLUMNS:
        min_value = float(df[feature].min())
        max_value = float(df[feature].max())
        default_value = float(DEFAULT_VALUES[feature])
        input_values[feature] = st.slider(
            FIELD_LABELS[feature],
            min_value=min_value,
            max_value=max_value,
            value=default_value,
            step=0.1,
        )

    predict_button = st.button("Predecir especie", type="primary", use_container_width=True)

tab_prediction, tab_model, tab_data = st.tabs(["Prediccion", "Modelo", "Dataset"])

with tab_prediction:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        st.subheader("Perfil ingresado")
        st.dataframe(
            pd.DataFrame([input_values]).rename(columns=FIELD_LABELS),
            use_container_width=True,
            hide_index=True,
        )

        if predict_button:
            result = predict_species(artifact, input_values)
        else:
            result = predict_species(artifact, input_values)

        st.markdown(
            f"""
            <div class="result-card">
                <p class="small-note">Especie estimada</p>
                <h2>{result["prediction"].title()}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Probabilidad por clase")
        probabilities = (
            pd.DataFrame(
                result["probabilities"].items(),
                columns=["species", "probability"],
            )
            .sort_values("probability", ascending=False)
            .reset_index(drop=True)
        )
        st.bar_chart(probabilities, x="species", y="probability", height=300)
        st.dataframe(probabilities, use_container_width=True, hide_index=True)

with tab_model:
    st.subheader("Rendimiento del modelo")
    baseline = metrics["baseline_model"]
    optimized = metrics["optimized_model"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy base", f"{baseline['accuracy']:.3f}")
    col2.metric("F1 macro base", f"{baseline['macro_f1']:.3f}")
    col3.metric("Accuracy optimizado", f"{optimized['accuracy']:.3f}")
    col4.metric("F1 macro optimizado", f"{optimized['macro_f1']:.3f}")

    st.write("Mejores hiperparametros")
    st.json(optimized["best_params"])

    importance = pd.DataFrame(metrics["feature_importance"])
    st.write("Importancia de variables")
    st.bar_chart(importance, x="feature", y="importance", height=280)

    figure_path = REPORTS_DIR / "feature_importance.png"
    if figure_path.exists():
        st.image(str(figure_path), caption="Importancia de variables del Random Forest")

with tab_data:
    st.subheader("Dataset Iris")
    rows = metrics["dataset"]["rows"]
    classes = ", ".join(metrics["dataset"]["classes"])
    st.write(f"Fuente: {metrics['dataset']['source']}")
    st.write(f"Registros: {rows} | Clases: {classes}")

    data_col1, data_col2 = st.columns([1, 1], gap="large")
    with data_col1:
        st.write("Vista inicial")
        st.dataframe(df.head(12), use_container_width=True)
    with data_col2:
        st.write("Distribucion de clases")
        class_distribution = df[TARGET_COLUMN].value_counts().rename_axis("species").reset_index(name="records")
        st.dataframe(class_distribution, use_container_width=True, hide_index=True)
        st.bar_chart(class_distribution, x="species", y="records", height=240)

    scatter_path = REPORTS_DIR / "petal_scatter.png"
    if scatter_path.exists():
        st.image(str(scatter_path), caption="Separacion visual por medidas del petalo")

st.divider()
st.caption(
    f"Modelo cargado desde `{MODEL_PATH.relative_to(PROJECT_ROOT)}`. "
    "Aplicacion preparada para Streamlit Community Cloud y Render."
)
