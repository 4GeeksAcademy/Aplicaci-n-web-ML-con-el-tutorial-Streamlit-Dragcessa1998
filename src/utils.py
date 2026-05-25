from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "iris.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
MODEL_PATH = MODEL_DIR / "iris_classifier.joblib"
METRICS_PATH = MODEL_DIR / "iris_metrics.json"
RANDOM_STATE = 42

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_COLUMN = "species"


def load_iris_dataframe() -> pd.DataFrame:
    """Load the UCI Iris dataset as a clean DataFrame."""
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df.columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(dict(enumerate(iris.target_names)))
    return df


def build_model(random_state: int = RANDOM_STATE, **classifier_params: Any) -> Pipeline:
    """Create a compact classification pipeline suitable for web inference."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(random_state=random_state, **classifier_params),
            ),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_save_model() -> dict[str, Any]:
    """Train, optimize, persist the model and return a metrics dictionary."""
    df = load_iris_dataframe()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat([X_train, y_train], axis=1).to_csv(PROCESSED_DIR / "train.csv", index=False)
    pd.concat([X_test, y_test], axis=1).to_csv(PROCESSED_DIR / "test.csv", index=False)

    baseline_model = build_model()
    baseline_model.fit(X_train, y_train)

    search = GridSearchCV(
        estimator=build_model(),
        param_grid={
            "classifier__n_estimators": [80, 120, 180],
            "classifier__max_depth": [2, 3, None],
            "classifier__min_samples_leaf": [1, 2],
        },
        cv=5,
        scoring="accuracy",
        n_jobs=1,
    )
    search.fit(X_train, y_train)
    optimized_model = search.best_estimator_

    feature_importance = get_feature_importance(optimized_model)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": optimized_model,
        "feature_columns": FEATURE_COLUMNS,
        "class_names": sorted(y.unique().tolist()),
        "dataset_source": "UCI Iris Dataset via scikit-learn",
    }
    joblib.dump(artifact, MODEL_PATH)

    results = {
        "dataset": {
            "source": "UCI Iris Dataset, loaded with sklearn.datasets.load_iris",
            "rows": int(df.shape[0]),
            "features": FEATURE_COLUMNS,
            "target": TARGET_COLUMN,
            "classes": sorted(y.unique().tolist()),
            "train_rows": int(X_train.shape[0]),
            "test_rows": int(X_test.shape[0]),
        },
        "baseline_model": evaluate_model(baseline_model, X_test, y_test),
        "optimized_model": {
            "best_params": search.best_params_,
            "best_cv_accuracy": round(float(search.best_score_), 4),
            **evaluate_model(optimized_model, X_test, y_test),
        },
        "feature_importance": feature_importance,
        "artifacts": {
            "model_path": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
            "metrics_path": str(METRICS_PATH.relative_to(PROJECT_ROOT)),
            "raw_data_path": str(RAW_DATA_PATH.relative_to(PROJECT_ROOT)),
            "train_path": str((PROCESSED_DIR / "train.csv").relative_to(PROJECT_ROOT)),
            "test_path": str((PROCESSED_DIR / "test.csv").relative_to(PROJECT_ROOT)),
            "figures_dir": str(REPORTS_DIR.relative_to(PROJECT_ROOT)),
        },
    }

    write_json(results, METRICS_PATH)
    create_visualizations(df, feature_importance)
    return results


def get_feature_importance(model: Pipeline) -> list[dict[str, Any]]:
    classifier = model.named_steps["classifier"]
    return [
        {"feature": feature, "importance": round(float(importance), 4)}
        for feature, importance in sorted(
            zip(FEATURE_COLUMNS, classifier.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]


def create_visualizations(df: pd.DataFrame, feature_importance: list[dict[str, Any]]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x=TARGET_COLUMN, hue=TARGET_COLUMN, palette="Set2", legend=False)
    plt.title("Distribucion de especies")
    plt.xlabel("Especie")
    plt.ylabel("Registros")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "species_distribution.png", dpi=160)
    plt.close()

    importance_df = pd.DataFrame(feature_importance)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=importance_df, x="importance", y="feature", color="#256f68")
    plt.title("Importancia de variables")
    plt.xlabel("Importancia")
    plt.ylabel("Variable")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "feature_importance.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="petal_length_cm",
        y="petal_width_cm",
        hue=TARGET_COLUMN,
        palette="Set2",
        s=80,
    )
    plt.title("Separacion por medidas del petalo")
    plt.xlabel("Longitud del petalo (cm)")
    plt.ylabel("Ancho del petalo (cm)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "petal_scatter.png", dpi=160)
    plt.close()


def load_model_artifact() -> dict[str, Any]:
    if not MODEL_PATH.exists():
        train_and_save_model()
    return joblib.load(MODEL_PATH)


def predict_species(artifact: dict[str, Any], features: dict[str, float]) -> dict[str, Any]:
    model: Pipeline = artifact["model"]
    class_names: list[str] = artifact["class_names"]
    row = pd.DataFrame([[features[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    probabilities = model.predict_proba(row)[0]
    prediction = model.predict(row)[0]

    return {
        "prediction": str(prediction),
        "probabilities": {
            class_name: round(float(probability), 4)
            for class_name, probability in zip(class_names, probabilities)
        },
    }


def read_metrics() -> dict[str, Any]:
    if not METRICS_PATH.exists():
        return train_and_save_model()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
