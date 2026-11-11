"""
model_training_evaluation.py
----------------------------
Entrena y evalúa varios modelos supervisados y selecciona el mejor según tres criterios:
    - Performance : ROC-AUC en el conjunto de evaluación.
    - Consistency : desviación estándar del ROC-AUC en validación cruzada (menor = mejor).
    - Scalability : tiempo de entrenamiento (menor = mejor).

La clase minoritaria (0 = no pagó a tiempo) es la de mayor riesgo, por eso además de la
exactitud se reportan precisión, recall y F1 de la clase 0.

Salida: el mejor modelo (pipeline completo) guardado en `mejor_modelo.joblib`.
"""

import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from mlops_pipeline.ft_engineering import RANDOM_STATE, get_data

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
MOSTRAR_GRAFICOS = True
CV_FOLDS = 5
RUTA_MODELO = Path(__file__).resolve().parent / "mejor_modelo.joblib"
PESOS = {"performance": 0.6, "consistency": 0.2, "scalability": 0.2}


def get_models() -> dict:
    """Modelos a comparar. class_weight compensa el desbalance (~95 % / 5 %)."""
    return {
        "Regresión Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Árbol de Decisión": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }


# ---------------------------------------------------------------------------
# Funciones reutilizables
# ---------------------------------------------------------------------------
def summarize_classification(y_true, y_pred, y_proba) -> dict:
    """Métricas de clasificación. y_proba = probabilidad de la clase 1."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_clase0": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_clase0": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "f1_clase0": f1_score(y_true, y_pred, pos_label=0, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def build_model(nombre, estimador, preprocessor, X_train, y_train, X_test, y_test):
    """Arma el pipeline (preprocesamiento + modelo), lo entrena y lo evalúa."""
    pasos = [("preprocessor", clone(preprocessor))]
    if isinstance(estimador, LogisticRegression):
        pasos.append(("scaler", StandardScaler()))  # los modelos lineales lo necesitan
    pasos.append(("model", estimador))
    pipe = Pipeline(pasos)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    inicio = time.perf_counter()
    pipe.fit(X_train, y_train)
    t_entrenamiento = time.perf_counter() - inicio

    inicio = time.perf_counter()
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    t_prediccion = time.perf_counter() - inicio

    resultado = {
        "modelo": nombre,
        **summarize_classification(y_test, y_pred, y_proba),
        "cv_auc_mean": cv_scores.mean(),
        "cv_auc_std": cv_scores.std(),
        "tiempo_entrenamiento_s": t_entrenamiento,
        "tiempo_prediccion_s": t_prediccion,
    }
    return pipe, resultado, y_proba


def _normalizar(serie: pd.Series) -> pd.Series:
    rango = serie.max() - serie.min()
    return (serie - serie.min()) / rango if rango else pd.Series(0.0, index=serie.index)


def rank_models(resultados: pd.DataFrame) -> pd.DataFrame:
    """Puntaje final = performance + consistency + scalability (pesos en PESOS)."""
    tabla = resultados.copy()
    tabla["score_final"] = (
        PESOS["performance"] * _normalizar(tabla["roc_auc"])
        + PESOS["consistency"] * (1 - _normalizar(tabla["cv_auc_std"]))
        + PESOS["scalability"] * (1 - _normalizar(tabla["tiempo_entrenamiento_s"]))
    )
    return tabla.sort_values("score_final", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Gráficos comparativos
# ---------------------------------------------------------------------------
def _cerrar_o_mostrar():
    if MOSTRAR_GRAFICOS:
        plt.show()
    else:
        plt.close()


def plot_metric_comparison(tabla: pd.DataFrame):
    metricas = ["roc_auc", "balanced_accuracy", "recall_clase0", "f1_clase0"]
    ax = tabla.set_index("modelo")[metricas].plot(kind="bar", figsize=(10, 5), rot=15)
    ax.set_title("Comparación de modelos (conjunto de evaluación)")
    ax.set_ylabel("Valor de la métrica")
    ax.set_ylim(0, 1)
    plt.tight_layout()
    _cerrar_o_mostrar()


def plot_roc_curves(probas: dict, y_test):
    plt.figure(figsize=(7, 6))
    for nombre, proba in probas.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Azar")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.title("Curvas ROC")
    plt.legend()
    plt.tight_layout()
    _cerrar_o_mostrar()


def plot_confusion_matrix(pipe, X_test, y_test, nombre):
    ConfusionMatrixDisplay.from_estimator(pipe, X_test, y_test, cmap="Blues")
    plt.title(f"Matriz de confusión - {nombre}")
    plt.tight_layout()
    _cerrar_o_mostrar()


# ---------------------------------------------------------------------------
# Flujo principal
# ---------------------------------------------------------------------------
def main():
    X_train, X_test, y_train, y_test, preprocessor = get_data()

    referencia = y_test.value_counts(normalize=True).max()
    print(f"Referencia: predecir siempre la clase mayoritaria da accuracy = {referencia:.4f}")

    pipelines, filas, probas = {}, [], {}
    for nombre, estimador in get_models().items():
        print(f"Entrenando {nombre}...")
        pipe, resultado, proba = build_model(
            nombre, estimador, preprocessor, X_train, y_train, X_test, y_test
        )
        pipelines[nombre], probas[nombre] = pipe, proba
        filas.append(resultado)

    tabla = rank_models(pd.DataFrame(filas))
    print("\nTabla resumen (ordenada por score_final):")
    print(tabla.round(4).to_string(index=False))

    mejor = tabla.loc[0, "modelo"]
    print(f"\nMejor modelo: {mejor}")
    print(classification_report(
        y_test, pipelines[mejor].predict(X_test),
        target_names=["No pagó a tiempo (0)", "Pagó a tiempo (1)"],
    ))

    plot_metric_comparison(tabla)
    plot_roc_curves(probas, y_test)
    plot_confusion_matrix(pipelines[mejor], X_test, y_test, mejor)

    joblib.dump(pipelines[mejor], RUTA_MODELO)
    print(f"Modelo guardado en: {RUTA_MODELO}")
    return pipelines[mejor], tabla


if __name__ == "__main__":
    main()
