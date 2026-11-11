import joblib
import numpy as np
import pandas as pd
import pytest

from mlops_pipeline import model_training_evaluation as mt


def test_summarize_classification_prediccion_perfecta():
    y = np.array([0, 0, 1, 1, 1])
    metricas = mt.summarize_classification(y, y, y.astype(float))
    assert metricas["accuracy"] == 1
    assert metricas["recall_clase0"] == 1
    assert metricas["roc_auc"] == 1
    assert set(metricas) >= {"balanced_accuracy", "precision_clase0", "f1_clase0"}


def test_normalizar():
    assert list(mt._normalizar(pd.Series([1.0, 2.0, 3.0]))) == [0, 0.5, 1]
    assert (mt._normalizar(pd.Series([2.0, 2.0])) == 0).all()


def test_rank_models_prioriza_performance_consistencia_y_velocidad():
    tabla = pd.DataFrame({
        "modelo": ["bueno", "malo"], "roc_auc": [0.8, 0.6],
        "cv_auc_std": [0.01, 0.05], "tiempo_entrenamiento_s": [0.1, 5.0],
    })
    ranking = mt.rank_models(tabla)
    assert ranking.loc[0, "modelo"] == "bueno"
    assert ranking.loc[0, "score_final"] == pytest.approx(1.0)


def test_get_models_incluye_los_cuatro_modelos():
    assert len(mt.get_models()) == 4


def test_build_model_retorna_pipeline_y_metricas(datos, monkeypatch):
    monkeypatch.setattr(mt, "CV_FOLDS", 2)
    X_train, X_test, y_train, y_test, prep = datos
    nombre, estimador = list(mt.get_models().items())[0]
    pipe, resultado, proba = mt.build_model(nombre, estimador, prep, X_train, y_train, X_test, y_test)
    assert resultado["modelo"] == nombre
    assert 0.5 < resultado["roc_auc"] <= 1
    assert len(proba) == len(X_test)
    assert pipe.predict(X_test).shape == (len(X_test),)


def test_graficos_y_flujo_principal(datos, monkeypatch, tmp_path):
    modelos = mt.get_models()
    reducidos = {k: modelos[k] for k in ["Regresión Logística", "Árbol de Decisión"]}
    monkeypatch.setattr(mt, "get_models", lambda: reducidos)
    monkeypatch.setattr(mt, "CV_FOLDS", 2)
    monkeypatch.setattr(mt, "MOSTRAR_GRAFICOS", False)
    monkeypatch.setattr(mt, "RUTA_MODELO", tmp_path / "modelo.joblib")
    monkeypatch.setattr(mt, "get_data", lambda: datos)

    mejor, tabla = mt.main()
    assert len(tabla) == 2
    assert tabla["score_final"].is_monotonic_decreasing
    assert (tmp_path / "modelo.joblib").exists()
    assert joblib.load(tmp_path / "modelo.joblib").predict_proba(datos[1]).shape[0] == len(datos[1])


def test_mostrar_graficos_activo_llama_a_show(monkeypatch):
    llamadas = []
    monkeypatch.setattr(mt, "MOSTRAR_GRAFICOS", True)
    monkeypatch.setattr(mt.plt, "show", lambda: llamadas.append(1))
    mt._cerrar_o_mostrar()
    assert llamadas == [1]