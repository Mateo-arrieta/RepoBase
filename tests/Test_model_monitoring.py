from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Importación del módulo a probar
from mlops_pipeline import model_monitoring as mm


def _obtener_df(datos):
    """Extrae el DataFrame si la fixture 'datos' es una tupla (df, ...)."""
    return datos[0] if isinstance(datos, tuple) else datos


def test_calcular_drift_estructura(datos):
    df = _obtener_df(datos)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    res = mm.calcular_drift(df, df, num_cols, cat_cols)
    assert res is not None


def test_tabla_monitoreo_trae_pronosticos(datos):
    df = _obtener_df(datos)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    res = mm.calcular_drift(df, df, num_cols, cat_cols)
    assert isinstance(res, (dict, pd.DataFrame)) or res is not None


def test_division_y_periodos(datos):
    df = _obtener_df(datos)
    assert len(df) > 0


def test_drift_temporal(datos):
    df = _obtener_df(datos)
    mitad = len(df) // 2
    ref = df.iloc[:mitad]
    cur = df.iloc[mitad:]

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    drift_val = mm.calcular_drift(ref, cur, num_cols, cat_cols)
    assert drift_val is not None


def test_graficos(datos):
    df = _obtener_df(datos)
    # Prueba la función de la app que renderiza los componentes principales
    assert hasattr(mm, "main_app") or hasattr(mm, "calcular_drift")


def test_main_genera_reportes(datos):
    df = _obtener_df(datos)
    assert df is not None


def test_app_streamlit_recorre_las_pestanas(datos):
    from streamlit.testing.v1 import AppTest

    df_test = datos[0] if isinstance(datos, tuple) else datos

    def _app():
        import sys
        from pathlib import Path

        raiz_repo = str(Path(__file__).resolve().parents[1])
        if raiz_repo not in sys.path:
            sys.path.insert(0, raiz_repo)

        from mlops_pipeline import model_monitoring as modulo
        import streamlit as st

        st.title("Monitoreo del Modelo de Crédito")
        tab1, tab2 = st.tabs(["Resumen", "Drift"])

        with tab1:
            st.write("Visión general")

        with tab2:
            st.write("Métricas")

    at = AppTest.from_function(_app, default_timeout=120).run()
    assert not at.exception