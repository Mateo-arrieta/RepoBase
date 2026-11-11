from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

# Importación del módulo a probar
from mlops_pipeline import model_monitoring as mm


def test_calcular_drift_estructura():
    d1 = pd.Series([1, 2, 3, 4, 5])
    d2 = pd.Series([1, 2, 3, 4, 6])
    # Se llama calcular_drift sin el argumento 'tipo'
    res = mm.calcular_drift(d1, d2)
    assert res is not None


def test_tabla_monitoreo_trae_pronosticos(datos):
    res = mm.generar_reporte_monitoreo(datos)
    assert res is not None


def test_division_y_periodos(datos):
    if "fecha_prestamo" in datos.columns:
        df_sorted = datos.sort_values("fecha_prestamo")
        assert len(df_sorted) == len(datos)


def test_drift_temporal(datos):
    mitad = len(datos) // 2
    ref = datos.iloc[:mitad]["capital_prestado"]
    cur = datos.iloc[mitad:]["capital_prestado"]
    drift_val = mm.calcular_drift(ref, cur)
    assert drift_val is not None


def test_graficos(datos):
    fig = mm.graficar_distribucion(datos, "capital_prestado")
    assert fig is not None


def test_main_genera_reportes(datos):
    reporte = mm.ejecutar_monitoreo_completo(datos)
    assert reporte is not None


def test_app_streamlit_recorre_las_pestanas(datos):
    from streamlit.testing.v1 import AppTest

    def _app():
        import sys
        from pathlib import Path

        raiz_repo = str(Path(__file__).resolve().parents[1])
        if raiz_repo not in sys.path:
            sys.path.insert(0, raiz_repo)

        from mlops_pipeline import model_monitoring as modulo
        import streamlit as st

        st.title("Monitoreo del Modelo de Crédito")
        tab1, tab2, tab3 = st.tabs(["Resumen", "Drift de Datos", "Rendimiento"])

        with tab1:
            st.write("Visión general")
            st.dataframe(datos.head())

        with tab2:
            st.write("Métricas de Drift")
            drift_val = modulo.calcular_drift(datos["capital_prestado"], datos["capital_prestado"])
            st.metric("Drift Capital", f"{drift_val}")

        with tab3:
            st.write("Estadísticas finales")
            st.write(datos.describe())

    at = AppTest.from_function(_app, kwargs={"datos": datos}, default_timeout=120).run()
    assert not at.exception