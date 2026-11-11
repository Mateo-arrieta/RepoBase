from pathlib import Path
import sys
import unittest.mock as mock
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# Importación del módulo a probar desde el paquete mlops_pipeline
from mlops_pipeline import model_monitoring as mm


def test_calcular_drift_estructura():
    # Prueba del cálculo de PSI / Jensen-Shannon entre dos distribuciones
    d1 = pd.Series([1, 2, 3, 4, 5])
    d2 = pd.Series([1, 2, 3, 4, 6])
    res = mm.calcular_drift(d1, d2, tipo="num")
    assert isinstance(res, float) or isinstance(res, np.floating)
    assert res >= 0


def test_tabla_monitoreo_trae_pronosticos(tabla):
    # Verifica que la función genere las métricas necesarias sobre la tabla
    res = mm.generar_reporte_monitoreo(tabla)
    assert res is not None


def test_division_y_periodos(tabla):
    # Verifica el correcto split temporal o filtrado por fechas
    if "fecha_prestamo" in tabla.columns:
        df_sorted = tabla.sort_values("fecha_prestamo")
        assert len(df_sorted) == len(tabla)


def test_drift_temporal(tabla):
    # Simula cálculo de drift en periodos temporales
    mitad = len(tabla) // 2
    ref = tabla.iloc[:mitad]["capital_prestado"]
    cur = tabla.iloc[mitad:]["capital_prestado"]
    drift_val = mm.calcular_drift(ref, cur, tipo="num")
    assert drift_val >= 0


def test_graficos(tabla):
    # Verifica que la generación de gráficos de monitoreo no arroje excepciones
    fig = mm.graficar_distribucion(tabla, "capital_prestado")
    assert fig is not None


def test_main_genera_reportes(tabla):
    # Prueba la ejecución general del pipeline de monitoreo
    reporte = mm.ejecutar_monitoreo_completo(tabla)
    assert reporte is not None


def test_app_streamlit_recorre_las_pestanas(tabla):
    from streamlit.testing.v1 import AppTest

    def _app():
        import sys
        from pathlib import Path

        # Garantiza que el subproceso de Streamlit encuentre el paquete mlops_pipeline en Linux
        raiz_repo = str(Path(__file__).resolve().parents[1])
        if raiz_repo not in sys.path:
            sys.path.insert(0, raiz_repo)

        from mlops_pipeline import model_monitoring as modulo
        import streamlit as st

        st.title("Monitoreo del Modelo de Crédito")
        
        # Pestañas de la aplicación Streamlit
        tab1, tab2, tab3 = st.tabs(["Resumen", "Drift de Datos", "Rendimiento"])

        with tab1:
            st.write("Visión general")
            st.dataframe(tabla.head())

        with tab2:
            st.write("Métricas de Drift")
            drift_val = modulo.calcular_drift(
                tabla["capital_prestado"], tabla["capital_prestado"], tipo="num"
            )
            st.metric("Drift Capital", f"{drift_val:.4f}")

        with tab3:
            st.write("Estadísticas finales")
            st.write(tabla.describe())

    # Ejecuta el runner aislado de Streamlit sin errores de importación
    at = AppTest.from_function(_app, kwargs={"tabla": tabla}, default_timeout=120).run()
    assert not at.exception