import numpy as np
import pandas as pd
import pytest

from mlops_pipeline import model_monitoring as mm


@pytest.fixture(scope="module")
def tabla(ruta_csv, ruta_modelo):
    return mm.cargar_tabla_monitoreo(ruta=ruta_csv, ruta_modelo=ruta_modelo)


# --- métricas -------------------------------------------------------------
def test_psi_y_js_distribuciones_iguales_y_distintas():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert mm.psi(p, p) == pytest.approx(0, abs=1e-9)
    assert mm.js_distance(p, p) == pytest.approx(0, abs=1e-9)
    q = np.array([0.7, 0.1, 0.1, 0.1])
    assert mm.psi(p, q) > mm.UMBRAL_PSI_CRITICO
    assert 0 < mm.js_distance(p, q) <= 1


@pytest.mark.parametrize("valor, esperado", [(0.05, 0), (0.15, 1), (0.30, 2)])
def test_nivel_psi(valor, esperado):
    assert mm.nivel_psi(valor) == esperado


def test_nivel_psi_nulo():
    assert pd.isna(mm.nivel_psi(np.nan))


def test_proporciones_numericas_continuas_y_discretas():
    rng = np.random.default_rng(0)
    p, q = mm._proporciones_numericas(pd.Series(rng.normal(size=500)), pd.Series(rng.normal(size=300)))
    assert p.sum() == pytest.approx(1) and q.sum() == pytest.approx(1)
    p, q = mm._proporciones_numericas(pd.Series([0, 1, 1, 0]), pd.Series([1, 1, 2]))
    assert len(p) == len(q) == 3


def test_drift_numerico_detecta_desplazamiento():
    rng = np.random.default_rng(1)
    ref = pd.Series(rng.normal(0, 1, 1000))
    estable = mm.drift_numerico(ref, pd.Series(rng.normal(0, 1, 800)))
    movido = mm.drift_numerico(ref, pd.Series(rng.normal(2, 1, 800)))
    assert estable["psi"] < mm.UMBRAL_PSI_ALERTA
    assert movido["psi"] > mm.UMBRAL_PSI_CRITICO
    assert movido["ks"] > estable["ks"]
    assert mm.drift_numerico(pd.Series([1.0]), pd.Series([np.nan, 2.0])) == {}


def test_drift_categorico_agrupa_raras_y_nulos():
    ref = pd.Series(["a"] * 600 + ["b"] * 395 + ["rara"] * 5)
    act = pd.Series(["a"] * 100 + ["b"] * 100 + [None] * 50 + ["otra"] * 5)
    res = mm.drift_categorico(ref, act)
    assert {"chi2", "chi2_p", "psi", "js"} <= set(res)
    assert res["psi"] > 0
    assert mm.drift_categorico(pd.Series([], dtype=object), act) == {}
    una = mm.drift_categorico(pd.Series(["a"] * 10), pd.Series(["a"] * 10))
    assert np.isnan(una["chi2"])


def test_calcular_drift_estructura(tabla):
    num, cat = mm.definir_variables(tabla)
    ref, act = mm.dividir_referencia_actual(tabla, "2025-06-01")
    drift = mm.calcular_drift(ref, act, num, cat)
    assert len(drift) == len(num) + len(cat)
    assert mm.COL_PROBA in drift["variable"].values
    assert "mes_prestamo" not in drift["variable"].values
    assert drift["estado"].isin(mm.SEMAFORO.values()).all()


# --- datos y ventanas -------------------------------------------------------
def test_tabla_monitoreo_trae_pronosticos(tabla):
    assert tabla[mm.COL_PROBA].between(0, 1).all()
    assert tabla[mm.COL_FECHA].is_monotonic_increasing
    assert "puntaje" not in tabla.columns


def test_division_y_periodos(tabla):
    ref, act = mm.dividir_referencia_actual(tabla, "2025-06-01")
    assert len(ref) + len(act) == len(tabla)
    assert ref[mm.COL_FECHA].max() < act[mm.COL_FECHA].min()
    assert asignar(act, "M").nunique() > asignar(act, "Q").nunique()


def asignar(df, freq):
    return mm.asignar_periodo(df, freq)


def test_drift_temporal(tabla):
    num, cat = mm.definir_variables(tabla)
    ref, act = mm.dividir_referencia_actual(tabla, "2025-06-01")
    temporal = mm.drift_temporal(tabla, ref, act, num, cat, "Q")
    assert {"periodo", "n", "muestra_suficiente", "psi"} <= set(temporal.columns)
    assert temporal["periodo"].nunique() == asignar(act, "Q").nunique()
    assert mm.drift_temporal(tabla, ref, act.iloc[0:0], num, cat).empty


def test_detectar_cambios():
    serie = pd.Series([0.02, 0.03, 0.05, 0.30], index=["a", "b", "c", "d"])
    mensajes = mm.detectar_cambios(serie)
    assert any("Tendencia creciente" in m for m in mensajes)
    assert any("abrupto" in m for m in mensajes)
    assert mm.detectar_cambios(pd.Series([0.02, 0.02])) == []


# --- alertas y recomendaciones ----------------------------------------------
def _drift(**filas):
    base = []
    for nombre, (psi_valor, nulos_act) in filas.items():
        base.append({
            "variable": nombre, "psi": psi_valor, "js": 0.2, "ks": 0.3, "chi2_p": np.nan,
            "nivel": mm.nivel_psi(psi_valor), "nulos_ref": 0.0, "nulos_act": nulos_act,
        })
    return pd.DataFrame(base)


def test_alertas_criticas_amarillas_y_nulos():
    alertas = mm.generar_alertas(_drift(a=(0.4, 0.0), b=(0.15, 0.0), c=(0.01, 0.3)))
    niveles = [n for n, _ in alertas]
    assert niveles.count("error") == 1 and niveles.count("warning") == 2


def test_alertas_sin_desviaciones():
    assert mm.generar_alertas(_drift(a=(0.01, 0.0)))[0][0] == "success"


def test_recomendaciones():
    estable = mm.generar_recomendaciones(_drift(a=(0.01, 0.0)))
    assert "No se requiere acción" in estable[0]
    muchas = {f"v{i}": (0.01, 0.0) for i in range(6)}
    revisar = " ".join(mm.generar_recomendaciones(_drift(a=(0.4, 0.0), c=(0.15, 0.3), **muchas)))
    assert "Revisa estas variables" in revisar and "captura de datos" in revisar
    reentrenar = " ".join(mm.generar_recomendaciones(_drift(a=(0.4, 0.0), b=(0.01, 0.0))))
    assert "reentrenar" in reentrenar
    pronostico = mm.generar_recomendaciones(_drift(**{mm.COL_PROBA: (0.5, 0.0)}))
    assert "pronósticos" in " ".join(pronostico)


# --- gráficos y consola --------------------------------------------------------
def test_graficos(tabla):
    num, cat = mm.definir_variables(tabla)
    ref, act = mm.dividir_referencia_actual(tabla, "2025-06-01")
    assert mm.plot_distribucion(ref[num[0]], act[num[0]], num[0], True) is not None
    assert mm.plot_distribucion(ref[cat[0]], act[cat[0]], cat[0], False) is not None
    temporal = mm.drift_temporal(tabla, ref, act, num, cat, "M")
    assert mm.plot_evolucion(temporal, num[:2]) is not None


def test_main_genera_reportes(tabla, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(mm, "cargar_tabla_monitoreo", lambda: tabla)
    monkeypatch.chdir(tmp_path)
    drift, temporal = mm.main(["--corte", "2025-06-01", "--freq", "Q", "--guardar"])
    assert (tmp_path / "reporte_drift.csv").exists()
    assert (tmp_path / "reporte_drift_temporal.csv").exists()
    assert "Alertas" in capsys.readouterr().out
    assert not drift.empty and not temporal.empty


# --- aplicación Streamlit -----------------------------------------------------
def _app(tabla):
    import model_monitoring as modulo

    modulo.cargar_tabla_monitoreo = lambda: tabla
    modulo.run_app()


def test_app_streamlit_recorre_las_pestanas(tabla):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_function(_app, kwargs={"tabla": tabla}, default_timeout=120).run()
    assert not at.exception
    assert len(at.tabs) == 5
    assert at.metric[0].value == f"{(tabla[mm.COL_FECHA] < '2025-06-01').sum():,}"

    num, cat = mm.definir_variables(tabla)
    variable = next(s for s in at.selectbox if s.label == "Variable")
    variable.select(cat[0]).run()          # gráfico categórico
    assert not at.exception

    ventana = next(s for s in at.selectbox if s.label == "Ventana a evaluar")
    ventana.select(ventana.options[1]).run()  # un período puntual
    assert not at.exception