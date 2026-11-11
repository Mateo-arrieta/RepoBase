import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import model_deploy as md

EJEMPLO = md.Credito.model_config["json_schema_extra"]["example"]


@pytest.fixture
def cliente(modelo_entrenado, monkeypatch):
    monkeypatch.setattr(md, "modelo", modelo_entrenado)
    return TestClient(md.app)


def _csv(df: pd.DataFrame) -> dict:
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return {"archivo": ("datos.csv", buffer.getvalue(), "text/csv")}


def test_health(cliente):
    assert cliente.get("/health").json() == {"status": "ok", "modelo_cargado": True}


def test_predict_un_registro(cliente):
    r = cliente.post("/predict", json={"registros": [EJEMPLO]})
    assert r.status_code == 200
    resultado = r.json()["resultados"][0]
    assert 0 <= resultado["prob_pago_atiempo"] <= 1
    assert resultado["prediccion"] in (0, 1)
    assert resultado["riesgo"] in ("alto", "medio", "bajo")


def test_predict_lote_con_nulos_y_umbral(cliente):
    sin_datos = dict(EJEMPLO, edad_cliente=None, salario_cliente=None, tendencia_ingresos=None)
    r = cliente.post("/predict", json={"registros": [EJEMPLO, sin_datos], "umbral": 1.0})
    assert r.json()["total"] == 2
    assert all(x["prediccion"] == 0 for x in r.json()["resultados"])  # umbral máximo


def test_predict_valida_entrada(cliente):
    assert cliente.post("/predict", json={"registros": []}).status_code == 422
    assert cliente.post("/predict", json={"registros": [{"tipo_credito": 4}]}).status_code == 422


def test_predict_csv(cliente, ruta_csv):
    df = pd.read_csv(ruta_csv).head(50)[md.COLS_ENTRADA]
    r = cliente.post("/predict/csv", files=_csv(df))
    assert r.status_code == 200 and r.json()["total"] == 50


def test_predict_csv_errores(cliente, ruta_csv):
    df = pd.read_csv(ruta_csv).head(5)[md.COLS_ENTRADA]
    assert cliente.post("/predict/csv", files=_csv(df.drop(columns=["plazo_meses"]))).status_code == 422
    assert cliente.post("/predict/csv", files=_csv(df.iloc[0:0])).status_code == 422
    assert cliente.post("/predict/csv?umbral=2", files=_csv(df)).status_code == 422
    mala_fecha = df.assign(fecha_prestamo="no-es-fecha")
    assert cliente.post("/predict/csv", files=_csv(mala_fecha)).status_code == 422
    vacio = {"archivo": ("v.csv", b"", "text/csv")}
    assert cliente.post("/predict/csv", files=vacio).status_code == 400


def test_sin_modelo_responde_503(monkeypatch):
    monkeypatch.setattr(md, "modelo", None)
    r = TestClient(md.app).post("/predict", json={"registros": [EJEMPLO]})
    assert r.status_code == 503


def test_lifespan_carga_el_modelo(modelo_entrenado, monkeypatch):
    monkeypatch.setattr(md, "modelo", None)
    monkeypatch.setattr(md, "cargar_modelo", lambda: modelo_entrenado)
    with TestClient(md.app) as c:
        assert c.get("/health").json()["modelo_cargado"] is True


def test_cargar_modelo(ruta_modelo, tmp_path):
    assert md.cargar_modelo(ruta_modelo) is not None
    with pytest.raises(FileNotFoundError):
        md.cargar_modelo(tmp_path / "no_existe.joblib")


@pytest.mark.parametrize("prob, esperado", [(0.2, "alto"), (0.5, "medio"), (0.9, "bajo")])
def test_nivel_riesgo(prob, esperado):
    assert md.nivel_riesgo(prob) == esperado