"""
model_deploy.py
---------------
Despliegue del mejor modelo como servicio REST (FastAPI).

Responsabilidades:
    1. Cargar el modelo entrenado (`mejor_modelo.joblib`).
    2. Aplicar las mismas transformaciones del entrenamiento (limpieza y atributos derivados
       de ft_engineering.py); el imputado y la codificación van dentro del pipeline del modelo.
    3. Exponer el endpoint /predict (JSON) y /predict/csv (archivo CSV), ambos por lotes.

Ejecución local:
    uvicorn model_deploy:app --reload --port 8000
    Documentación interactiva: http://localhost:8000/docs
"""

import io
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

import joblib
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from mlops_pipeline.ft_engineering import crear_variables_derivadas, limpiar_datos

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_MODELO = Path(os.getenv("MODEL_PATH", Path(__file__).resolve().parent / "mejor_modelo.joblib"))
UMBRAL_DEFECTO = 0.5
MAX_REGISTROS = 10_000

COLS_NUMERICAS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal", "promedio_ingresos_datacredito",
]
COLS_ENTRADA = ["tipo_credito", "fecha_prestamo", "tipo_laboral", "tendencia_ingresos"] + COLS_NUMERICAS

modelo = None  # se carga al iniciar la aplicación


# ---------------------------------------------------------------------------
# Esquemas
# ---------------------------------------------------------------------------
class Credito(BaseModel):
    """Una solicitud de crédito. Los campos opcionales pueden venir nulos (el modelo los imputa)."""

    tipo_credito: Union[int, str]
    fecha_prestamo: datetime
    capital_prestado: float
    plazo_meses: int
    edad_cliente: Optional[float] = None
    tipo_laboral: Optional[str] = None
    salario_cliente: Optional[float] = None
    total_otros_prestamos: float = 0
    cuota_pactada: float
    puntaje_datacredito: Optional[float] = None
    cant_creditosvigentes: int = 0
    huella_consulta: int = 0
    saldo_mora: Optional[float] = None
    saldo_total: Optional[float] = None
    saldo_principal: Optional[float] = None
    saldo_mora_codeudor: Optional[float] = None
    creditos_sectorFinanciero: int = 0
    creditos_sectorCooperativo: int = 0
    creditos_sectorReal: int = 0
    promedio_ingresos_datacredito: Optional[float] = None
    tendencia_ingresos: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "tipo_credito": 4, "fecha_prestamo": "2026-01-07T14:40:36",
                "capital_prestado": 1852560, "plazo_meses": 12, "edad_cliente": 32,
                "tipo_laboral": "Empleado", "salario_cliente": 3500000,
                "total_otros_prestamos": 1000000, "cuota_pactada": 128650,
                "puntaje_datacredito": 795, "cant_creditosvigentes": 0, "huella_consulta": 2,
                "saldo_mora": 0, "saldo_total": 0, "saldo_principal": None,
                "saldo_mora_codeudor": None, "creditos_sectorFinanciero": 0,
                "creditos_sectorCooperativo": 0, "creditos_sectorReal": 0,
                "promedio_ingresos_datacredito": 916148, "tendencia_ingresos": "Creciente",
            }
        }
    }


class SolicitudPrediccion(BaseModel):
    registros: List[Credito] = Field(..., min_length=1, max_length=MAX_REGISTROS)
    umbral: float = Field(UMBRAL_DEFECTO, ge=0, le=1, description="Probabilidad mínima para predecir 'paga a tiempo'")


class Prediccion(BaseModel):
    indice: int
    prob_pago_atiempo: float
    prediccion: int = Field(..., description="1 = pagará a tiempo, 0 = riesgo de no pagar")
    riesgo: str


class RespuestaPrediccion(BaseModel):
    total: int
    resultados: List[Prediccion]


# ---------------------------------------------------------------------------
# Lógica de predicción
# ---------------------------------------------------------------------------
def cargar_modelo(ruta: Path = RUTA_MODELO):
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecuta model_training_evaluation.py primero."
        )
    return joblib.load(ruta)


def preparar_datos(registros: pd.DataFrame) -> pd.DataFrame:
    """Aplica las mismas transformaciones que ft_engineering.py (antes del pipeline del modelo)."""
    faltantes = [c for c in COLS_ENTRADA if c not in registros.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")
    df = registros[COLS_ENTRADA].copy()
    df[COLS_NUMERICAS] = df[COLS_NUMERICAS].apply(pd.to_numeric, errors="coerce")
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"], errors="coerce")
    if df["fecha_prestamo"].isna().any():
        raise ValueError("fecha_prestamo tiene valores inválidos")
    return crear_variables_derivadas(limpiar_datos(df))


def nivel_riesgo(prob_pago: float) -> str:
    if prob_pago < 0.40:
        return "alto"
    return "medio" if prob_pago < 0.60 else "bajo"


def predecir(registros: pd.DataFrame, umbral: float = UMBRAL_DEFECTO) -> RespuestaPrediccion:
    if modelo is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado")
    try:
        X = preparar_datos(registros)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    probas = modelo.predict_proba(X)[:, 1]
    resultados = [
        Prediccion(
            indice=i, prob_pago_atiempo=round(float(p), 4),
            prediccion=int(p >= umbral), riesgo=nivel_riesgo(float(p)),
        )
        for i, p in enumerate(probas)
    ]
    return RespuestaPrediccion(total=len(resultados), resultados=resultados)


# ---------------------------------------------------------------------------
# Aplicación
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global modelo
    modelo = cargar_modelo()
    yield


app = FastAPI(
    title="API de scoring de crédito",
    description="Predice la probabilidad de que un cliente pague su crédito a tiempo.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": modelo is not None}


@app.post("/predict", response_model=RespuestaPrediccion)
def predict(solicitud: SolicitudPrediccion):
    """Predicción por lotes desde JSON: envía uno o varios registros en `registros`."""
    registros = pd.DataFrame([r.model_dump() for r in solicitud.registros])
    return predecir(registros, solicitud.umbral)


@app.post("/predict/csv", response_model=RespuestaPrediccion)
async def predict_csv(archivo: UploadFile = File(...), umbral: float = UMBRAL_DEFECTO):
    """Predicción por lotes desde un CSV con las columnas de entrada (sin `Pago_atiempo`)."""
    contenido = await archivo.read()
    try:
        registros = pd.read_csv(io.BytesIO(contenido))
    except Exception as error:
        raise HTTPException(status_code=400, detail="El archivo no es un CSV válido") from error
    if not 0 <= umbral <= 1:
        raise HTTPException(status_code=422, detail="umbral debe estar entre 0 y 1")
    if not 0 < len(registros) <= MAX_REGISTROS:
        raise HTTPException(status_code=422, detail=f"El CSV debe tener entre 1 y {MAX_REGISTROS} filas")
    return predecir(registros, umbral)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("model_deploy:app", host="0.0.0.0", port=8000)