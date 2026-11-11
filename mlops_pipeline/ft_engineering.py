"""
ft_engineering.py
-----------------
Ingeniería de características del proyecto de scoring de crédito.

Qué hace:
    1. Carga el dataset (CSV generado por Cargar_datos.ipynb).
    2. Aplica las mismas reglas de limpieza del EDA (nulos unificados, tipos, valores atípicos).
    3. Crea atributos derivados.
    4. Define el ColumnTransformer con tres pipelines (numérico, categórico, ordinal).
    5. Divide en entrenamiento y evaluación y retorna los conjuntos.

Uso desde otro script:
    from ft_engineering import get_data
    X_train, X_test, y_train, y_test, preprocessor = get_data()
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
TARGET = "Pago_atiempo"          # 1 = pagó a tiempo, 0 = no pagó a tiempo
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Reglas de limpieza (deben coincidir con las del notebook de EDA)
EDAD_MIN, EDAD_MAX = 18, 100
SALARIO_MAX = 50_000_000
ORDEN_TENDENCIA = ["Decreciente", "Estable", "Creciente"]

# Columnas por tipo
COLS_NOMINALES = ["tipo_credito", "tipo_laboral"]
COLS_ORDINALES = ["tendencia_ingresos"]
COLS_FECHA = ["fecha_prestamo"]

# Variables excluidas por fuga de datos (data leakage): `puntaje` separa el objetivo
# casi perfectamente (ROC-AUC = 1.0 por sí sola), lo que indica que se calcula con
# información posterior al resultado del pago. Para incluirla, deja la lista vacía.
COLS_EXCLUIDAS = ["puntaje"]

# Sitios donde puede estar el CSV (se prueba en este orden)
RUTAS_CSV = [
    Path("../data/BD_creditos.csv"),
    Path("Base_de_datos.csv"),
    Path("../Base_de_datos.csv"),
    Path("../data/BD_creditos.csv"),
]


# ---------------------------------------------------------------------------
# Carga y limpieza
# ---------------------------------------------------------------------------
def cargar_datos(ruta=None) -> pd.DataFrame:
    """Lee el CSV. Si no se indica ruta, busca en las rutas habituales del repo."""
    raiz_proyecto = Path(__file__).resolve().parent.parent

    if ruta:
        path_ruta = Path(ruta)
        candidatas = [
            path_ruta,
            raiz_proyecto / path_ruta,
            raiz_proyecto / path_ruta.name,
            raiz_proyecto / "data" / path_ruta.name,
        ]
    else:
        candidatas = [
            raiz_proyecto / "data" / "BD_creditos.csv",
            raiz_proyecto / "Base_de_datos.csv",
            raiz_proyecto / "data" / "Base_de_datos.csv",
            Path("data/BD_creditos.csv"),
            Path("Base_de_datos.csv"),
        ]

    for candidata in candidatas:
        if candidata.exists():
            return pd.read_csv(candidata, parse_dates=COLS_FECHA)

    raise FileNotFoundError(
        f"No se encontró el CSV en ninguna de las rutas intentadas: {candidatas}"
    )


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica nulos, corrige valores imposibles y asigna el tipo correcto a cada columna."""
    df = df.copy()

    # Valores imposibles -> NaN (así todos los nulos se representan igual)
    df.loc[~df["edad_cliente"].between(EDAD_MIN, EDAD_MAX), "edad_cliente"] = np.nan
    df.loc[df["salario_cliente"] > SALARIO_MAX, "salario_cliente"] = np.nan
    df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan

    # tendencia_ingresos trae números sueltos mezclados con las categorías válidas
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(ORDEN_TENDENCIA)
    )

    # Tipos: los códigos de tipo_credito son categorías, no cantidades
    df["tipo_credito"] = df["tipo_credito"].astype(str)
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    return df


def crear_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Atributos calculados que pueden aportar poder predictivo."""
    df = df.copy()
    salario = df["salario_cliente"].replace(0, np.nan)
    df["relacion_cuota_salario"] = df["cuota_pactada"] / salario
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    return df.drop(columns=COLS_FECHA)


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """ColumnTransformer con los tres pipelines: numérico, categórico y ordinal."""
    cols_categoricas = [c for c in COLS_NOMINALES if c in X.columns]
    cols_ordinales = [c for c in COLS_ORDINALES if c in X.columns]
    cols_numericas = [c for c in X.columns if c not in cols_categoricas + cols_ordinales]

    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categoric = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    categoric_ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(
            categories=[ORDEN_TENDENCIA] * len(cols_ordinales),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric, cols_numericas),
        ("categoric", categoric, cols_categoricas),
        ("categoric ordinales", categoric_ordinal, cols_ordinales),
    ])


def get_data(ruta=None, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Retorna X_train, X_test, y_train, y_test y el preprocesador (sin ajustar).

    El preprocesador NO se ajusta aquí: se ajusta dentro del pipeline de cada modelo
    solo con los datos de entrenamiento, para evitar fuga de información (data leakage).
    """
    df = cargar_datos(ruta)
    df = limpiar_datos(df)
    df = crear_variables_derivadas(df)

    X = df.drop(columns=[TARGET] + COLS_EXCLUIDAS)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, build_preprocessor(X)


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, prep = get_data()
    print(f"X_train: {X_tr.shape} | X_test: {X_te.shape}")
    print("Distribución del objetivo (train):")
    print(y_tr.value_counts(normalize=True).round(4))
    print(prep)
