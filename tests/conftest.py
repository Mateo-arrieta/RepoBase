"""Fixtures compartidas: datos reales del repo y un modelo pequeño entrenado al vuelo."""
from pathlib import Path

import joblib
import matplotlib
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parents[1]
CSV = RAIZ / "Base_de_datos.csv"


@pytest.fixture(scope="session")
def ruta_csv():
    return CSV


@pytest.fixture(scope="session")
def datos():
    from ft_engineering import get_data

    return get_data(ruta=CSV)


@pytest.fixture(scope="session")
def modelo_entrenado(datos):
    X_train, _, y_train, _, preprocessor = datos
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    return pipe.fit(X_train, y_train)


@pytest.fixture(scope="session")
def ruta_modelo(modelo_entrenado, tmp_path_factory):
    ruta = tmp_path_factory.mktemp("modelo") / "mejor_modelo.joblib"
    joblib.dump(modelo_entrenado, ruta)
    return ruta