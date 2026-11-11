import numpy as np
import pandas as pd
import pytest

from mlops_pipeline import ft_engineering as ft


def _df_base(**cambios):
    fila = {
        "edad_cliente": 35, "salario_cliente": 3_000_000, "puntaje_datacredito": 800,
        "tendencia_ingresos": "Estable", "tipo_credito": 4,
        "fecha_prestamo": "2025-03-15", "cuota_pactada": 300_000,
    }
    fila.update(cambios)
    return pd.DataFrame([fila])


def test_limpiar_convierte_valores_imposibles_en_nulo():
    df = ft.limpiar_datos(_df_base(edad_cliente=150, salario_cliente=60_000_000,
                                   puntaje_datacredito=-5, tendencia_ingresos="123"))
    fila = df.iloc[0]
    assert pd.isna(fila["edad_cliente"])
    assert pd.isna(fila["salario_cliente"])
    assert pd.isna(fila["puntaje_datacredito"])
    assert pd.isna(fila["tendencia_ingresos"])


def test_limpiar_conserva_valores_validos_y_tipos():
    df = ft.limpiar_datos(_df_base())
    assert df.loc[0, "edad_cliente"] == 35
    assert df.loc[0, "tendencia_ingresos"] == "Estable"
    assert df.loc[0, "tipo_credito"] == "4"
    assert pd.api.types.is_datetime64_any_dtype(df["fecha_prestamo"])


def test_variables_derivadas():
    df = ft.crear_variables_derivadas(ft.limpiar_datos(_df_base()))
    assert df.loc[0, "relacion_cuota_salario"] == pytest.approx(0.1)
    assert df.loc[0, "mes_prestamo"] == 3
    assert "fecha_prestamo" not in df.columns


def test_relacion_cuota_salario_con_salario_cero_es_nula():
    df = ft.crear_variables_derivadas(ft.limpiar_datos(_df_base(salario_cliente=0)))
    assert np.isnan(df.loc[0, "relacion_cuota_salario"])


def test_get_data_particion_y_exclusiones(datos):
    X_train, X_test, y_train, y_test, _ = datos
    total = len(X_train) + len(X_test)
    assert len(X_test) / total == pytest.approx(ft.TEST_SIZE, abs=0.01)
    assert list(X_train.columns) == list(X_test.columns)
    for col in ft.COLS_EXCLUIDAS + [ft.TARGET]:
        assert col not in X_train.columns
    assert y_train.mean() == pytest.approx(y_test.mean(), abs=0.01)  # estratificado


def test_preprocesador_no_deja_nulos(datos):
    X_train, _, _, _, preprocessor = datos
    transformado = preprocessor.fit_transform(X_train)
    assert not np.isnan(transformado).any()
    assert transformado.shape[0] == len(X_train)


def test_cargar_datos_ruta_inexistente():
    with pytest.raises(FileNotFoundError):
        ft.cargar_datos("no_existe.csv")