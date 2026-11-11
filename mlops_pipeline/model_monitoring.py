"""
model_monitoring.py
-------------------
Monitoreo del modelo de scoring de crédito y detección de data drift.

Compara los datos "históricos" (referencia, con los que se entrenó el modelo) contra los
datos "actuales" (loans posteriores a una fecha de corte), con una periodicidad definida,
y calcula por variable:
    - KS (Kolmogorov-Smirnov)            -> variables numéricas
    - PSI (Population Stability Index)   -> numéricas y categóricas
    - Jensen-Shannon                     -> numéricas y categóricas
    - Chi-cuadrado                       -> variables categóricas
También monitorea el drift de los pronósticos (probabilidad de pago a tiempo).

El semáforo se define con el PSI (estándar de la industria):
    verde < 0.10  |  amarillo 0.10 - 0.25  |  rojo >= 0.25
KS, JS y chi-cuadrado se reportan como evidencia complementaria.

Uso:
    Consola   : python model_monitoring.py [--corte 2025-06-01] [--freq M] [--guardar]
    Streamlit : streamlit run model_monitoring.py
"""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

from mlops_pipeline.ft_engineering (
    COLS_EXCLUIDAS,
    COLS_NOMINALES,
    COLS_ORDINALES,
    TARGET,
    cargar_datos,
    crear_variables_derivadas,
    limpiar_datos,
)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
RUTA_MODELO = Path(__file__).resolve().parent / "mejor_modelo.joblib"
FECHA_CORTE = "2025-06-01"      # antes = referencia histórica | desde aquí = monitoreo
PERIODICIDAD = "M"              # "M" mensual, "Q" trimestral
MIN_MUESTRA = 100               # ventanas con menos registros no se usan para alertar
N_BINES = 10
EPS = 1e-4
UMBRAL_PSI_ALERTA, UMBRAL_PSI_CRITICO = 0.10, 0.25
UMBRAL_P_VALOR = 0.05
UMBRAL_NULOS = 0.10             # cambio en el % de nulos que dispara revisión
PROPORCION_ROJAS_RETRAINING = 0.20
COL_FECHA = "fecha_prestamo"
COL_PROBA = "proba_pago"
# mes_prestamo es calendario: cambia por construcción al pasar el tiempo, no es drift real.
VARS_SIN_MONITOREO = ["mes_prestamo"]
SEMAFORO = {0: "🟢 Estable", 1: "🟡 Alerta", 2: "🔴 Crítico"}


# ---------------------------------------------------------------------------
# Datos + pronósticos
# ---------------------------------------------------------------------------
def cargar_tabla_monitoreo(ruta=None, ruta_modelo=RUTA_MODELO) -> pd.DataFrame:
    """Tabla con los datos (ya limpios), la fecha y el pronóstico entregado por el modelo."""
    df = limpiar_datos(cargar_datos(ruta))
    fecha = df[COL_FECHA].copy()
    df = crear_variables_derivadas(df)

    y = df[TARGET]
    X = df.drop(columns=[TARGET] + COLS_EXCLUIDAS)

    tabla = X.copy()
    tabla[COL_PROBA] = joblib.load(ruta_modelo).predict_proba(X)[:, 1]
    tabla[TARGET] = y.values
    tabla[COL_FECHA] = fecha.values
    return tabla.sort_values(COL_FECHA).reset_index(drop=True)


def definir_variables(tabla: pd.DataFrame):
    """Separa variables categóricas y numéricas a monitorear (incluye el pronóstico)."""
    excluir = {TARGET, COL_FECHA, *VARS_SIN_MONITOREO}
    cat = [c for c in COLS_NOMINALES + COLS_ORDINALES if c in tabla.columns]
    num = [c for c in tabla.columns if c not in excluir and c not in cat]
    return num, cat


def dividir_referencia_actual(tabla: pd.DataFrame, fecha_corte=FECHA_CORTE):
    corte = pd.Timestamp(fecha_corte)
    return tabla[tabla[COL_FECHA] < corte], tabla[tabla[COL_FECHA] >= corte]


def asignar_periodo(tabla: pd.DataFrame, freq=PERIODICIDAD) -> pd.Series:
    return tabla[COL_FECHA].dt.to_period(freq).astype(str)


# ---------------------------------------------------------------------------
# Métricas de drift
# ---------------------------------------------------------------------------
def _proporciones_numericas(ref: pd.Series, act: pd.Series, bines=N_BINES):
    """Proporciones por intervalo (cuantiles de la referencia) o por valor si hay pocos."""
    if ref.nunique() <= bines:
        cats = sorted(set(ref.unique()) | set(act.unique()))
        p = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        q = act.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        return p, q
    bordes = np.unique(np.quantile(ref, np.linspace(0, 1, bines + 1)))
    bordes[0], bordes[-1] = -np.inf, np.inf
    return (np.histogram(ref, bordes)[0] / len(ref),
            np.histogram(act, bordes)[0] / len(act))


def _conteos_categoricos(ref: pd.Series, act: pd.Series):
    """Conteos por categoría. Nulos = 'Sin dato'; categorías <1 % en la referencia = 'otros'."""
    ref, act = ref.fillna("Sin dato").astype(str), act.fillna("Sin dato").astype(str)
    frecuentes = ref.value_counts(normalize=True)
    frecuentes = set(frecuentes[frecuentes >= 0.01].index)
    ref = ref.where(ref.isin(frecuentes), "otros")
    act = act.where(act.isin(frecuentes), "otros")
    cats = sorted(set(ref.unique()) | set(act.unique()))
    return (ref.value_counts().reindex(cats, fill_value=0).to_numpy(),
            act.value_counts().reindex(cats, fill_value=0).to_numpy())


def psi(p, q) -> float:
    """Population Stability Index entre dos vectores de proporciones."""
    p, q = np.clip(p, EPS, None), np.clip(q, EPS, None)
    return float(np.sum((q - p) * np.log(q / p)))


def js_distance(p, q) -> float:
    """Distancia de Jensen-Shannon (base 2, rango 0-1)."""
    return float(jensenshannon(p, q, base=2))


def nivel_psi(valor: float):
    if pd.isna(valor):
        return np.nan
    if valor >= UMBRAL_PSI_CRITICO:
        return 2
    return 1 if valor >= UMBRAL_PSI_ALERTA else 0


def drift_numerico(ref: pd.Series, act: pd.Series) -> dict:
    r, a = ref.dropna(), act.dropna()
    if len(r) < 2 or len(a) < 2:
        return {}
    p, q = _proporciones_numericas(r, a)
    ks = stats.ks_2samp(r, a)
    return {"ks": ks.statistic, "ks_p": ks.pvalue, "psi": psi(p, q), "js": js_distance(p, q)}


def drift_categorico(ref: pd.Series, act: pd.Series) -> dict:
    c_ref, c_act = _conteos_categoricos(ref, act)
    if c_ref.sum() == 0 or c_act.sum() == 0:
        return {}
    p, q = c_ref / c_ref.sum(), c_act / c_act.sum()
    tabla = np.array([c_ref, c_act])
    tabla = tabla[:, tabla.sum(axis=0) > 0]
    chi2, p_val = (stats.chi2_contingency(tabla)[:2] if tabla.shape[1] > 1 else (np.nan, np.nan))
    return {"chi2": chi2, "chi2_p": p_val, "psi": psi(p, q), "js": js_distance(p, q)}


def calcular_drift(ref: pd.DataFrame, act: pd.DataFrame, num_cols, cat_cols) -> pd.DataFrame:
    """Tabla con las métricas de drift de cada variable (referencia vs. actual)."""
    filas = []
    for col in num_cols + cat_cols:
        es_num = col in num_cols
        metricas = drift_numerico(ref[col], act[col]) if es_num else drift_categorico(ref[col], act[col])
        p_val = metricas.get("ks_p" if es_num else "chi2_p", np.nan)
        filas.append({
            "variable": col,
            "tipo": "numérica" if es_num else "categórica",
            "ks": np.nan, "ks_p": np.nan, "chi2": np.nan, "chi2_p": np.nan,
            "psi": np.nan, "js": np.nan,
            **metricas,
            "significativo": bool(p_val < UMBRAL_P_VALOR) if pd.notna(p_val) else False,
            "nulos_ref": ref[col].isna().mean(),
            "nulos_act": act[col].isna().mean(),
        })
    tabla = pd.DataFrame(filas)
    tabla["nivel"] = tabla["psi"].apply(nivel_psi)
    tabla["estado"] = tabla["nivel"].map(SEMAFORO)
    return tabla


def drift_temporal(tabla: pd.DataFrame, ref: pd.DataFrame, act: pd.DataFrame,
                   num_cols, cat_cols, freq=PERIODICIDAD, min_muestra=MIN_MUESTRA) -> pd.DataFrame:
    """Drift de cada período (ventana) contra la referencia. Formato largo."""
    periodos = asignar_periodo(act, freq)
    partes = []
    for periodo, grupo in act.groupby(periodos):
        res = calcular_drift(ref, grupo, num_cols, cat_cols)
        res.insert(0, "periodo", periodo)
        res.insert(1, "n", len(grupo))
        res["muestra_suficiente"] = len(grupo) >= min_muestra
        partes.append(res)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def detectar_cambios(serie: pd.Series, salto=0.10) -> list:
    """Tendencia (pendiente) y cambios abruptos en una serie de PSI por período."""
    serie = serie.dropna()
    mensajes = []
    if len(serie) >= 4:
        pendiente = np.polyfit(range(len(serie)), serie.to_numpy(), 1)[0]
        if pendiente > 0.01:
            mensajes.append(f"Tendencia creciente del drift (+{pendiente:.3f} de PSI por período).")
    saltos = serie.diff()
    for periodo, valor in saltos[saltos > salto].items():
        mensajes.append(f"Cambio abrupto en {periodo}: el PSI medio subió {valor:.2f}.")
    return mensajes


# ---------------------------------------------------------------------------
# Alertas y recomendaciones
# ---------------------------------------------------------------------------
def generar_alertas(drift: pd.DataFrame) -> list:
    """Lista de (nivel, mensaje). nivel: 'error', 'warning' o 'success'."""
    alertas = []
    for _, f in drift[drift["nivel"] == 2].iterrows():
        alertas.append(("error", f"Drift crítico en '{f['variable']}': PSI={f['psi']:.3f}, "
                                 f"JS={f['js']:.3f}" + (f", KS={f['ks']:.3f}" if pd.notna(f["ks"]) else "")
                                 + (f", chi²-p={f['chi2_p']:.3g}" if pd.notna(f["chi2_p"]) else "") + "."))
    for _, f in drift[drift["nivel"] == 1].iterrows():
        alertas.append(("warning", f"Alerta en '{f['variable']}': PSI={f['psi']:.3f} "
                                   f"(entre {UMBRAL_PSI_ALERTA} y {UMBRAL_PSI_CRITICO})."))
    for _, f in drift.iterrows():
        cambio = f["nulos_act"] - f["nulos_ref"]
        if abs(cambio) >= UMBRAL_NULOS:
            alertas.append(("warning", f"El % de nulos de '{f['variable']}' cambió "
                                       f"{cambio * 100:+.1f} puntos (de {f['nulos_ref']:.1%} a {f['nulos_act']:.1%})."))
    if not alertas:
        alertas.append(("success", "Sin desviaciones significativas: la población actual es estable."))
    return alertas


def generar_recomendaciones(drift: pd.DataFrame) -> list:
    n = len(drift)
    rojas = drift[drift["nivel"] == 2]["variable"].tolist()
    amarillas = drift[drift["nivel"] == 1]["variable"].tolist()
    recs = []
    if COL_PROBA in rojas:
        recs.append("🔴 Los pronósticos cambiaron de forma crítica: el score del modelo ya no se "
                    "comporta como en el entrenamiento. Evalúa reentrenar con datos recientes.")
    if n and len(rojas) / n >= PROPORCION_ROJAS_RETRAINING:
        recs.append(f"🔴 {len(rojas)} de {n} variables están en rojo "
                    f"(≥ {PROPORCION_ROJAS_RETRAINING:.0%}): se recomienda **reentrenar** el modelo.")
    elif rojas:
        recs.append("🟠 Revisa estas variables antes de decidir el reentrenamiento: " + ", ".join(rojas) + ".")
    if amarillas:
        recs.append("🟡 Mantén bajo observación: " + ", ".join(amarillas) + ".")
    cambios_nulos = drift[(drift["nulos_act"] - drift["nulos_ref"]).abs() >= UMBRAL_NULOS]["variable"].tolist()
    if cambios_nulos:
        recs.append("🔎 Revisa el proceso de captura de datos (cambió el % de nulos): " + ", ".join(cambios_nulos) + ".")
    if not recs:
        recs.append("✅ No se requiere acción. Continúa con el monitoreo periódico.")
    return recs


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------
def plot_distribucion(ref: pd.Series, act: pd.Series, variable: str, es_numerica: bool):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if es_numerica:
        r, a = ref.dropna(), act.dropna()
        limite = np.nanpercentile(pd.concat([r, a]), [1, 99])
        bordes = np.linspace(limite[0], limite[1], 30) if limite[0] < limite[1] else 30
        ax.hist(r, bins=bordes, alpha=0.55, density=True, label="Histórico")
        ax.hist(a, bins=bordes, alpha=0.55, density=True, label="Actual")
        ax.set_ylabel("Densidad")
    else:
        c_ref, c_act = _conteos_categoricos(ref, act)
        etiquetas = _categorias_usadas(ref, act)
        x = np.arange(len(etiquetas))
        ax.bar(x - 0.2, c_ref / c_ref.sum(), 0.4, label="Histórico")
        ax.bar(x + 0.2, c_act / c_act.sum(), 0.4, label="Actual")
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, rotation=20)
        ax.set_ylabel("Proporción")
    ax.set_title(f"{variable}: histórico vs. actual")
    ax.legend()
    fig.tight_layout()
    return fig


def _categorias_usadas(ref: pd.Series, act: pd.Series) -> list:
    """Etiquetas en el mismo orden que _conteos_categoricos."""
    r, a = ref.fillna("Sin dato").astype(str), act.fillna("Sin dato").astype(str)
    frecuentes = r.value_counts(normalize=True)
    frecuentes = set(frecuentes[frecuentes >= 0.01].index)
    r, a = r.where(r.isin(frecuentes), "otros"), a.where(a.isin(frecuentes), "otros")
    return sorted(set(r.unique()) | set(a.unique()))


def plot_evolucion(temporal: pd.DataFrame, variables: list):
    suficiente = temporal[temporal["muestra_suficiente"]]
    pivote = suficiente.pivot(index="periodo", columns="variable", values="psi")
    fig, ax = plt.subplots(figsize=(8, 4))
    for v in variables:
        if v in pivote:
            ax.plot(pivote.index, pivote[v], marker="o", alpha=0.8, label=v)
    ax.plot(pivote.index, pivote.mean(axis=1), "k--", linewidth=2, label="PSI medio")
    ax.axhline(UMBRAL_PSI_ALERTA, color="orange", linestyle=":")
    ax.axhline(UMBRAL_PSI_CRITICO, color="red", linestyle=":")
    ax.set_ylabel("PSI")
    ax.set_title("Evolución del drift por período")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Consola
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(description="Monitoreo de data drift del modelo")
    parser.add_argument("--corte", default=FECHA_CORTE, help="Fecha de corte AAAA-MM-DD")
    parser.add_argument("--freq", default=PERIODICIDAD, choices=["M", "Q"], help="Periodicidad")
    parser.add_argument("--guardar", action="store_true", help="Guarda los reportes en CSV")
    args = parser.parse_args(argv)

    tabla = cargar_tabla_monitoreo()
    num_cols, cat_cols = definir_variables(tabla)
    ref, act = dividir_referencia_actual(tabla, args.corte)
    print(f"Referencia: {len(ref)} registros | Actual: {len(act)} registros (corte {args.corte})")

    drift = calcular_drift(ref, act, num_cols, cat_cols).sort_values("psi", ascending=False)
    columnas = ["variable", "tipo", "psi", "ks", "js", "chi2_p", "estado"]
    print("\nDrift por variable (histórico vs. actual):")
    print(drift[columnas].round(4).to_string(index=False))

    print("\nAlertas:")
    for nivel, mensaje in generar_alertas(drift):
        print(f" [{nivel.upper()}] {mensaje}")
    print("\nRecomendaciones:")
    for rec in generar_recomendaciones(drift):
        print(f" - {rec}")

    temporal = drift_temporal(tabla, ref, act, num_cols, cat_cols, args.freq)
    if args.guardar:
        drift.to_csv("reporte_drift.csv", index=False)
        temporal.to_csv("reporte_drift_temporal.csv", index=False)
        print("\nReportes guardados: reporte_drift.csv, reporte_drift_temporal.csv")
    return drift, temporal


# ---------------------------------------------------------------------------
# Aplicación Streamlit
# ---------------------------------------------------------------------------
def run_app():
    import streamlit as st

    st.set_page_config(page_title="Monitoreo del modelo", page_icon="📊", layout="wide")
    st.title("📊 Monitoreo del modelo de scoring de crédito")
    st.caption("Detección de data drift: población histórica vs. actual")

    @st.cache_data
    def _cargar():
        return cargar_tabla_monitoreo()

    tabla = _cargar()
    num_cols, cat_cols = definir_variables(tabla)

    with st.sidebar:
        st.header("Parámetros")
        corte = st.date_input("Fecha de corte (histórico | actual)", pd.Timestamp(FECHA_CORTE),
                              min_value=tabla[COL_FECHA].min().date(), max_value=tabla[COL_FECHA].max().date())
        etiquetas = {"Mensual": "M", "Trimestral": "Q"}
        freq = etiquetas[st.selectbox("Periodicidad", list(etiquetas))]
        min_muestra = st.number_input("Muestra mínima por período", 10, 1000, MIN_MUESTRA, step=10)

    ref, act = dividir_referencia_actual(tabla, corte)
    if len(act) == 0 or len(ref) == 0:
        st.error("La fecha de corte deja vacío el período histórico o el actual.")
        st.stop()

    temporal = drift_temporal(tabla, ref, act, num_cols, cat_cols, freq, min_muestra)
    periodos = ["Todo el período actual"] + sorted(temporal["periodo"].unique())
    with st.sidebar:
        ventana = st.selectbox("Ventana a evaluar", periodos)
    act_sel = act if ventana == periodos[0] else act[asignar_periodo(act, freq) == ventana]
    if len(act_sel) < min_muestra:
        st.warning(f"La ventana tiene solo {len(act_sel)} registros: los resultados son poco confiables.")
    drift = calcular_drift(ref, act_sel, num_cols, cat_cols)

    tab_res, tab_dist, tab_met, tab_tmp, tab_rec = st.tabs(
        ["Resumen", "Distribuciones", "Métricas por variable", "Evolución temporal", "Alertas y recomendaciones"])

    with tab_res:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Registros históricos", f"{len(ref):,}")
        c2.metric("Registros actuales", f"{len(act_sel):,}")
        c3.metric("🔴 Variables críticas", int((drift["nivel"] == 2).sum()))
        c4.metric("🟡 Variables en alerta", int((drift["nivel"] == 1).sum()))
        pron = drift[drift["variable"] == COL_PROBA].iloc[0]
        st.subheader(f"Pronóstico del modelo: {pron['estado']}")
        st.progress(float(min(pron["psi"] / UMBRAL_PSI_CRITICO, 1.0)),
                    text=f"PSI del pronóstico = {pron['psi']:.3f} (crítico ≥ {UMBRAL_PSI_CRITICO})")
        st.subheader("Riesgo por variable")
        for _, f in drift.sort_values("psi", ascending=False).iterrows():
            st.progress(float(min(f["psi"] / UMBRAL_PSI_CRITICO, 1.0)) if pd.notna(f["psi"]) else 0.0,
                        text=f"{f['estado']}  {f['variable']}  (PSI {f['psi']:.3f})")

    with tab_dist:
        variable = st.selectbox("Variable", num_cols + cat_cols)
        st.pyplot(plot_distribucion(ref[variable], act_sel[variable], variable, variable in num_cols))
        plt.close("all")

    with tab_met:
        mostrar = drift[["variable", "tipo", "estado", "psi", "ks", "ks_p", "js", "chi2", "chi2_p",
                         "nulos_ref", "nulos_act"]].sort_values("psi", ascending=False)
        st.dataframe(mostrar.round(4), hide_index=True)
        st.caption("PSI < 0.10 estable · 0.10–0.25 alerta · ≥ 0.25 crítico. KS aplica a numéricas, chi² a categóricas.")

    with tab_tmp:
        suficiente = temporal[temporal["muestra_suficiente"]]
        if suficiente.empty:
            st.info("Ningún período alcanza la muestra mínima.")
        else:
            top = (suficiente.groupby("variable")["psi"].mean().sort_values(ascending=False).head(4).index.tolist())
            elegidas = st.multiselect("Variables a graficar", num_cols + cat_cols, default=top)
            st.pyplot(plot_evolucion(temporal, elegidas))
            plt.close("all")
            medio = suficiente.groupby("periodo")["psi"].mean()
            for mensaje in detectar_cambios(medio) or ["Sin tendencias ni cambios abruptos en el PSI medio."]:
                st.write("• " + mensaje)
            resumen = suficiente.pivot(index="periodo", columns="variable", values="psi")
            st.dataframe(resumen.round(3))
        excluidos = temporal[~temporal["muestra_suficiente"]][["periodo", "n"]].drop_duplicates()
        if not excluidos.empty:
            st.caption("Períodos excluidos por muestra insuficiente: "
                       + ", ".join(f"{r.periodo} (n={r.n})" for r in excluidos.itertuples()))

    with tab_rec:
        for nivel, mensaje in generar_alertas(drift):
            getattr(st, nivel)(mensaje)
        st.subheader("Recomendaciones")
        for rec in generar_recomendaciones(drift):
            st.markdown(rec)


def _en_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:  # streamlit no instalado o versión distinta
        return False


if __name__ == "__main__":
    run_app() if _en_streamlit() else main()