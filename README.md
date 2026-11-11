# Scoring de crédito: predicción de pago a tiempo

Proyecto final de **Machine Learning** — Universidad Católica Luis Amigó
Docente: Juan Sebastián Parra Sánchez · Entrega: 11 de noviembre de 2026

Modelo supervisado que predice si un cliente **pagará a tiempo** su crédito, con un flujo MLOps
completo: exploración de datos, ingeniería de características, entrenamiento y selección del
modelo, monitoreo de data drift y despliegue como API.

---

## 1. Caso de negocio

Una entidad financiera otorga créditos de consumo y necesita anticipar cuáles clientes
incumplirán sus pagos. Un modelo que priorice a los clientes de mayor riesgo permite ajustar
el monto, el plazo o las condiciones del crédito, o pedir garantías adicionales, antes de desembolsar.

- **Variable objetivo:** `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).
- **Clase de interés:** la **0** (incumplimiento). Es la minoritaria (~4,7 %), por eso no basta
  con medir la exactitud: predecir siempre "pagó a tiempo" ya da 95 % de accuracy. Se evalúan
  también precisión, recall y F1 de la clase 0, y el ROC-AUC.
- **Decisión que apoya:** el costo de aprobar un crédito que no se paga es mayor que el de
  rechazar uno bueno, así que se privilegia el **recall de la clase 0**.

## 2. Datos

| Aspecto | Detalle |
|---|---|
| Archivo | `Base_de_datos.csv` (origen: `BD_creditos.xlsx`) |
| Tamaño | 10.763 créditos × 23 columnas, sin filas duplicadas |
| Periodo | 26-nov-2024 a 26-abr-2026 (el volumen de créditos baja con el tiempo) |
| Objetivo | 95,25 % pagó a tiempo · 4,75 % no pagó |

Variables: tipo de crédito, fecha del préstamo, capital prestado, plazo, edad, tipo laboral,
salario, otros préstamos, cuota pactada, puntaje y puntaje Datacrédito, créditos vigentes,
huella de consulta, saldos (mora, total, principal, mora del codeudor), créditos por sector
(financiero, cooperativo, real), promedio de ingresos y tendencia de ingresos.

## 3. Estructura del repositorio

La estructura es fija (los pipelines de Jenkins la validan) y no debe modificarse.

```
├── mlops_pipeline/
│   ├── Cargar_datos.ipynb            # carga del dataset
│   ├── comprension_eda.ipynb         # limpieza y análisis exploratorio
│   ├── ft_engineering.py             # features, pipelines y partición train/test
│   ├── model_training_evaluation.py  # entrenamiento, evaluación y selección del mejor modelo
│   ├── model_deploy.py               # API de predicción
│   └── model_monitoring.py           # data drift + app Streamlit
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
├── setup.bat
└── readme.md
```

Ramas: `developer` (desarrollo), `certification` (validación) y `master` (producción).

## 4. Instalación

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 5. Proceso

### 5.1 Comprensión y EDA (`comprension_eda.ipynb`)

Hallazgos que definieron las reglas de limpieza:

- **Valores imposibles:** 150 edades fuera de 18–100 años (máximo 123), 97 salarios mayores a
  50 millones y 146 puntajes de Datacrédito menores o iguales a 0. Se convierten a nulo para
  que todos los nulos se representen igual.
- **`tendencia_ingresos`** mezclaba números sueltos con las categorías válidas
  (Decreciente, Estable, Creciente); los números se tratan como nulo.
- **Nulos:** `promedio_ingresos_datacredito` (~27 %) y `tendencia_ingresos` (~27 %) son los más
  afectados; `saldo_mora_codeudor` (~5 %), `saldo_principal` (~4 %) y `saldo_mora` / `saldo_total` (~1 %).
- **Tipos:** `tipo_credito` es un código, se trata como categoría y no como número;
  `fecha_prestamo` pasa a fecha.
- **Fuga de datos (data leakage):** la variable `puntaje` separa el objetivo a la perfección
  (ROC-AUC = 1,0 por sí sola), lo que indica que se calcula con información posterior al pago.
  **Se excluye del modelo.** Incluirla daría métricas irreales y un modelo inútil en producción.
- **Atributos derivados:** `relacion_cuota_salario` (cuota / salario) y `mes_prestamo`.

### 5.2 Ingeniería de características (`ft_engineering.py`)

`ColumnTransformer` con tres pipelines:

| Pipeline | Variables | Pasos |
|---|---|---|
| numeric | numéricas | `SimpleImputer` (mediana) |
| categoric | `tipo_credito`, `tipo_laboral` | `SimpleImputer` (moda) + `OneHotEncoder` |
| categoric ordinales | `tendencia_ingresos` | `SimpleImputer` (moda) + `OrdinalEncoder` (Decreciente < Estable < Creciente) |

Partición 80 % entrenamiento / 20 % evaluación, estratificada por el objetivo. El preprocesador se
ajusta dentro del pipeline de cada modelo, solo con datos de entrenamiento, para evitar fuga de información.

### 5.3 Entrenamiento y selección (`model_training_evaluation.py`)

Se comparan cuatro modelos con `class_weight` balanceado, usando las funciones reutilizables
`summarize_classification` y `build_model`. El mejor se elige con un puntaje que combina
**performance** (ROC-AUC en test, 60 %), **consistency** (desviación estándar del AUC en
validación cruzada de 5 folds, 20 %) y **scalability** (tiempo de entrenamiento, 20 %).

| Modelo | ROC-AUC | Recall clase 0 | F1 clase 0 | CV AUC (std) | Entrenamiento (s) |
|---|---|---|---|---|---|
| **Regresión Logística** | **0,712** | **0,588** | 0,141 | 0,0076 | 0,07 |
| Random Forest | 0,696 | 0,039 | 0,074 | 0,0171 | 2,74 |
| HistGradientBoosting | 0,650 | 0,186 | 0,162 | 0,0179 | 0,85 |
| Árbol de Decisión | 0,574 | 0,529 | 0,109 | 0,0152 | 0,08 |

**Modelo seleccionado: Regresión Logística.** Es el de mejor AUC, el más estable en validación
cruzada y el más rápido, y detecta ~59 % de los clientes que no pagan.

Limitaciones: el poder predictivo es moderado (AUC ≈ 0,71) y la precisión de la clase 0 es baja
(0,08). Por cada cliente de riesgo bien detectado hay varios falsos positivos, así que conviene
usar el modelo para **priorizar y revisar** solicitudes, no para rechazarlas automáticamente.
El modelo se guarda en `mejor_modelo.joblib`.

### 5.4 Monitoreo de data drift (`model_monitoring.py`)

Compara la población histórica (créditos anteriores al 1-jun-2025, 7.713 registros) contra la
actual (3.050 registros) por periodos mensuales o trimestrales. Métricas por variable: KS,
PSI, Jensen-Shannon y chi-cuadrado para las categóricas, además del drift del pronóstico del
modelo. El semáforo usa el PSI: verde < 0,10 · amarillo 0,10–0,25 · rojo ≥ 0,25.

```bash
cd mlops_pipeline
python model_monitoring.py --guardar     # reporte por consola y CSV
streamlit run model_monitoring.py        # aplicación Streamlit
```

La app incluye resumen con semáforo, distribución histórica vs. actual, tabla de métricas,
evolución temporal con detección de tendencias o cambios abruptos, y alertas con
recomendaciones de retraining.

Resultados:

- **Crítico:** `promedio_ingresos_datacredito` (PSI 0,37).
- **Alerta:** `plazo_meses` (0,15) y `total_otros_prestamos` (0,13).
- **Pronósticos del modelo:** estables (PSI 0,004).
- El drift crece con el tiempo (PSI medio de 0,06 en junio de 2025 a 0,25 en febrero de 2026),
  aunque los últimos meses tienen muy pocos créditos y no son concluyentes.
- `mes_prestamo` se excluye del monitoreo porque cambia por construcción con el calendario.

**Recomendación:** revisar el origen de `promedio_ingresos_datacredito` y, si el drift continúa,
reentrenar el modelo con datos recientes.

### 5.5 Despliegue (`model_deploy.py`)

API REST con **FastAPI** que carga `mejor_modelo.joblib`, aplica la misma limpieza y los mismos
atributos derivados de `ft_engineering.py` y devuelve la probabilidad de pago a tiempo por
solicitud. Soporta predicción por lotes.

| Endpoint | Descripción |
|---|---|
| `GET /health` | Estado del servicio y del modelo |
| `POST /predict` | JSON con `registros` (uno o varios) y `umbral` opcional (por defecto 0,5) |
| `POST /predict/csv` | Archivo CSV con las columnas de entrada (sin `Pago_atiempo`) |

Cada resultado trae `prob_pago_atiempo`, `prediccion` (1 = pagará a tiempo, 0 = riesgo) y
`riesgo` (alto < 0,40 · medio 0,40–0,60 · bajo ≥ 0,60). Los campos opcionales pueden ir nulos:
el pipeline del modelo los imputa. Documentación interactiva en `/docs`.

```bash
# Local
cd mlops_pipeline
uvicorn model_deploy:app --port 8000

# Docker (desde la raíz del repo)
docker build -t scoring-credito .
docker run -p 8000:8000 scoring-credito
```

Ejemplo:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d "{\"registros\": [{\"tipo_credito\": 4, \"fecha_prestamo\": \"2026-01-07T14:40:36\", \"capital_prestado\": 1852560, \"plazo_meses\": 12, \"cuota_pactada\": 128650}]}"
```

La imagen (`Dockerfile`, `.dockerignore`) instala `requirements.txt`, copia el código y los datos,
entrena el modelo al construirse y levanta el servidor con Uvicorn.

## 6. Pruebas Unitarias, Cobertura y Calidad de Código (CI/CD)

El pipeline de MLOps cuenta con integración continua mediante **GitHub Actions** y análisis de calidad continuo con **SonarCloud**.

### Resumen de Ejecución
* **Estado de la suite de pruebas:** `PASS` (33/33 pruebas ejecutadas y aprobadas en runner Linux)
* **Cobertura de Código (Code Coverage):** `62.3%` (Reportado por SonarCloud sobre 559 líneas evaluables)
* **Líneas de Código Analizadas:** 723
* **Duplicaciones:** `0.0%`

### Badges de SonarCloud
![SonarCloud Coverage](https://sonarcloud.io/api/project_badges/measure?project=Mateo-arrieta_RepoBase&metric=coverage)

Cobertura local: **98 %** (45 pruebas). El análisis corre en GitHub Actions
(`.github/workflows/sonarcloud.yml`) en cada push a `developer`, `certification` y `master`,
con la configuración de `sonar-project.properties`.

_Resultados de SonarCloud (quality gate, bugs, vulnerabilidades, duplicación): pendiente de completar._
