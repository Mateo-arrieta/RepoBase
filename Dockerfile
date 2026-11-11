FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Dependencias primero para aprovechar la caché de capas de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente y datos
COPY mlops_pipeline/ mlops_pipeline/
COPY Base_de_datos.csv .

# Entrena el modelo durante la construcción de la imagen (genera mejor_modelo.joblib)
WORKDIR /app/mlops_pipeline
RUN python model_training_evaluation.py

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]