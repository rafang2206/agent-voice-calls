# Imagen de ejecución mínima para FastAPI/Uvicorn.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Separar dependencias del código mantiene la capa de pip en caché mientras
# requirements.txt no cambie.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# La aplicación no necesita privilegios de root en tiempo de ejecución.
RUN addgroup --system app && adduser --system --ingroup app app
COPY --chown=app:app . ./
USER app

EXPOSE 8000

# El WebSocket de Telnyx requiere que el proceso escuche fuera de localhost.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
