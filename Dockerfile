# syntax=docker/dockerfile:1.7

########################
# Stage 1: Build wheels
########################
FROM python:3.11-slim as builder

WORKDIR /app

# Optimización: No escribir .pyc, salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Instalar dependencias del sistema necesarias solo para build
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Copiar solo requirements para aprovechar el cache
COPY requirements.txt .

# Instalar PyTorch CPU y construir ruedas de dependencias
RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch==2.1.2+cpu torchvision==0.16.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip wheel --no-cache-dir --wheel-dir /app/wheels -r requirements.txt

########################
# Stage 2: Runtime
########################
FROM python:3.11-slim

WORKDIR /app

# Seguridad: Usuario no root
RUN useradd -m appuser

# Copiar las ruedas y dependencias
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
COPY . .

# Instalar dependencias desde las ruedas
RUN pip install --no-cache-dir /wheels/*

# Variables de entorno recomendadas para PyTorch/OpenMP (evita conflictos OMP)
ENV KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2

# Seguridad: Cambiar a usuario no root
USER appuser

# Exponer el puerto si es necesario
EXPOSE 8080

# Healthcheck opcional
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import torch; assert torch.__version__"

# Comando de inicio
CMD ["python", "demo.py"]
