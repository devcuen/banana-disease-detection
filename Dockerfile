# Dockerfile para Banana Disease Detection System
FROM python:3.9-slim

# Información del mantenedor
LABEL maintainer="jordanviion@gmail.com"
LABEL description="Sistema de detección de enfermedades en banano con Deep Learning"
LABEL version="1.0.0"

# Establecer directorio de trabajo
WORKDIR /app

# Variables de entorno
ENV PYTHONPATH="${PYTHONPATH}:/app"
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de requirements primero (para cache de Docker)
COPY requirements.txt .

# Instalar dependencias de Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY . .

# Instalar el paquete en modo desarrollo
RUN pip install -e .

# Crear directorios necesarios
RUN mkdir -p data/samples models/pretrained logs

# Crear usuario no-root para seguridad
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Exponer el puerto
EXPOSE 8000

# Comando por defecto
CMD ["python", "-m", "src.app"]

# Healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1