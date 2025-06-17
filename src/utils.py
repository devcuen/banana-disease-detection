"""
Utilidades para el sistema de detección de enfermedades en banano
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from typing import List, Dict, Tuple
import torch

def load_image(image_path: str) -> Image.Image:
    """
    Cargar imagen desde archivo

    Args:
        image_path (str): Ruta de la imagen

    Returns:
        PIL.Image: Imagen cargada
    """
    try:
        image = Image.open(image_path).convert('RGB')
        return image
    except Exception as e:
        print(f"Error cargando imagen {image_path}: {e}")
        return None

def save_results_json(results: List[Dict], output_path: str):
    """
    Guardar resultados en formato JSON

    Args:
        results (List[Dict]): Lista de resultados
        output_path (str): Ruta de salida
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"✅ Resultados guardados en: {output_path}")
    except Exception as e:
        print(f"❌ Error guardando resultados: {e}")

def plot_predictions(results: List[Dict], save_path: str = None):
    """
    Crear gráfico de predicciones

    Args:
        results (List[Dict]): Resultados de predicción
        save_path (str): Ruta para guardar el gráfico
    """
    try:
        # Contar predicciones por clase
        class_counts = {}
        for result in results:
            if result['status'] == 'success':
                pred_class = result['predicted_class']
                class_counts[pred_class] = class_counts.get(pred_class, 0) + 1

        if not class_counts:
            print("No hay resultados válidos para graficar")
            return

        # Crear gráfico
        plt.figure(figsize=(10, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.bar(classes, counts, color=['green', 'red', 'orange', 'purple'])
        plt.title('Distribución de Predicciones por Clase')
        plt.xlabel('Clase Predicha')
        plt.ylabel('Número de Imágenes')
        plt.xticks(rotation=45)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Gráfico guardado en: {save_path}")
        else:
            plt.show()

    except Exception as e:
        print(f"❌ Error creando gráfico: {e}")

def calculate_metrics(y_true: List[int], y_pred: List[int], classes: List[str]) -> Dict:
    """
    Calcular métricas de evaluación

    Args:
        y_true (List[int]): Etiquetas verdaderas
        y_pred (List[int]): Predicciones
        classes (List[str]): Lista de clases

    Returns:
        Dict: Métricas calculadas
    """
    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'classes': classes
        }
    except ImportError:
        print("⚠️ scikit-learn no está instalado. Métricas no disponibles.")
        return {}
    except Exception as e:
        print(f"❌ Error calculando métricas: {e}")
        return {}

def create_model_summary(model, input_size: Tuple[int, int, int] = (3, 224, 224)) -> str:
    """
    Crear resumen del modelo

    Args:
        model: Modelo PyTorch
        input_size: Tamaño de entrada (C, H, W)

    Returns:
        str: Resumen del modelo
    """
    try:
        # Calcular parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Estimar tamaño del modelo
        param_size = total_params * 4 / (1024**2)  # MB (float32)

        summary = f"""
📊 RESUMEN DEL MODELO
{'='*50}
Arquitectura: {model.__class__.__name__}
Parámetros totales: {total_params:,}
Parámetros entrenables: {trainable_params:,}
Parámetros congelados: {total_params - trainable_params:,}
Tamaño estimado: {param_size:.2f} MB
Entrada: {input_size}
"""
        return summary
    except Exception as e:
        return f"❌ Error generando resumen: {e}"

def setup_logging(log_file: str = "banana_detector.log"):
    """
    Configurar sistema de logging

    Args:
        log_file (str): Archivo de log
    """
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info("🍌 Sistema de detección de enfermedades en banano iniciado")
    return logger
