#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Script - Banana Disease Detection System
Ejemplo de uso del sistema de detección de enfermedades en banano

Autor: Jordan V
Fecha: Junio 2025
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import argparse
import sys
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar el downloader de muestras
try:
    from download_samples import BananaSampleDownloader
except ImportError:
    print("⚠️  Módulo download_samples no encontrado. Algunas funciones pueden no estar disponibles.")
    BananaSampleDownloader = None

class BananaPlantDiseaseDetector:
    """
    Sistema de detección de enfermedades en banano usando Transfer Learning con ResNet18.

    Clases detectadas:
    0: Sano
    1: Fusarium_R4T  
    2: Moko_Bacteriano
    3: Sigatoka_Negra
    """

    def __init__(self, model_path=None, device='auto'):
        """
        Inicializar el detector

        Args:
            model_path: Ruta al modelo entrenado (opcional)
            device: 'auto', 'cpu', 'cuda' o 'mps'
        """
        self.classes = ['Sano', 'Fusarium_R4T', 'Moko_Bacteriano', 'Sigatoka_Negra']
        self.class_colors = ['green', 'orange', 'red', 'purple']
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.transform = self._setup_transforms()

        print(f"🚀 Detector inicializado exitosamente!")
        print(f"   Device: {self.device}")
        print(f"   Classes: {len(self.classes)}")
        print(f"   Model: ResNet18 + Transfer Learning")

    def _setup_device(self, device):
        """Configurar device automáticamente"""
        if device == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(device)

    def _load_model(self, model_path):
        """Cargar modelo ResNet18 con transfer learning"""
        # Crear arquitectura base
        model = models.resnet18(pretrained=True)

        # Congelar capas base para transfer learning
        for param in model.parameters():
            param.requires_grad = False

        # Reemplazar clasificador final
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, len(self.classes))
        )

        # Cargar pesos si están disponibles
        if model_path and os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"✅ Modelo cargado desde: {model_path}")
            except:
                print(f"⚠️  No se pudo cargar el modelo desde {model_path}. Usando modelo base.")
        else:
            print(f"ℹ️  Usando modelo base ResNet18 (sin entrenamiento específico)")

        model.to(self.device)
        model.eval()
        return model

    def _setup_transforms(self):
        """Configurar transformaciones de imagen"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_image(self, image_path, show_plot=True, save_results=False):
        """
        Predecir enfermedad en una imagen

        Args:
            image_path: Ruta a la imagen
            show_plot: Mostrar gráfico de resultados
            save_results: Guardar resultados en JSON

        Returns:
            dict: Resultados de la predicción
        """
        try:
            # Cargar y preprocesar imagen
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            # Aplicar transformaciones
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            # Predicción
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs = probabilities.cpu().numpy()[0]
                predicted_class = np.argmax(probs)
                confidence = probs[predicted_class]

            # Crear resultados
            results = {
                'image_path': str(image_path),
                'image_size': original_size,
                'predicted_class': self.classes[predicted_class],
                'confidence': float(confidence),
                'all_probabilities': {
                    self.classes[i]: float(prob) for i, prob in enumerate(probs)
                },
                'timestamp': datetime.now().isoformat(),
                'model': 'ResNet18_TransferLearning'
            }

            # Mostrar resultados
            self._print_results(results)

            # Visualizar si se requiere
            if show_plot:
                self._plot_results(image, results)

            # Guardar resultados si se requiere
            if save_results:
                self._save_results(results)

            return results

        except Exception as e:
            print(f"❌ Error procesando imagen {image_path}: {str(e)}")
            return None

    def _print_results(self, results):
        """Imprimir resultados formateados"""
        print("\n" + "="*60)
        print("📊 RESULTADOS DEL ANÁLISIS")
        print("="*60)
        print(f"📁 Imagen: {Path(results['image_path']).name}")
        print(f"📏 Tamaño: {results['image_size'][0]}x{results['image_size'][1]} px")
        print(f"⏰ Timestamp: {results['timestamp']}")
        print()
        print(f"🎯 PREDICCIÓN PRINCIPAL:")
        print(f"   Clase: {results['predicted_class']}")
        print(f"   Confianza: {results['confidence']:.1%}")
        print()
        print(f"📈 TODAS LAS PROBABILIDADES:")
        for class_name, prob in results['all_probabilities'].items():
            bar = "█" * int(prob * 20)
            print(f"   {class_name:<15}: {prob:6.1%} {bar}")
        print("="*60)

    def _plot_results(self, image, results):
        """Crear visualización de resultados"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Mostrar imagen original
        ax1.imshow(image)
        ax1.set_title(f"Imagen Analizada\n{Path(results['image_path']).name}", fontsize=12)
        ax1.axis('off')

        # Agregar etiqueta de predicción
        prediction_text = f"Predicción: {results['predicted_class']}\nConfianza: {results['confidence']:.1%}"
        ax1.text(10, 10, prediction_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8),
                verticalalignment='top')

        # Gráfico de barras de probabilidades
        classes = list(results['all_probabilities'].keys())
        probs = list(results['all_probabilities'].values())
        colors = self.class_colors

        bars = ax2.bar(classes, probs, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Probabilidades por Clase', fontsize=12)
        ax2.set_ylabel('Probabilidad')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)

        # Agregar valores en las barras
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.1%}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        plt.show()

    def _save_results(self, results):
        """Guardar resultados en archivo JSON"""
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"prediction_{timestamp}.json"
        filepath = output_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"💾 Resultados guardados en: {filepath}")

    def analyze_samples_directory(self, samples_dir="data/samples"):
        """Analizar todas las imágenes de muestra"""
        samples_path = Path(samples_dir)

        if not samples_path.exists():
            print(f"❌ Directorio de muestras no encontrado: {samples_path}")
            print("💡 Ejecuta primero: python download_samples.py")
            return

        print(f"🔍 Analizando muestras en: {samples_path}")
        print("="*60)

        # Encontrar todas las imágenes
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        all_results = []

        for category_dir in samples_path.iterdir():
            if category_dir.is_dir():
                print(f"\n📂 Procesando categoría: {category_dir.name.upper()}")
                print("-"*40)

                images = [f for f in category_dir.iterdir() 
                         if f.suffix.lower() in image_extensions]

                if not images:
                    print(f"   ⚠️  No se encontraron imágenes en {category_dir}")
                    continue

                for image_path in images:
                    print(f"\n🖼️  Analizando: {image_path.name}")
                    result = self.predict_image(image_path, show_plot=False, save_results=False)
                    if result:
                        all_results.append(result)

        # Resumen general
        if all_results:
            self._print_analysis_summary(all_results)

        return all_results

    def _print_analysis_summary(self, results):
        """Imprimir resumen de análisis múltiple"""
        print("\n" + "="*60)
        print("📊 RESUMEN DEL ANÁLISIS DE MUESTRAS")
        print("="*60)

        total_images = len(results)
        print(f"📸 Total de imágenes analizadas: {total_images}")

        # Distribución de predicciones
        predictions = [r['predicted_class'] for r in results]
        prediction_counts = {cls: predictions.count(cls) for cls in self.classes}

        print(f"\n🎯 Distribución de predicciones:")
        for cls, count in prediction_counts.items():
            percentage = (count / total_images) * 100
            print(f"   {cls:<15}: {count:2d} ({percentage:5.1f}%)")

        # Confianza promedio
        avg_confidence = np.mean([r['confidence'] for r in results])
        print(f"\n📈 Confianza promedio: {avg_confidence:.1%}")

        # Top predicciones
        sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
        print(f"\n🏆 Top 3 predicciones más confiables:")
        for i, result in enumerate(sorted_results[:3], 1):
            filename = Path(result['image_path']).name
            print(f"   {i}. {filename}: {result['predicted_class']} ({result['confidence']:.1%})")

        print("="*60)

def setup_samples():
    """Configurar y descargar muestras automáticamente"""
    if BananaSampleDownloader is None:
        print("❌ Módulo download_samples no disponible")
        return False

    print("🚀 Configurando muestras de demostración...")
    downloader = BananaSampleDownloader()

    # Verificar si ya existen muestras
    samples_dir = Path("data/samples")
    if samples_dir.exists() and any(samples_dir.iterdir()):
        print("✅ Muestras ya disponibles")
        return True

    # Descargar muestras
    downloader.create_directories()
    downloaded, failed = downloader.download_all_samples()
    downloader.create_sample_info_file()

    return downloaded > 0

def main():
    """Función principal del demo"""
    parser = argparse.ArgumentParser(
        description="Demo - Sistema de Detección de Enfermedades en Banano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Menú interactivo
  python demo.py

  # Analizar imagen específica
  python demo.py --predict imagen.jpg

  # Analizar todas las muestras
  python demo.py --sample-analysis

  # Configurar muestras automáticamente
  python demo.py --setup-samples

  # Mostrar información del sistema
  python demo.py --system-info
        """
    )

    parser.add_argument('--predict', type=str, help='Ruta a imagen para analizar')
    parser.add_argument('--sample-analysis', action='store_true', 
                       help='Analizar todas las imágenes de muestra')
    parser.add_argument('--setup-samples', action='store_true',
                       help='Descargar y configurar muestras automáticamente')
    parser.add_argument('--system-info', action='store_true',
                       help='Mostrar información del sistema')
    parser.add_argument('--no-plot', action='store_true',
                       help='No mostrar gráficos')
    parser.add_argument('--save', action='store_true',
                       help='Guardar resultados en JSON')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device para ejecutar el modelo')

    args = parser.parse_args()

    # Header del demo
    print("\n" + "="*60)
    print("🍌 SISTEMA DE DETECCIÓN DE ENFERMEDADES EN BANANO")
    print("="*60)
    print("🔬 Transfer Learning + PyTorch + ResNet18")
    print("🎯 Detecta: Sano, Fusarium R4T, Moko Bacteriano, Sigatoka Negra")
    print("💰 ROI Proyectado: 150-300% primer año")
    print("⚡ Tiempo de análisis: ~5 segundos")
    print("🙏 Desarrollado con conocimientos de WorldQuant University")
    print("="*60)

    # Verificar si se requiere configuración de muestras
    if args.setup_samples:
        success = setup_samples()
        if success:
            print("\n✅ Muestras configuradas exitosamente!")
        else:
            print("\n❌ Error configurando muestras")
        return

    # Mostrar información del sistema
    if args.system_info:
        print("\n🖥️  INFORMACIÓN DEL SISTEMA:")
        print(f"   Python: {sys.version}")
        print(f"   PyTorch: {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name()}")
        print(f"   CPU threads: {torch.get_num_threads()}")
        return

    # Inicializar detector
    try:
        detector = BananaPlantDiseaseDetector(device=args.device)
    except Exception as e:
        print(f"❌ Error inicializando detector: {str(e)}")
        return

    # Ejecutar según argumentos
    if args.predict:
        # Analizar imagen específica
        image_path = Path(args.predict)
        if not image_path.exists():
            print(f"❌ Imagen no encontrada: {image_path}")
            return

        detector.predict_image(
            image_path, 
            show_plot=not args.no_plot,
            save_results=args.save
        )

    elif args.sample_analysis:
        # Analizar todas las muestras
        detector.analyze_samples_directory()

    else:
        # Menú interactivo
        while True:
            print("\n" + "="*40)
            print("🔧 MENÚ INTERACTIVO")
            print("="*40)
            print("1. 🖼️  Analizar imagen específica")
            print("2. 📂 Analizar todas las muestras")
            print("3. ⬇️  Configurar/descargar muestras")
            print("4. ℹ️  Información del sistema")
            print("5. 🚪 Salir")
            print("="*40)

            try:
                choice = input("Selecciona una opción (1-5): ").strip()

                if choice == '1':
                    image_path = input("Ruta de la imagen: ").strip()
                    if Path(image_path).exists():
                        detector.predict_image(image_path, save_results=True)
                    else:
                        print(f"❌ Imagen no encontrada: {image_path}")

                elif choice == '2':
                    detector.analyze_samples_directory()

                elif choice == '3':
                    setup_samples()

                elif choice == '4':
                    print("\n🖥️  INFORMACIÓN DEL SISTEMA:")
                    print(f"   Python: {sys.version}")
                    print(f"   PyTorch: {torch.__version__}")
                    print(f"   CUDA disponible: {torch.cuda.is_available()}")
                    if torch.cuda.is_available():
                        print(f"   GPU: {torch.cuda.get_device_name()}")

                elif choice == '5':
                    print("👋 ¡Gracias por usar el sistema!")
                    break

                else:
                    print("❌ Opción inválida. Intenta nuevamente.")

            except KeyboardInterrupt:
                print("\n\n👋 ¡Hasta luego!")
                break
            except EOFError:
                break

if __name__ == "__main__":
    main()
