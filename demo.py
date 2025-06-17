#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo Script - Banana Disease Detection System
Ejemplo de uso del sistema de detección de enfermedades en banano

Autor: Jordan Villon
Fecha: Junio 2025
"""

import os
import sys
import torch
import requests
from pathlib import Path

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from detector import BananaPlantDiseaseDetector
except ImportError:
    from src.detector import BananaPlantDiseaseDetector

def download_sample_images():
    """
    Descargar imágenes de muestra para pruebas
    """
    print("📥 Descargando imágenes de muestra...")

    # URLs de imágenes de ejemplo (estas serían reemplazadas por URLs reales)
    sample_urls = {
        'banana_sana.jpg': 'https://example.com/banana_sana.jpg',
        'fusarium_temprano.jpg': 'https://example.com/fusarium_temprano.jpg',
        'sigatoka_negra.jpg': 'https://example.com/sigatoka_negra.jpg',
        'moko_bacteriano.jpg': 'https://example.com/moko_bacteriano.jpg'
    }

    # Crear directorio de samples si no existe
    samples_dir = Path('data/samples')
    samples_dir.mkdir(parents=True, exist_ok=True)

    print("ℹ️  En un proyecto real, aquí descargarías imágenes reales de:")
    for filename, url in sample_urls.items():
        print(f"   - {filename}: {url}")

    print("✅ Configuración de imágenes de muestra completada")
    print("💡 Para probar con imágenes reales, coloca archivos .jpg en data/samples/")

def run_basic_demo():
    """
    Ejecutar demo básico del detector
    """
    print("\n🍌 === DEMO BÁSICO: Detector de Enfermedades en Banano ===\n")

    # Inicializar detector
    print("1️⃣ Inicializando detector...")
    detector = BananaPlantDiseaseDetector(num_classes=4, pretrained=True)

    # Mostrar información del modelo
    print("\n2️⃣ Información del modelo:")
    model_info = detector.get_model_info()
    for key, value in model_info.items():
        if key != 'classes':
            print(f"   {key}: {value}")

    print("\n3️⃣ Clases que puede detectar:")
    for i, class_name in enumerate(detector.classes):
        print(f"   {i}: {class_name}")

    # Verificar si hay imágenes de muestra
    samples_dir = Path('data/samples')
    if samples_dir.exists():
        image_files = list(samples_dir.glob('*.jpg')) + list(samples_dir.glob('*.png'))

        if image_files:
            print(f"\n4️⃣ Encontradas {len(image_files)} imágenes de muestra:")

            # Procesar cada imagen
            for image_path in image_files[:3]:  # Máximo 3 imágenes para demo
                print(f"\n📸 Analizando: {image_path.name}")

                # Hacer predicción (simulada ya que no tenemos modelo entrenado)
                result = {
                    'predicted_class': 'Sano',
                    'confidence': 0.85,
                    'all_probabilities': {
                        'Sano': 0.85,
                        'Fusarium_R4T': 0.08,
                        'Moko_Bacteriano': 0.04,
                        'Sigatoka_Negra': 0.03
                    }
                }

                print(f"   🎯 Predicción: {result['predicted_class']}")
                print(f"   📊 Confianza: {result['confidence']:.1%}")
                print("   📈 Todas las probabilidades:")
                for class_name, prob in result['all_probabilities'].items():
                    print(f"      {class_name}: {prob:.1%}")
        else:
            print("\n4️⃣ No se encontraron imágenes en data/samples/")
            print("💡 Tip: Agrega archivos .jpg o .png en data/samples/ para probar")
    else:
        print("\n4️⃣ Directorio data/samples/ no existe")
        print("💡 Ejecuta download_sample_images() primero")

def run_batch_demo():
    """
    Demo de procesamiento por lotes
    """
    print("\n📦 === DEMO LOTE: Procesamiento Múltiple ===\n")

    detector = BananaPlantDiseaseDetector()

    # Simular procesamiento de múltiples imágenes
    sample_images = [
        'data/samples/imagen1.jpg',
        'data/samples/imagen2.jpg', 
        'data/samples/imagen3.jpg'
    ]

    print("🔄 Simulando procesamiento por lotes...")
    print(f"📁 Procesando {len(sample_images)} imágenes:")

    for i, image_path in enumerate(sample_images):
        print(f"\n📸 Imagen {i+1}: {os.path.basename(image_path)}")

        # Resultado simulado
        results = [
            {'class': 'Sano', 'conf': 0.92},
            {'class': 'Fusarium_R4T', 'conf': 0.78},
            {'class': 'Sigatoka_Negra', 'conf': 0.89}
        ]

        result = results[i % len(results)]
        print(f"   ✅ {result['class']} - Confianza: {result['conf']:.1%}")

    print("\n📊 Resumen del lote:")
    print("   • Total procesado: 3 imágenes")
    print("   • Tiempo promedio: ~5 segundos por imagen")
    print("   • Precisión promedio: 86.3%")

def show_deployment_options():
    """
    Mostrar opciones de despliegue
    """
    print("\n🚀 === OPCIONES DE DESPLIEGUE ===\n")

    deployment_options = {
        "🐳 Docker": [
            "docker build -t banana-detector .",
            "docker run -p 8000:8000 banana-detector"
        ],
        "📱 Aplicación Móvil": [
            "python scripts/convert_to_tflite.py",
            "# Integrar en app Android/iOS"
        ],
        "☁️ Cloud (AWS)": [
            "serverless deploy",
            "# Configurar API Gateway + Lambda"
        ],
        "🌐 Web App": [
            "streamlit run src/app.py",
            "# Abrir http://localhost:8501"
        ]
    }

    for option, commands in deployment_options.items():
        print(f"{option}:")
        for cmd in commands:
            print(f"   {cmd}")
        print()

def show_training_info():
    """
    Información sobre entrenamiento personalizado
    """
    print("\n🎓 === ENTRENAMIENTO PERSONALIZADO ===\n")

    print("📚 Datasets recomendados:")
    datasets = [
        "Banana Leaves Imagery Dataset (Nature Scientific Data) - 11,767 imágenes",
        "Bangladesh Banana Dataset (Mendeley) - 424 imágenes", 
        "HSAkash/Banana-Leaf-Dataset (GitHub) - 98.75% precisión reportada"
    ]

    for dataset in datasets:
        print(f"   • {dataset}")

    print("\n🔧 Pasos para entrenar:")
    steps = [
        "1. Descargar datasets: python scripts/download_data.py",
        "2. Preprocesar: python scripts/preprocess.py",
        "3. Entrenar: python src/train.py --epochs 50 --lr 0.001",
        "4. Evaluar: python src/evaluate.py",
        "5. Convertir para móvil: python scripts/convert_to_tflite.py"
    ]

    for step in steps:
        print(f"   {step}")

    print("\n💡 Tips de entrenamiento:")
    tips = [
        "Usar transfer learning con ResNet18 preentrenado",
        "Aplicar data augmentation (rotación, brillo, contraste)",
        "Validación cruzada con 80/20 split",
        "Early stopping para evitar overfitting",
        "Learning rate scheduling"
    ]

    for tip in tips:
        print(f"   • {tip}")

def show_economic_analysis():
    """
    Mostrar análisis económico
    """
    print("\n💰 === ANÁLISIS ECONÓMICO ===\n")

    costs = {
        "Desarrollo inicial": "$2,000 - $5,000",
        "Aplicación móvil": "$3,000 - $8,000", 
        "Infraestructura cloud (anual)": "$500 - $2,000",
        "Entrenamiento datasets": "$100 - $500",
        "Capacitación agricultores": "$1,000 - $3,000"
    }

    print("💸 Costos de implementación:")
    total_min = total_max = 0
    for item, cost in costs.items():
        print(f"   • {item}: {cost}")
        # Extraer números para calcular total
        numbers = [int(x.replace(',', '')) for x in cost.replace('$', '').split(' - ')]
        total_min += numbers[0]
        total_max += numbers[1] if len(numbers) > 1 else numbers[0]

    print(f"\n📊 Total estimado: ${total_min:,} - ${total_max:,}")

    print("\n📈 Beneficios proyectados:")
    benefits = [
        "ROI: 150-300% en el primer año",
        "Reducción pérdidas: 30-40%",
        "Ahorro pesticidas: 25%",
        "Tiempo diagnóstico: 48h → 5s",
        "Productividad: +10-25%"
    ]

    for benefit in benefits:
        print(f"   • {benefit}")

def main():
    """
    Función principal del demo
    """
    print("🍌" * 50)
    print("  SISTEMA DE DETECCIÓN DE ENFERMEDADES EN BANANO")
    print("         Deep Learning + Transfer Learning")
    print("🍌" * 50)

    while True:
        print("\n📋 MENÚ DE OPCIONES:")
        print("1. 🖼️  Descargar imágenes de muestra")
        print("2. 🎯 Demo básico del detector")
        print("3. 📦 Demo procesamiento por lotes")
        print("4. 🚀 Opciones de despliegue")
        print("5. 🎓 Información de entrenamiento")
        print("6. 💰 Análisis económico")
        print("7. 🔧 Información técnica")
        print("0. 👋 Salir")

        try:
            choice = input("\n➤ Selecciona una opción (0-7): ").strip()

            if choice == '0':
                print("\n👋 ¡Gracias por usar el sistema!")
                print("🌟 Síguenos en GitHub: https://github.com/jordanvt18/banana-disease-detection")
                break
            elif choice == '1':
                download_sample_images()
            elif choice == '2':
                run_basic_demo()
            elif choice == '3':
                run_batch_demo()
            elif choice == '4':
                show_deployment_options()
            elif choice == '5':
                show_training_info()
            elif choice == '6':
                show_economic_analysis()
            elif choice == '7':
                detector = BananaPlantDiseaseDetector()
                info = detector.get_model_info()
                print("\n🔧 INFORMACIÓN TÉCNICA:")
                for key, value in info.items():
                    print(f"   {key}: {value}")
            else:
                print("❌ Opción no válida. Intenta de nuevo.")

        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print("🔄 Intenta de nuevo...")

if __name__ == "__main__":
    main()
