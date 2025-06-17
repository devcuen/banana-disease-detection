"""
Banana Plant Disease Detector using Transfer Learning with ResNet18
Sistema de detección de enfermedades en plantas de banano
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import os
import warnings
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

class BananaPlantDiseaseDetector(nn.Module):
    """
    Detector de enfermedades en plantas de banano usando Transfer Learning
    con ResNet18 preentrenado en ImageNet.
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        """
        Inicializar el modelo detector

        Args:
            num_classes (int): Número de clases a detectar (4 por defecto)
            pretrained (bool): Usar pesos preentrenados de ImageNet
        """
        super(BananaPlantDiseaseDetector, self).__init__()

        # Clases que puede detectar el modelo
        self.classes = [
            'Sano',           # Plantas saludables
            'Fusarium_R4T',   # Marchitez por Fusarium R4T
            'Moko_Bacteriano', # Moko bacteriano (Ralstonia solanacearum)
            'Sigatoka_Negra'  # Sigatoka negra (Mycosphaerella fijiensis)
        ]

        # Cargar ResNet18 preentrenado
        self.backbone = models.resnet18(pretrained=pretrained)

        # Congelar las primeras capas para transfer learning
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False

        # Modificar la capa final para nuestras clases
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        # Configurar transformaciones de imagen
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            )
        ])

        print(f"✅ BananaPlantDiseaseDetector inicializado")
        print(f"📊 Clases detectables: {', '.join(self.classes)}")
        print(f"🔢 Parámetros totales: {sum(p.numel() for p in self.parameters()):,}")
        print(f"🎯 Parámetros entrenables: {sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass del modelo"""
        return self.backbone(x)

    def predict_image(self, image_path: str, device: str = 'cpu') -> Dict:
        """
        Predecir enfermedad en una imagen de hoja de banano

        Args:
            image_path (str): Ruta a la imagen
            device (str): Dispositivo a usar ('cpu' o 'cuda')

        Returns:
            Dict: Diccionario con predicción y confianzas
        """
        try:
            # Cargar y preprocesar imagen
            image = Image.open(image_path).convert('RGB')
            original_size = image.size

            # Aplicar transformaciones
            input_tensor = self.transform(image).unsqueeze(0)

            # Mover a dispositivo
            self.to(device)
            input_tensor = input_tensor.to(device)

            # Hacer predicción
            self.eval()
            with torch.no_grad():
                outputs = self(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class_idx].item()

            # Preparar resultados
            result = {
                'image_path': image_path,
                'predicted_class': self.classes[predicted_class_idx],
                'predicted_class_idx': predicted_class_idx,
                'confidence': confidence,
                'all_probabilities': {
                    class_name: prob.item() 
                    for class_name, prob in zip(self.classes, probabilities[0])
                },
                'image_size': original_size,
                'status': 'success'
            }

            return result

        except Exception as e:
            return {
                'image_path': image_path,
                'error': str(e),
                'status': 'error'
            }

    def predict_batch(self, image_paths: List[str], device: str = 'cpu') -> List[Dict]:
        """
        Predecir múltiples imágenes en lote

        Args:
            image_paths (List[str]): Lista de rutas de imágenes
            device (str): Dispositivo a usar

        Returns:
            List[Dict]: Lista de resultados
        """
        results = []

        print(f"🔄 Procesando {len(image_paths)} imágenes...")

        for i, image_path in enumerate(image_paths):
            print(f"📸 Procesando imagen {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            result = self.predict_image(image_path, device)
            results.append(result)

            if result['status'] == 'success':
                print(f"   ✅ {result['predicted_class']} - Confianza: {result['confidence']:.1%}")
            else:
                print(f"   ❌ Error: {result['error']}")

        return results

    def get_model_info(self) -> Dict:
        """
        Obtener información del modelo

        Returns:
            Dict: Información del modelo
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'model_name': 'BananaPlantDiseaseDetector',
            'backbone': 'ResNet18',
            'num_classes': len(self.classes),
            'classes': self.classes,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'parameter_size_mb': total_params * 4 / (1024**2),  # Assuming float32
            'input_size': (224, 224),
            'framework': 'PyTorch'
        }

    def save_model(self, filepath: str):
        """
        Guardar modelo entrenado

        Args:
            filepath (str): Ruta donde guardar el modelo
        """
        try:
            torch.save({
                'model_state_dict': self.state_dict(),
                'classes': self.classes,
                'model_info': self.get_model_info()
            }, filepath)
            print(f"✅ Modelo guardado en: {filepath}")
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")

    def load_model(self, filepath: str, device: str = 'cpu'):
        """
        Cargar modelo preentrenado

        Args:
            filepath (str): Ruta del modelo guardado
            device (str): Dispositivo donde cargar
        """
        try:
            checkpoint = torch.load(filepath, map_location=device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.classes = checkpoint.get('classes', self.classes)
            print(f"✅ Modelo cargado desde: {filepath}")
            return checkpoint.get('model_info', {})
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return {}


def create_sample_detector() -> BananaPlantDiseaseDetector:
    """
    Crear un detector de muestra para pruebas

    Returns:
        BananaPlantDiseaseDetector: Instancia del detector
    """
    print("🚀 Creando detector de muestra...")
    detector = BananaPlantDiseaseDetector(num_classes=4, pretrained=True)

    # Mostrar información del modelo
    info = detector.get_model_info()
    print("\n📊 Información del modelo:")
    for key, value in info.items():
        if key != 'classes':
            print(f"   {key}: {value}")

    return detector


def demonstrate_model():
    """
    Demostración del funcionamiento del modelo
    """
    print("🍌 === DEMO: Sistema de Detección de Enfermedades en Banano ===\n")

    # Crear detector
    detector = create_sample_detector()

    # Información del modelo
    print("\n📋 Clases que puede detectar:")
    for i, class_name in enumerate(detector.classes):
        print(f"   {i}: {class_name}")

    print("\n✨ ¡Detector listo para usar!")
    print("💡 Para usar el detector:")
    print("   result = detector.predict_image('ruta/a/imagen.jpg')")
    print("   print(f'Predicción: {result["predicted_class"]} - Confianza: {result["confidence"]:.1%}')")


if __name__ == "__main__":
    demonstrate_model()
