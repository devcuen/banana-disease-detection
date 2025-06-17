#!/usr/bin/env python3
"""
Script para descargar imágenes de muestra de enfermedades en banano
Autor: J.V Sistema de Detección de Enfermedades en Banano
"""

import os
import requests
import shutil
from urllib.parse import urlparse
from pathlib import Path
import time
import hashlib

class BananaSampleDownloader:
    def __init__(self):
        self.base_dir = Path("data/samples")
        self.samples_info = {
            'sano': [
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1750123352/pplx_project_search_images/5631ff0de8e4dcb30e65b9a62d6b8a99f2ccae67.jpg',
                    'filename': 'banana_sano_1.jpg',
                    'description': 'Plantación saludable de banano en Ecuador'
                },
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1749802641/pplx_project_search_images/d88820a6fac21c173e0267418c5002cbf36f7d62.jpg',
                    'filename': 'banana_sano_2.jpg',
                    'description': 'Plantas de banano sanas con bolsas protectoras'
                },
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1750123352/pplx_project_search_images/0ed414d84f3c734950fb870f89af0526ffc8db8c.jpg',
                    'filename': 'banana_sano_3.jpg',
                    'description': 'Planta de banano saludable con frutos en desarrollo'
                }
            ],
            'fusarium_r4t': [
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1750123353/pplx_project_search_images/3d72cc63b2d39c067b7c3f546b8a96aea0cb8239.jpg',
                    'filename': 'fusarium_r4t_1.jpg',
                    'description': 'Síntomas de Fusarium R4T con amarillamiento y marchitez'
                },
                {
                    'url': 'https://agriculture.gov.tt/wp-content/uploads/2020/07/TR4-fusarium-wilt.jpg',
                    'filename': 'fusarium_r4t_2.jpg',
                    'description': 'Marchitez por Fusarium en etapa avanzada'
                }
            ],
            'sigatoka_negra': [
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1750123352/pplx_project_search_images/e85d4affdaf9ad1057803127b957ae6284fac751.jpg',
                    'filename': 'sigatoka_negra_1.jpg',
                    'description': 'Síntomas típicos de Sigatoka Negra con manchas características'
                },
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1750123353/pplx_project_search_images/bf87070395ebe23e9aa9acc23fd38593f795078c.jpg',
                    'filename': 'sigatoka_negra_2.jpg',
                    'description': 'Hoja de banano con manchas negras de Sigatoka'
                },
                {
                    'url': 'https://www.seipasa.com/files/images/240307-sigatoka-negra-banano-ecuador-y-colombia_blog.jpg',
                    'filename': 'sigatoka_negra_3.jpg',
                    'description': 'Síntomas de Sigatoka Negra en campo ecuatoriano'
                }
            ],
            'moko_bacteriano': [
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1750123352/pplx_project_search_images/cda628c654940c8a2ab16cc223d4dd0ba984e797.jpg',
                    'filename': 'moko_bacteriano_1.jpg',
                    'description': 'Síntomas completos de Moko Bacteriano incluyendo cortes transversales'
                },
                {
                    'url': 'https://pplx-res.cloudinary.com/image/upload/v1750112803/pplx_project_search_images/f102290319935a80a322b6f95e1a21e2ce4bd3b6.jpg',
                    'filename': 'moko_bacteriano_2.jpg',
                    'description': 'Cortes transversales mostrando descoloración vascular'
                },
                {
                    'url': 'https://apps.lucidcentral.org/pppw_v11/images/entities/banana_moko_disease_525/bunch_closeup.jpg',
                    'filename': 'moko_bacteriano_3.jpg',
                    'description': 'Frutos afectados por Moko Bacteriano'
                }
            ]
        }

    def create_directories(self):
        """Crear estructura de directorios para las muestras"""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        for category in self.samples_info.keys():
            category_dir = self.base_dir / category
            category_dir.mkdir(exist_ok=True)
            print(f"✅ Directorio creado: {category_dir}")

    def download_image(self, url, filepath, description=""):
        """Descargar una imagen desde URL"""
        try:
            print(f"📥 Descargando: {description}")
            print(f"   URL: {url}")
            print(f"   Destino: {filepath}")

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=30, stream=True)
            response.raise_for_status()

            with open(filepath, 'wb') as f:
                shutil.copyfileobj(response.raw, f)

            # Verificar que el archivo se descargó correctamente
            if filepath.stat().st_size > 0:
                print(f"   ✅ Descarga exitosa: {filepath.stat().st_size} bytes")
                return True
            else:
                print(f"   ❌ Error: Archivo vacío")
                return False

        except Exception as e:
            print(f"   ❌ Error descargando {url}: {str(e)}")
            return False

    def download_all_samples(self):
        """Descargar todas las imágenes de muestra"""
        print("🚀 Iniciando descarga de imágenes de muestra...")
        print("=" * 60)

        total_images = sum(len(images) for images in self.samples_info.values())
        downloaded = 0
        failed = 0

        for category, images in self.samples_info.items():
            print(f"\n📂 Procesando categoría: {category.upper()}")
            print("-" * 40)

            category_dir = self.base_dir / category

            for img_info in images:
                filepath = category_dir / img_info['filename']

                # Saltar si ya existe
                if filepath.exists():
                    print(f"⏭️  Ya existe: {img_info['filename']}")
                    downloaded += 1
                    continue

                success = self.download_image(
                    img_info['url'], 
                    filepath, 
                    img_info['description']
                )

                if success:
                    downloaded += 1
                else:
                    failed += 1

                # Pequeña pausa entre descargas
                time.sleep(1)

        print("\n" + "=" * 60)
        print(f"📊 RESUMEN DE DESCARGA:")
        print(f"   Total de imágenes: {total_images}")
        print(f"   ✅ Descargadas exitosamente: {downloaded}")
        print(f"   ❌ Fallidas: {failed}")
        print(f"   📁 Directorio: {self.base_dir.absolute()}")

        return downloaded, failed

    def create_sample_info_file(self):
        """Crear archivo con información de las muestras"""
        info_content = """# Imágenes de Muestra - Sistema de Detección de Enfermedades en Banano

## Descripción de las Muestras

### 🌱 Plantas Sanas (sano/)
- **banana_sano_1.jpg**: Plantación saludable de banano en Ecuador
- **banana_sano_2.jpg**: Plantas de banano sanas con bolsas protectoras
- **banana_sano_3.jpg**: Planta de banano saludable con frutos en desarrollo

### 🟡 Fusarium R4T (fusarium_r4t/)
- **fusarium_r4t_1.jpg**: Síntomas de Fusarium R4T con amarillamiento y marchitez
- **fusarium_r4t_2.jpg**: Marchitez por Fusarium en etapa avanzada

### ⚫ Sigatoka Negra (sigatoka_negra/)
- **sigatoka_negra_1.jpg**: Síntomas típicos de Sigatoka Negra con manchas características
- **sigatoka_negra_2.jpg**: Hoja de banano con manchas negras de Sigatoka
- **sigatoka_negra_3.jpg**: Síntomas de Sigatoka Negra en campo ecuatoriano

### 🔴 Moko Bacteriano (moko_bacteriano/)
- **moko_bacteriano_1.jpg**: Síntomas completos de Moko Bacteriano incluyendo cortes transversales
- **moko_bacteriano_2.jpg**: Cortes transversales mostrando descoloración vascular
- **moko_bacteriano_3.jpg**: Frutos afectados por Moko Bacteriano

## Uso en el Demo

Estas imágenes pueden ser utilizadas con el comando:

```bash
python demo.py --sample-analysis
```

O individualmente:

```bash
python demo.py --predict data/samples/sano/banana_sano_1.jpg
```

## Créditos

Las imágenes han sido obtenidas de fuentes científicas y repositorios públicos, 
incluyendo CABI Digital Library, instituciones agrícolas y centros de investigación.
"""

        info_file = self.base_dir / "README.md"
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(info_content)

        print(f"📄 Archivo de información creado: {info_file}")

def main():
    """Función principal"""
    downloader = BananaSampleDownloader()

    # Crear directorios
    downloader.create_directories()

    # Descargar imágenes
    downloaded, failed = downloader.download_all_samples()

    # Crear archivo de información
    downloader.create_sample_info_file()

    print("\n🎉 ¡Descarga de muestras completada!")

    if failed > 0:
        print(f"⚠️  Algunas descargas fallaron. Puedes intentar nuevamente ejecutando el script.")

    return downloaded > 0

if __name__ == "__main__":
    main()
