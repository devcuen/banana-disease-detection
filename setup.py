"""
Setup configuration for Banana Disease Detection System
"""

import os
import subprocess
import sys
from pathlib import Path
import importlib.util

def check_python_version():
    """Verificar versión de Python"""
    if sys.version_info < (3, 7):
        print("❌ Se requiere Python 3.7 o superior")
        return False
    print(f"✅ Python {sys.version}")
    return True

def install_requirements():
    """Instalar dependencias desde requirements.txt"""
    requirements_file = Path("requirements.txt")

    if not requirements_file.exists():
        print("⚠️  requirements.txt no encontrado")
        return False

    try:
        print("📦 Instalando dependencias...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencias instaladas exitosamente")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error instalando dependencias: {e}")
        return False

def check_required_modules():
    """Verificar módulos requeridos"""
    required_modules = [
        'torch', 'torchvision', 'PIL', 'cv2', 'matplotlib', 
        'seaborn', 'numpy', 'requests'
    ]

    missing_modules = []

    for module in required_modules:
        try:
            if module == 'cv2':
                importlib.import_module('cv2')
            elif module == 'PIL':
                importlib.import_module('PIL')
            else:
                importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"❌ {module}")

    return len(missing_modules) == 0, missing_modules

def create_project_structure():
    """Crear estructura de directorios del proyecto"""
    directories = [
        "data/samples",
        "data/raw", 
        "data/processed",
        "models/pretrained",
        "models/checkpoints",
        "results",
        "logs",
        "scripts",
        "tests",
        "docs"
    ]

    print("📁 Creando estructura de directorios..")
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {directory}")

    return True

def download_sample_images():
    """Descargar imágenes de muestra"""
    try:
        from download_samples import BananaSampleDownloader

        print("🖼️  Descargando imágenes de muestra...")
        downloader = BananaSampleDownloader()
        downloader.create_directories()
        downloaded, failed = downloader.download_all_samples()
        downloader.create_sample_info_file()

        if downloaded > 0:
            print(f"✅ {downloaded} imágenes descargadas exitosamente")
            if failed > 0:
                print(f"⚠️  {failed} descargas fallaron")
            return True
        else:
            print("❌ No se pudieron descargar las imágenes")
            return False

    except ImportError:
        print("❌ Módulo download_samples no encontrado")
        return False
    except Exception as e:
        print(f"❌ Error descargando muestras: {e}")
        return False

def verify_installation():
    """Verificar que todo esté instalado correctamente"""
    try:
        print("🔍 Verificando instalación...")

        # Verificar PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        print(f"   CUDA disponible: {torch.cuda.is_available()}")

        # Verificar estructura de archivos principales
        required_files = ["demo.py", "download_samples.py", "requirements.txt"]
        for file in required_files:
            if Path(file).exists():
                print(f"✅ {file}")
            else:
                print(f"⚠️  {file} no encontrado")

        # Verificar imágenes de muestra
        samples_dir = Path("data/samples")
        if samples_dir.exists():
            image_count = sum(1 for f in samples_dir.rglob("*.jpg"))
            print(f"✅ {image_count} imágenes de muestra disponibles")

        return True

    except Exception as e:
        print(f"❌ Error en verificación: {e}")
        return False

def main():
    """Función principal de configuración"""
    print("\n" + "="*60)
    print("🚀 CONFIGURACIÓN DEL PROYECTO")
    print("🍌 Sistema de Detección de Enfermedades en Banano")
    print("="*60)

    steps = [
        ("Verificar Python", check_python_version),
        ("Crear estructura", create_project_structure),
        ("Instalar dependencias", install_requirements),
        ("Verificar módulos", lambda: check_required_modules()[0]),
        ("Descargar muestras", download_sample_images),
        ("Verificar instalación", verify_installation)
    ]

    success_count = 0

    for step_name, step_function in steps:
        print(f"\n🔧 {step_name}...")
        try:
            if step_function():
                success_count += 1
                print(f"   ✅ {step_name} completado")
            else:
                print(f"   ❌ {step_name} falló")
        except Exception as e:
            print(f"   ❌ Error en {step_name}: {e}")

    print("\n" + "="*60)
    print("📊 RESUMEN DE CONFIGURACIÓN")
    print("="*60)
    print(f"Pasos completados: {success_count}/{len(steps)}")

    if success_count == len(steps):
        print("\n🎉 ¡Configuración completada exitosamente!")
        print("\n🚀 Para ejecutar el demo:")
        print("   python demo.py")
        print("\n📖 Para más información:")
        print("   python demo.py --help")
    else:
        print(f"\n⚠️  Configuración parcial ({success_count}/{len(steps)})")
        print("\n💡 Revisa los errores anteriores y ejecuta nuevamente:")
        print("   python setup.py")

    print("="*60)

if __name__ == "__main__":
    main()
