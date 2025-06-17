"""
Setup configuration for Banana Disease Detection System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_long_description():
    """Read long description from README.md"""
    try:
        with open("README.md", "r", encoding="utf-8") as fh:
            return fh.read()
    except FileNotFoundError:
        return "Sistema de detección de enfermedades en banano usando Deep Learning"

# Read requirements
def read_requirements():
    """Read requirements from requirements.txt"""
    try:
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = []
            for line in fh.read().splitlines():
                if line and not line.startswith("#"):
                    requirements.append(line.strip())
            return requirements
    except FileNotFoundError:
        return [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
            "numpy>=1.21.0",
            "Pillow>=8.3.0",
            "opencv-python>=4.5.0",
            "scikit-learn>=1.0.0",
            "matplotlib>=3.5.0",
        ]

setup(
    name="banana-disease-detector",
    version="1.0.0",
    author="Jordan Villon",
    author_email="jordanviion@gmail.com",
    description="Sistema de detección de enfermedades en banano usando Deep Learning",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/jordanvt18/banana-disease-detection",
    project_urls={
        "Bug Reports": "https://github.com/jordanvt18/banana-disease-detection/issues",
        "Source": "https://github.com/jordanvt18/banana-disease-detection",
        "Documentation": "https://github.com/jordanvt18/banana-disease-detection/docs",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=2.12.0",
            "black>=21.9.0",
            "flake8>=3.9.0",
            "isort>=5.9.0",
        ],
        "mobile": [
            "tensorflow>=2.8.0",
            "onnx>=1.10.0",
            "onnxruntime>=1.9.0",
        ],
        "cloud": [
            "boto3>=1.18.0",
            "google-cloud-storage>=1.42.0",
        ],
        "webapp": [
            "flask>=2.0.0",
            "streamlit>=1.2.0",
            "fastapi>=0.70.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "banana-detect=src.cli:main",
            "banana-train=src.train:main",
            "banana-download=scripts.download_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.txt"],
        "data": ["samples/*"],
        "models": ["pretrained/*"],
    },
    keywords=[
        "deep learning",
        "computer vision",
        "agriculture",
        "plant disease",
        "pytorch",
        "transfer learning",
        "banana",
        "disease detection",
        "AI",
        "machine learning",
    ],
    zip_safe=False,
)
