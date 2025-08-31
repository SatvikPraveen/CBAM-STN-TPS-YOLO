# setup.py
"""
Enhanced setup configuration for CBAM-STN-TPS-YOLO
Professional package setup with comprehensive dependencies and metadata
"""

from setuptools import setup, find_packages
from pathlib import Path
import os

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8') if (this_directory / "README.md").exists() else ""

# Read requirements from requirements.txt
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    # Fallback requirements
    requirements = [
        "torch>=1.12.0,<2.1.0",
        "torchvision>=0.13.0,<0.16.0",
        "numpy>=1.21.0",
        "opencv-python>=4.6.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.9.0,<2.0.0",
        "scikit-learn>=1.1.0",
        "tqdm>=4.64.0",
        "wandb>=0.13.0",
        "pyyaml>=6.0",
        "tensorboard>=2.9.0",
        "albumentations>=1.2.0",
        "Pillow>=9.0.0",
        "pandas>=1.4.0",
        "onnx>=1.12.0",
        "onnxruntime>=1.12.0",
    ]

# Development requirements
dev_requirements = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=22.0.0",
    "flake8>=4.0.0",
    "isort>=5.10.0",
    "mypy>=0.950",
    "pre-commit>=2.17.0",
    "jupyter>=1.0.0",
    "notebook>=6.4.0",
    "ipywidgets>=7.7.0",
]

# Documentation requirements
doc_requirements = [
    "sphinx>=4.5.0",
    "sphinx-rtd-theme>=1.0.0",
    "sphinx-autodoc-typehints>=1.17.0",
    "myst-parser>=0.17.0",
]

# Deployment requirements
deploy_requirements = [
    "onnxsim>=0.4.0",
    "tensorrt>=8.4.0",  # Optional, for NVIDIA deployment
    "openvino>=2022.1.0",  # Optional, for Intel deployment
]

setup(
    # Basic package information
    name="cbam-stn-tps-yolo",
    version="2.0.0",
    
    # Package description
    description="CBAM-STN-TPS-YOLO: Enhancing Agricultural Object Detection through Spatially Adaptive Attention Mechanisms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Satvik Praveen, Yoonsung Jung",
    author_email="satvikpraveen_164@tamu.edu, yojung@tamu.edu",
    maintainer="Satvik Praveen",
    maintainer_email="satvikpraveen_164@tamu.edu",
    
    # URLs
    url="https://github.com/your-username/cbam-stn-tps-yolo",
    download_url="https://github.com/your-username/cbam-stn-tps-yolo/archive/v2.0.0.tar.gz",
    project_urls={
        "Bug Reports": "https://github.com/your-username/cbam-stn-tps-yolo/issues",
        "Source": "https://github.com/your-username/cbam-stn-tps-yolo",
        "Documentation": "https://cbam-stn-tps-yolo.readthedocs.io/",
        "Paper": "https://arxiv.org/abs/your-paper-id",
    },
    
    # Package configuration
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*"]),
    package_dir={"": "."},
    
    # Include package data
    include_package_data=True,
    package_data={
        "cbam_stn_tps_yolo": [
            "config/*.yaml",
            "config/*.yml",
            "data/sample_data/*",
            "pretrained_models/*.pth",
        ],
    },
    
    # Dependencies
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "docs": doc_requirements,
        "deploy": deploy_requirements,
        "all": dev_requirements + doc_requirements + deploy_requirements,
        "gpu": ["torch>=1.12.0+cu116", "torchvision>=0.13.0+cu116"],
        "cpu": ["torch>=1.12.0+cpu", "torchvision>=0.13.0+cpu"],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Entry points for command-line scripts
    entry_points={
        "console_scripts": [
            "cbam-train=src.training.trainer:main",
            "cbam-predict=src.inference.predict:main",
            "cbam-evaluate=src.utils.evaluation:main",
            "cbam-export=src.inference.predict:export_main",
            "cbam-experiment=experiments.run_experiments:main",
        ],
    },
    
    # Classifiers for PyPI
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Programming Language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        
        # Operating System
        "Operating System :: OS Independent",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
        
        # Topic
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
        
        # Natural Language
        "Natural Language :: English",
        
        # Framework
        "Framework :: Jupyter",
        
        # Environment
        "Environment :: GPU :: NVIDIA CUDA",
        "Environment :: Console",
    ],
    
    # Keywords for discoverability
    keywords=[
        "computer-vision", "object-detection", "agricultural-ai", "attention-mechanism",
        "spatial-transformer", "yolo", "cbam", "thin-plate-spline", "pytorch",
        "machine-learning", "deep-learning", "precision-agriculture", "remote-sensing",
        "multi-spectral", "crop-monitoring"
    ],
    
    # License
    license="MIT",
    
    # Zip safety
    zip_safe=False,
    
    # Test suite
    test_suite="tests",
    tests_require=dev_requirements,
    
    # Additional metadata
    platforms=["any"],
)

# =============================================================================
# POST-INSTALL CONFIGURATION
# =============================================================================

def post_install():
    """Post-installation configuration"""
    import os
    import subprocess
    import sys
    
    print("🚀 CBAM-STN-TPS-YOLO installation completed!")
    print("=" * 60)
    
    # Check PyTorch installation
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} detected")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("⚠️  PyTorch not found. Please install PyTorch manually:")
        print("   CPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print("   GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    
    # Check other critical dependencies
    critical_deps = ["cv2", "numpy", "matplotlib", "yaml"]
    for dep in critical_deps:
        try:
            __import__(dep)
            print(f"✅ {dep} installed")
        except ImportError:
            print(f"❌ {dep} not found")
    
    # Create necessary directories
    directories = [
        "data", "results", "checkpoints", "logs", "plots"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
    
    print("\n📋 Next steps:")
    print("1. Download sample data: python -m src.data.download_sample_data")
    print("2. Run tests: pytest tests/")
    print("3. Start training: cbam-train --config config/training_configs.yaml")
    print("4. View documentation: https://cbam-stn-tps-yolo.readthedocs.io/")
    
    print("\n🌱 Happy agricultural AI research!")

if __name__ == "__main__":
    # Run post-install if this script is executed directly
    import sys
    if "install" in sys.argv:
        post_install()