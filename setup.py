#!/usr/bin/env python3
"""
Setup script for Ottoman NER package
"""

from pathlib import Path
from setuptools import find_packages, setup

# Load package version from single source of truth
version_path = Path(__file__).parent / "ottoman_ner" / "_version.py"
version_ns = {}
exec(version_path.read_text(encoding="utf-8"), version_ns)
PACKAGE_VERSION = version_ns["__version__"]

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r', encoding='utf-8') as f:
        requirements = [
            line.strip() for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('# ')
        ]
else:
    requirements = [
        "torch>=1.9.0",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "seqeval>=1.2.0",
        "tqdm>=4.62.0",
        "PyYAML>=6.0",
    ]

setup(
    name="ottoman-ner",
    version=PACKAGE_VERSION,
    author="Fatih Burak Karagöz",
    author_email="fatihburak@pm.me",
    description="Ottoman Turkish Named Entity Recognition toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fbkaragoz/ottoman-ner",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "isort>=5.9.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "full": [
            "tensorboard>=2.8.0",
            "wandb>=0.12.0",
            "mlflow>=2.0.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
        ]
    },
    entry_points={
        'console_scripts': [
            'ottoman-ner=ottoman_ner.cli:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "nlp",
        "ner", 
        "named-entity-recognition",
        "ottoman-turkish",
        "transformers",
        "bert",
    ],
    project_urls={
        "Bug Reports": "https://github.com/fbkaragoz/ottoman-ner/issues",
        "Source": "https://github.com/fbkaragoz/ottoman-ner",
    },
)
