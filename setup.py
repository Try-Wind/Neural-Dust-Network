#!/usr/bin/env python3
"""
Neural Dust Network - Setup Configuration
==========================================

Installation script for the Neural Dust Network package.
Enables decentralized AI learning across devices without data sharing.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    try:
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Neural Dust Network - Decentralized AI Learning Platform"

# Read requirements from requirements.txt
def read_requirements():
    try:
        with open("requirements.txt", "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    except FileNotFoundError:
        return [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "lz4>=4.0.0",
            "PyNaCl>=1.5.0"
        ]

setup(
    # Package metadata
    name="neural-dust-network",
    version="1.0.0",
    author="Neural Dust Network Contributors",
    author_email="contributors@neural-dust-network.org",
    description="Decentralized AI learning platform with privacy-preserving weight sharing",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/neural-dust-network",
    project_urls={
        "Bug Reports": "https://github.com/your-username/neural-dust-network/issues",
        "Source": "https://github.com/your-username/neural-dust-network",
        "Documentation": "https://neural-dust-network.readthedocs.io/",
    },
    
    # Package discovery
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Dependencies
    install_requires=read_requirements(),
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "mnist": [
            "torchvision>=0.15.0",
        ],
        "networking": [
            "netifaces>=0.11.0",
            "zeroconf>=0.47.0",
        ],
        "mobile": [
            "kivy>=2.1.0",
            "kivymd>=1.1.0",
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Package classification
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
        "Topic :: System :: Distributed Computing",
        "Topic :: Security :: Cryptography",
    ],
    
    # Keywords for discovery
    keywords="AI, machine learning, federated learning, decentralized, privacy, neural networks, gossip protocol",
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "neural-dust-demo=neural_dust.examples.basic_demo:main",
            "neural-dust-node=neural_dust.tools.node_cli:main",
        ],
    },
    
    # Include additional files
    include_package_data=True,
    zip_safe=False,
    
    # License
    license="MIT",
)