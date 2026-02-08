"""
Depth Estimation with Semantic Segmentation

A multi-modal depth estimation pipeline using DepthNet architecture.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="depth-estimation-semantic-segmentation",
    version="1.0.0",
    author="Dhruv Garg",
    author_email="",
    description="Multi-modal depth estimation using RGB, sparse depth, and semantic segmentation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DhruvGarg111/Depth-Estimation-with-Semantic-Segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.0.0",
        "scipy>=1.10.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "h5py>=3.8.0",
        "pyyaml>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "isort>=5.12.0",
        ],
        "notebook": [
            "ipykernel>=6.0.0",
            "ipywidgets>=8.0.0",
            "jupyterlab>=4.0.0",
        ],
        "visualization": [
            "torchview>=0.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "depthnet-train=train:main",
            "depthnet-inference=inference:main",
        ],
    },
)
