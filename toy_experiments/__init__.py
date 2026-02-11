"""
Toy Experiments Module

Based on "Back to Basics: Let Denoising Generative Models Denoise" (arXiv:2511.13720)

This module provides:
- Data generation for toy experiments (d-dimensional data in D-dimensional space)
- ToyDataset for PyTorch DataLoader
- Visualization utilities
"""

from .toy_dataset import ToyDataset, ToyDatasetConfig
from .generate_data import generate_toy_dataset

__all__ = ["ToyDataset", "ToyDatasetConfig", "generate_toy_dataset"]
