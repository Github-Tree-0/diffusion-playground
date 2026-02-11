"""
Toy Dataset Generator

Generates d-dimensional data embedded in D-dimensional space using random orthogonal projection.
Based on "Back to Basics: Let Denoising Generative Models Denoise" (arXiv:2511.13720)

The intrinsic data is a mixture of 8 Gaussians arranged in a circle.

Usage:
    python -m toy_experiments.generate_data --embed-dims 2 8 16 64 512
    python -m toy_experiments.generate_data --num-samples 50000 --output-dir toy_data
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, asdict


@dataclass
class DatasetMetadata:
    """Metadata for a generated toy dataset."""
    intrinsic_dim: int
    embed_dim: int
    num_samples: int
    num_components: int
    radius: float
    std: float
    seed: int


def create_projection_matrix(embed_dim: int, intrinsic_dim: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Create a random column-orthogonal projection matrix P in R^(D x d).
    x_D = x_d @ P.T maps d-dimensional data to D-dimensional space.

    Args:
        embed_dim: Target embedding dimension D
        intrinsic_dim: Source intrinsic dimension d
        seed: Random seed for reproducibility

    Returns:
        P: Orthogonal projection matrix (D, d)
    """
    if seed is not None:
        np.random.seed(seed)

    P = np.random.randn(embed_dim, intrinsic_dim)
    Q, _ = np.linalg.qr(P)
    return Q[:, :intrinsic_dim].astype(np.float32)


def generate_intrinsic_data(
    num_samples: int,
    num_components: int = 8,
    radius: float = 3.0,
    std: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate 2D intrinsic data as a mixture of Gaussians arranged in a circle.

    Args:
        num_samples: Number of samples to generate
        num_components: Number of Gaussian components (arranged in circle)
        radius: Radius of the circle
        std: Standard deviation of each Gaussian
        seed: Random seed

    Returns:
        data: (num_samples, 2) array of intrinsic 2D data
    """
    if seed is not None:
        np.random.seed(seed)

    samples_per_component = num_samples // num_components
    extra = num_samples - samples_per_component * num_components

    data = []
    angle_step = 2 * np.pi / num_components

    for i in range(num_components):
        n = samples_per_component + (1 if i < extra else 0)
        angle = i * angle_step
        center = np.array([radius * np.cos(angle), radius * np.sin(angle)])
        samples = np.random.randn(n, 2) * std + center
        data.append(samples)

    data = np.concatenate(data, axis=0).astype(np.float32)
    np.random.shuffle(data)
    return data


def embed_data(intrinsic_data: np.ndarray, projection: np.ndarray) -> np.ndarray:
    """
    Embed intrinsic data into higher dimensional space.

    Args:
        intrinsic_data: (N, d) intrinsic data
        projection: (D, d) projection matrix

    Returns:
        embedded: (N, D) embedded data
    """
    return (intrinsic_data @ projection.T).astype(np.float32)


def generate_toy_dataset(
    embed_dims: List[int],
    num_samples: int = 50000,
    output_dir: str = "toy_data",
    intrinsic_dim: int = 2,
    num_components: int = 8,
    radius: float = 3.0,
    std: float = 0.3,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    Generate toy datasets for multiple embedding dimensions.

    Args:
        embed_dims: List of embedding dimensions to generate
        num_samples: Number of samples per dataset
        output_dir: Output directory
        intrinsic_dim: Intrinsic dimension (default 2)
        num_components: Number of Gaussian components
        radius: Radius of the circle arrangement
        std: Standard deviation of each Gaussian
        seed: Random seed
        verbose: Print progress

    Returns:
        info: Dictionary with dataset paths and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate intrinsic data (shared across all embeddings)
    if verbose:
        print(f"Generating {num_samples} intrinsic samples (d={intrinsic_dim})...")
    intrinsic_data = generate_intrinsic_data(
        num_samples=num_samples,
        num_components=num_components,
        radius=radius,
        std=std,
        seed=seed,
    )

    # Save intrinsic data
    intrinsic_path = output_path / "intrinsic_data.npy"
    np.save(intrinsic_path, intrinsic_data)
    if verbose:
        print(f"  Saved intrinsic data to {intrinsic_path}")

    info = {
        "intrinsic_dim": intrinsic_dim,
        "num_samples": num_samples,
        "num_components": num_components,
        "radius": radius,
        "std": std,
        "seed": seed,
        "datasets": {},
    }

    # Generate embedded datasets for each dimension
    for D in embed_dims:
        if verbose:
            print(f"\nGenerating D={D} embedding...")

        # Use different seed for each dimension's projection (but deterministic)
        proj_seed = seed + D
        projection = create_projection_matrix(D, intrinsic_dim, seed=proj_seed)

        # Embed data
        embedded_data = embed_data(intrinsic_data, projection)

        # Create dimension-specific directory
        dim_dir = output_path / f"D{D}"
        dim_dir.mkdir(exist_ok=True)

        # Save embedded data and projection
        data_path = dim_dir / "data.npy"
        proj_path = dim_dir / "projection.npy"

        np.save(data_path, embedded_data)
        np.save(proj_path, projection)

        # Save metadata
        metadata = DatasetMetadata(
            intrinsic_dim=intrinsic_dim,
            embed_dim=D,
            num_samples=num_samples,
            num_components=num_components,
            radius=radius,
            std=std,
            seed=seed,
        )
        meta_path = dim_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        info["datasets"][D] = {
            "data_path": str(data_path),
            "projection_path": str(proj_path),
            "metadata_path": str(meta_path),
        }

        if verbose:
            print(f"  Data shape: {embedded_data.shape}")
            print(f"  Projection shape: {projection.shape}")
            print(f"  Saved to {dim_dir}")

    # Save overall info
    info_path = output_path / "dataset_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    if verbose:
        print(f"\nDataset info saved to {info_path}")

    return info


def main():
    parser = argparse.ArgumentParser(
        description="Generate toy datasets for diffusion experiments"
    )
    parser.add_argument(
        "--embed-dims",
        type=int,
        nargs="+",
        default=[2, 8, 16, 64, 512],
        help="Embedding dimensions to generate",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="toy_data",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--num-components",
        type=int,
        default=8,
        help="Number of Gaussian components",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=3.0,
        help="Radius of component arrangement",
    )
    parser.add_argument(
        "--std",
        type=float,
        default=0.3,
        help="Standard deviation of each component",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Toy Dataset Generator")
    print("=" * 60)
    print(f"Embedding dimensions: {args.embed_dims}")
    print(f"Samples: {args.num_samples}")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print()

    generate_toy_dataset(
        embed_dims=args.embed_dims,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        num_components=args.num_components,
        radius=args.radius,
        std=args.std,
        seed=args.seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
