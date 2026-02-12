"""
Toy Dataset Generator

Generates d-dimensional data embedded in D-dimensional space using random orthogonal projection.
Based on "Back to Basics: Let Denoising Generative Models Denoise" (arXiv:2511.13720)

The intrinsic data is sampled uniformly from the black squares of an 8x8 checkerboard.

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
    grid_size: int
    board_range: float
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
    grid_size: int = 8,
    board_range: float = 4.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate 2D intrinsic data by uniformly sampling from the black squares
    of an 8x8 checkerboard.

    Args:
        num_samples: Number of samples to generate
        grid_size: Number of squares per side (default 8 for standard checkerboard)
        board_range: Half-width of the board in coordinate space (board spans [-range, range])
        seed: Random seed

    Returns:
        data: (num_samples, 2) array of intrinsic 2D data
    """
    if seed is not None:
        np.random.seed(seed)

    cell_size = 2 * board_range / grid_size

    # Enumerate black squares: (row + col) % 2 == 0
    black_squares = []
    for r in range(grid_size):
        for c in range(grid_size):
            if (r + c) % 2 == 0:
                black_squares.append((r, c))

    num_black = len(black_squares)
    samples_per_square = num_samples // num_black
    extra = num_samples - samples_per_square * num_black

    data = []
    for i, (r, c) in enumerate(black_squares):
        n = samples_per_square + (1 if i < extra else 0)
        # Bottom-left corner of this cell
        x0 = -board_range + c * cell_size
        y0 = -board_range + r * cell_size
        # Uniform samples within the cell
        samples = np.random.rand(n, 2) * cell_size + np.array([x0, y0])
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
    grid_size: int = 8,
    board_range: float = 4.0,
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
        grid_size: Checkerboard grid size (default 8)
        board_range: Half-width of the checkerboard
        seed: Random seed
        verbose: Print progress

    Returns:
        info: Dictionary with dataset paths and metadata
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate intrinsic data (shared across all embeddings)
    if verbose:
        print(f"Generating {num_samples} intrinsic samples (d={intrinsic_dim}, {grid_size}x{grid_size} checkerboard)...")
    intrinsic_data = generate_intrinsic_data(
        num_samples=num_samples,
        grid_size=grid_size,
        board_range=board_range,
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
        "grid_size": grid_size,
        "board_range": board_range,
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
            grid_size=grid_size,
            board_range=board_range,
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
        "--grid-size",
        type=int,
        default=8,
        help="Checkerboard grid size (default 8 for 8x8)",
    )
    parser.add_argument(
        "--board-range",
        type=float,
        default=4.0,
        help="Half-width of the checkerboard (board spans [-range, range])",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Toy Dataset Generator (Checkerboard)")
    print("=" * 60)
    print(f"Embedding dimensions: {args.embed_dims}")
    print(f"Samples: {args.num_samples}")
    print(f"Grid: {args.grid_size}x{args.grid_size} checkerboard")
    print(f"Output: {args.output_dir}")
    print(f"Seed: {args.seed}")
    print()

    generate_toy_dataset(
        embed_dims=args.embed_dims,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        grid_size=args.grid_size,
        board_range=args.board_range,
        seed=args.seed,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
