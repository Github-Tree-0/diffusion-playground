"""
Visualization utilities for toy experiments.

Provides functions to visualize:
- 2D samples (original and generated)
- Training curves
- Comparison across prediction strategies
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Union
import json

# Try to import matplotlib (optional for headless servers)
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def check_matplotlib():
    """Check if matplotlib is available."""
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install with: pip install matplotlib"
        )


def plot_2d_samples(
    samples: Union[np.ndarray, torch.Tensor],
    projection: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Samples",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 8),
    alpha: float = 0.5,
    s: float = 10,
    reference_samples: Optional[Union[np.ndarray, torch.Tensor]] = None,
) -> None:
    """
    Plot 2D samples. If data is high-dimensional, project to 2D using the projection matrix.

    Args:
        samples: (N, D) array of samples
        projection: (D, d) projection matrix for projecting back to intrinsic space
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        alpha: Point transparency
        s: Point size
        reference_samples: Optional reference samples to plot alongside
    """
    check_matplotlib()

    # Convert to numpy
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    if projection is not None and isinstance(projection, torch.Tensor):
        projection = projection.detach().cpu().numpy()
    if reference_samples is not None and isinstance(reference_samples, torch.Tensor):
        reference_samples = reference_samples.detach().cpu().numpy()

    # Project to 2D if needed
    if samples.shape[1] > 2 and projection is not None:
        # Project back to intrinsic space: x_d = x_D @ P
        samples_2d = samples @ projection
    elif samples.shape[1] == 2:
        samples_2d = samples
    else:
        # Just take first 2 dimensions
        samples_2d = samples[:, :2]

    fig, ax = plt.subplots(figsize=figsize)

    # Plot reference samples if provided
    if reference_samples is not None:
        if reference_samples.shape[1] > 2 and projection is not None:
            ref_2d = reference_samples @ projection
        elif reference_samples.shape[1] == 2:
            ref_2d = reference_samples
        else:
            ref_2d = reference_samples[:, :2]

        ax.scatter(ref_2d[:, 0], ref_2d[:, 1], c="gray", alpha=alpha * 0.5,
                   s=s, label="Reference")

    # Plot generated samples
    ax.scatter(samples_2d[:, 0], samples_2d[:, 1], c="blue", alpha=alpha,
               s=s, label="Generated")

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.close()


def plot_training_curves(
    losses: Dict[str, List[float]],
    title: str = "Training Loss",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    log_scale: bool = True,
) -> None:
    """
    Plot training loss curves for multiple runs.

    Args:
        losses: Dict mapping run name to list of loss values
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        log_scale: Use log scale for y-axis
    """
    check_matplotlib()

    fig, ax = plt.subplots(figsize=figsize)

    colors = {
        "epsilon": "blue",
        "x": "green",
        "v": "orange",
    }

    for name, loss_values in losses.items():
        color = colors.get(name, None)
        ax.plot(loss_values, label=name, color=color, alpha=0.8)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.close()


def plot_comparison(
    results: Dict[int, Dict[str, Dict[str, float]]],
    metric: str = "final_loss",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    log_scale: bool = True,
) -> None:
    """
    Plot comparison across embedding dimensions and prediction strategies.

    Args:
        results: Nested dict: {embed_dim: {pred_type: {metric: value}}}
        metric: Metric to plot ("final_loss", "coverage", etc.)
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        log_scale: Use log scale for y-axis
    """
    check_matplotlib()

    embed_dims = sorted(results.keys())
    pred_types = list(results[embed_dims[0]].keys())

    colors = {
        "epsilon": "blue",
        "x": "green",
        "v": "orange",
    }

    fig, ax = plt.subplots(figsize=figsize)

    for pred_type in pred_types:
        values = [results[D][pred_type][metric] for D in embed_dims]
        color = colors.get(pred_type, None)
        ax.plot(embed_dims, values, marker="o", label=f"{pred_type}-prediction",
                color=color, linewidth=2)

    ax.set_xlabel("Embedding Dimension D")
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs Embedding Dimension")
    ax.legend()
    ax.set_xscale("log")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.close()


def create_experiment_report(
    results: Dict[int, Dict[str, Dict[str, float]]],
    output_dir: str,
    experiment_name: str = "toy_experiment",
) -> str:
    """
    Create a full visualization report for an experiment.

    Args:
        results: Nested dict: {embed_dim: {pred_type: {metric: value}}}
        output_dir: Output directory for figures
        experiment_name: Name for the experiment

    Returns:
        Path to the report directory
    """
    check_matplotlib()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    embed_dims = sorted(results.keys())
    pred_types = list(results[embed_dims[0]].keys())

    # 1. Loss comparison
    plot_comparison(
        results, metric="final_loss",
        title="Training Loss vs Embedding Dimension",
        save_path=str(output_path / f"{experiment_name}_loss.png"),
        log_scale=True,
    )

    # 2. Coverage comparison (if available)
    if "coverage" in results[embed_dims[0]][pred_types[0]]:
        plot_comparison(
            results, metric="coverage",
            title="Sample Coverage vs Embedding Dimension",
            save_path=str(output_path / f"{experiment_name}_coverage.png"),
            log_scale=False,
        )

    # 3. Bar chart for highest dimension
    fig, ax = plt.subplots(figsize=(8, 6))
    max_D = max(embed_dims)
    x_pos = np.arange(len(pred_types))
    losses = [results[max_D][pt]["final_loss"] for pt in pred_types]

    colors = ["blue", "green", "orange"][:len(pred_types)]
    ax.bar(x_pos, losses, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{pt}\nprediction" for pt in pred_types])
    ax.set_ylabel("Final Loss")
    ax.set_title(f"Comparison at D={max_D}")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path / f"{experiment_name}_bar_D{max_D}.png", dpi=150)
    plt.close()

    # 4. Save results as JSON
    json_results = {}
    for D in results:
        json_results[str(D)] = {}
        for pred_type in results[D]:
            json_results[str(D)][pred_type] = {
                k: float(v) for k, v in results[D][pred_type].items()
            }

    with open(output_path / f"{experiment_name}_results.json", "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"Report saved to {output_path}")
    return str(output_path)


def plot_denoising_trajectory(
    trajectory: List[Union[np.ndarray, torch.Tensor]],
    projection: Optional[Union[np.ndarray, torch.Tensor]] = None,
    title: str = "Denoising Trajectory",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 4),
    num_steps_to_show: int = 5,
) -> None:
    """
    Plot the denoising trajectory from noise to data.

    Args:
        trajectory: List of (N, D) arrays at different timesteps
        projection: Projection matrix for visualizing high-D data
        title: Plot title
        save_path: Path to save the figure
        figsize: Figure size
        num_steps_to_show: Number of intermediate steps to show
    """
    check_matplotlib()

    # Convert to numpy
    trajectory = [
        t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else t
        for t in trajectory
    ]
    if projection is not None and isinstance(projection, torch.Tensor):
        projection = projection.detach().cpu().numpy()

    # Select steps to show
    total_steps = len(trajectory)
    step_indices = np.linspace(0, total_steps - 1, num_steps_to_show, dtype=int)

    fig, axes = plt.subplots(1, num_steps_to_show, figsize=figsize)

    for ax_idx, step_idx in enumerate(step_indices):
        samples = trajectory[step_idx]

        # Project to 2D if needed
        if samples.shape[1] > 2 and projection is not None:
            samples_2d = samples @ projection
        elif samples.shape[1] == 2:
            samples_2d = samples
        else:
            samples_2d = samples[:, :2]

        t_value = step_idx / (total_steps - 1)
        ax = axes[ax_idx]
        ax.scatter(samples_2d[:, 0], samples_2d[:, 1], alpha=0.5, s=10)
        ax.set_title(f"t = {t_value:.2f}")
        ax.set_aspect("equal")
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved figure to {save_path}")

    plt.close()
