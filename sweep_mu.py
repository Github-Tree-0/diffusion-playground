"""
Sweep: mu vs prediction type

Tests logit_normal(mu, sigma=0.8) across a range of mu values for each of the
three prediction types (epsilon, x, v), running N_EPOCHS epochs per run.

Metric: validation loss computed with UNIFORM t sampling in v-space.
  - Uniform t: removes the influence of mu from the metric itself
  - v-space:   implicit weighting = 1 for all pred types, so they're comparable
               (epsilon has t², x has (1-t)², v has 1 — using v normalises them)

Usage:
    python sweep_mu.py
    python sweep_mu.py --epochs 5 --data-dir toy_data_spiral --embed-dim 16
    python sweep_mu.py --mu-values -2 -1 0 1 2

Results saved to: sweep_mu_results.json  +  sweep_mu_results.png
"""

import argparse
import json
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from models import ToyDiffusion
from toy_experiments import ToyDataset, ToyDatasetConfig


# ── default sweep config ──────────────────────────────────────────────────────

PRED_TYPES = ["epsilon", "x", "v"]
COLORS     = {"epsilon": "#2196F3", "x": "#4CAF50", "v": "#FF9800"}

DEFAULT_MU_VALUES = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
SIGMA = 0.8

BASE_CFG = dict(
    data_dir    = "toy_data_spiral",
    embed_dim   = 16,
    seed        = 42,
    batch_size  = 256,
    lr          = 1e-3,
    epochs      = 5,
    base_channels     = 64,
    time_emb_dim      = 128,
    num_res_blocks    = 2,
    channel_multiples = (1, 2, 4),
    max_grad_norm     = 1.0,
)


# ── validation loss (uniform t, v-space) ─────────────────────────────────────

@torch.no_grad()
def compute_val_loss(model: ToyDiffusion, val_loader: DataLoader,
                     device: torch.device) -> float:
    """
    Unbiased validation loss:
      - uniform t in (0, 1)  → not affected by the mu being tested
      - v-space MSE          → implicit weighting = 1, comparable across pred types
                               (epsilon uses t², x uses (1-t)², v uses 1)
    """
    model.eval()
    losses = []
    for batch in val_loader:
        x_0 = batch["data"].to(device)
        x_0_3d = model._to_3d(x_0)                          # (B, 1, D)
        B = x_0.shape[0]

        t = torch.rand(B, device=device).clamp(1e-5, 1 - 1e-5)   # uniform
        z_t, noise = model.scheduler.add_noise(x_0_3d, t)

        model_output = model.unet(z_t, t)
        preds = model.scheduler.get_all_predictions(z_t, model_output, t)

        v_pred = preds["v"]
        v_true = model.scheduler.compute_velocity(x_0_3d, noise)  # x_0 - ε

        losses.append(nn.functional.mse_loss(v_pred, v_true).item())

    return float(np.mean(losses))


# ── training helper ───────────────────────────────────────────────────────────

def run_one(pred_type: str, mu: float, cfg: dict,
            device: torch.device) -> tuple[list[float], list[float]]:
    """
    Train one (pred_type, mu) combination for cfg['epochs'] epochs.
    Returns (train_epoch_losses, val_epoch_losses).
    val loss is always uniform-t v-space MSE for fair comparison.
    """
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Datasets
    ds_cfg = ToyDatasetConfig(
        data_dir  = cfg["data_dir"],
        embed_dim = cfg["embed_dim"],
        seed      = cfg["seed"],
    )
    train_ds = ToyDataset(ds_cfg, split="train")
    val_ds   = ToyDataset(ds_cfg, split="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda", drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"] * 2, shuffle=False,
        num_workers=0, pin_memory=device.type == "cuda",
    )

    # Model
    model = ToyDiffusion(
        input_dim           = cfg["embed_dim"],
        prediction_type     = pred_type,
        loss_type           = pred_type,
        time_sampler        = "logit_normal",
        time_sampler_params = {"mu": mu, "sigma": SIGMA},
        time_embedding_type = "continuous",
        base_channels       = cfg["base_channels"],
        time_emb_dim        = cfg["time_emb_dim"],
        num_res_blocks      = cfg["num_res_blocks"],
        channel_multiples   = tuple(cfg["channel_multiples"]),
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg["lr"])

    train_losses, val_losses = [], []
    for _ in range(cfg["epochs"]):
        # train
        model.train()
        batch_losses = []
        for batch in train_loader:
            data = batch["data"].to(device)
            loss = model.loss(data)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            batch_losses.append(loss.item())
        train_losses.append(float(np.mean(batch_losses)))

        # validate (uniform t, v-space)
        val_losses.append(compute_val_loss(model, val_loader, device))

    return train_losses, val_losses


# ── sweep ─────────────────────────────────────────────────────────────────────

def sweep(mu_values: list[float], cfg: dict, device: torch.device) -> dict:
    """
    Run the full sweep. Returns nested dict:
        results[pred_type][mu] = {
            "train_losses": [...],   # per-epoch training loss
            "val_losses":   [...],   # per-epoch val loss (uniform t, v-space)
            "final_val_loss": float,
        }
    """
    total = len(PRED_TYPES) * len(mu_values)
    done  = 0

    results = {pt: {} for pt in PRED_TYPES}

    for pred_type in PRED_TYPES:
        for mu in mu_values:
            done += 1
            print(f"[{done:>3}/{total}] pred={pred_type:<7}  mu={mu:+.1f} ...", end="  ", flush=True)
            t0 = time.time()

            train_losses, val_losses = run_one(pred_type, mu, cfg, device)

            results[pred_type][mu] = {
                "train_losses":   train_losses,
                "val_losses":     val_losses,
                "final_val_loss": val_losses[-1],
            }

            dt = time.time() - t0
            print(f"val_loss={val_losses[-1]:.5f}  ({dt:.1f}s)")

    return results


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_results(results: dict, mu_values: list[float], save_path: str, epochs: int):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: final VAL loss vs mu ────────────────────────────────────────────
    ax = axes[0]
    for pred_type in PRED_TYPES:
        val_losses = [results[pred_type][mu]["final_val_loss"] for mu in mu_values]
        ax.plot(mu_values, val_losses, marker="o", label=pred_type,
                color=COLORS[pred_type], linewidth=2)

    ax.set_xlabel("mu (logit_normal)", fontsize=12)
    ax.set_ylabel(f"Val loss after {epochs} epochs\n(uniform t, v-space MSE)", fontsize=11)
    ax.set_title("Val loss vs mu — by prediction type", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # mark optimal mu per pred_type
    for pred_type in PRED_TYPES:
        val_losses = [results[pred_type][mu]["final_val_loss"] for mu in mu_values]
        best_mu    = mu_values[int(np.argmin(val_losses))]
        ax.axvline(best_mu, color=COLORS[pred_type], linestyle="--", alpha=0.4)

    # ── right: val curves at each type's best mu ──────────────────────────────
    ax2 = axes[1]
    for pred_type in PRED_TYPES:
        val_losses = [results[pred_type][mu]["final_val_loss"] for mu in mu_values]
        best_mu    = mu_values[int(np.argmin(val_losses))]
        curve      = results[pred_type][best_mu]["val_losses"]
        ax2.plot(range(1, len(curve) + 1), curve,
                 marker="o", label=f"{pred_type} (mu={best_mu:+.1f})",
                 color=COLORS[pred_type], linewidth=2)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Val loss (uniform t, v-space MSE)", fontsize=11)
    ax2.set_title("Val curve at each type's best mu", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved → {save_path}")
    plt.close()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sweep mu for each prediction type")
    parser.add_argument("--epochs",    type=int,   default=BASE_CFG["epochs"],
                        help="Epochs per run (default 5)")
    parser.add_argument("--data-dir",  type=str,   default=BASE_CFG["data_dir"])
    parser.add_argument("--embed-dim", type=int,   default=BASE_CFG["embed_dim"])
    parser.add_argument("--lr",        type=float, default=BASE_CFG["lr"])
    parser.add_argument("--mu-values", type=float, nargs="+",
                        default=DEFAULT_MU_VALUES,
                        help="List of mu values to sweep")
    parser.add_argument("--output",    type=str,   default="sweep_mu_results",
                        help="Output file prefix (no extension)")
    parser.add_argument("--device",    type=str,   default=None)
    args = parser.parse_args()

    cfg = copy.deepcopy(BASE_CFG)
    cfg["epochs"]    = args.epochs
    cfg["data_dir"]  = args.data_dir
    cfg["embed_dim"] = args.embed_dim
    cfg["lr"]        = args.lr

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"  Sweep: mu in {args.mu_values}")
    print(f"  Pred types: {PRED_TYPES}")
    print(f"  Epochs/run: {cfg['epochs']}")
    print(f"  Data: {cfg['data_dir']}  D={cfg['embed_dim']}")
    print(f"  Device: {device}")
    print(f"  Total runs: {len(PRED_TYPES) * len(args.mu_values)}")
    print("=" * 60)

    results = sweep(args.mu_values, cfg, device)

    # ── save JSON ─────────────────────────────────────────────────────────────
    json_path = args.output + ".json"
    json_results = {
        pt: {str(mu): {
            "train_losses":   v["train_losses"],
            "val_losses":     v["val_losses"],
            "final_val_loss": v["final_val_loss"],
        } for mu, v in mu_dict.items()}
        for pt, mu_dict in results.items()
    }
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved → {json_path}")

    # ── print summary ─────────────────────────────────────────────────────────
    print("\n── Best mu per prediction type (by val loss) ────────────")
    for pred_type in PRED_TYPES:
        val_losses = [results[pred_type][mu]["final_val_loss"] for mu in args.mu_values]
        best_i     = int(np.argmin(val_losses))
        best_mu    = args.mu_values[best_i]
        print(f"  {pred_type:<8}  best mu = {best_mu:+.1f}  "
              f"(val_loss = {val_losses[best_i]:.5f})")

    # ── plot ──────────────────────────────────────────────────────────────────
    plot_results(results, args.mu_values, args.output + ".png", cfg["epochs"])


if __name__ == "__main__":
    main()
