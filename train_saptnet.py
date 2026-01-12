"""
Minimal training script for SAPTNet dual-head vector model.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

from SAPTNet_helper import (
    add_scaled_vector_columns,
    compute_pairwise_vectors,
    extract_coordinates,
    prepare_x1_targets,
)
from SAPTNet_model import SAPTDualModel


def build_dataset(csv_path: Path, scale_power: float = 3.0):
    """
    Load a CSV, build summed/normalized targets, and return arrays for training.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    prepare_x1_targets(df)
    add_scaled_vector_columns(df, ["X1_A", "X1_B"], out_suffix="R3", power=scale_power)

    coords, _ = extract_coordinates(df)
    pair_vecs, _ = compute_pairwise_vectors(coords)

    target_A = df[["X1_A_R3_x", "X1_A_R3_y", "X1_A_R3_z"]].to_numpy(dtype=np.float32)
    target_B = df[["X1_B_R3_x", "X1_B_R3_y", "X1_B_R3_z"]].to_numpy(dtype=np.float32)

    pair_vecs = pair_vecs.astype(np.float32)
    return pair_vecs, target_A, target_B, coords.shape[1]


def train_model(
    csv_path: Path,
    epochs: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    scale_power: float = 3.0,
    val_frac: float = 0.1,
    seed: int = 7,
    split_mode: str = "random",
    device: str | None = None,
    parity_plot: Path | None = None,
    parity_data: Path | None = None,
):
    """
    Train SAPTDualModel on pairwise vectors using MSE loss and Adam.
    """
    torch.manual_seed(seed)

    # Prepare inputs (pairwise vectors) and targets (X1_A/X1_B scaled by R^3).
    pair_vecs, target_A, target_B, n_atoms = build_dataset(
        csv_path, scale_power=scale_power
    )

    x = torch.from_numpy(pair_vecs)
    yA = torch.from_numpy(target_A)
    yB = torch.from_numpy(target_B)

    # Bundle into a dataset: each sample is (pair_vecs, target_A, target_B).
    dataset = TensorDataset(x, yA, yB)
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    if split_mode == "geom_id":
        # Deterministic split: first block for train, last block for validation.
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n_train + n_val))
        train_ds = torch.utils.data.Subset(dataset, train_indices)
        val_ds = torch.utils.data.Subset(dataset, val_indices)
    else:
        # Random split with a fixed seed for repeatability.
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model: two independent heads for A/B dipoles.
    model = SAPTDualModel(n_atoms=n_atoms)
    if device is None:
        # Auto-select device if not specified by the user.
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Optimizer and loss for vector regression.
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        if epoch == 200:
            # Drop LR at epoch 200 for finer convergence.
            for group in optimizer.param_groups:
                group["lr"] = 1e-4
        if epoch == 300:
            # Additional drop by 10x after epoch 300.
            for group in optimizer.param_groups:
                group["lr"] *= 0.1
        # Training loop.
        model.train()
        train_loss = 0.0
        for xb, yA_b, yB_b in train_loader:
            xb = xb.to(device)
            yA_b = yA_b.to(device)
            yB_b = yB_b.to(device)

            optimizer.zero_grad()
            pred_A, pred_B, _, _ = model.forward_from_vecs(xb)
            # Sum of A/B losses keeps both heads balanced.
            loss = loss_fn(pred_A, yA_b) + loss_fn(pred_B, yB_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= n_train

        # Validation loop (no gradients).
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yA_b, yB_b in val_loader:
                xb = xb.to(device)
                yA_b = yA_b.to(device)
                yB_b = yB_b.to(device)
                pred_A, pred_B, _, _ = model.forward_from_vecs(xb)
                loss = loss_fn(pred_A, yA_b) + loss_fn(pred_B, yB_b)
                val_loss += loss.item() * xb.size(0)
        val_loss /= n_val

        print(f"epoch {epoch:03d} | train {train_loss:.6f} | val {val_loss:.6f}")

    if parity_plot is not None or parity_data is not None:
        # Collect validation predictions for a parity plot.
        model.eval()
        preds_A = []
        preds_B = []
        trues_A = []
        trues_B = []
        with torch.no_grad():
            for xb, yA_b, yB_b in val_loader:
                xb = xb.to(device)
                pred_A, pred_B, _, _ = model.forward_from_vecs(xb)
                preds_A.append(pred_A.cpu().numpy())
                preds_B.append(pred_B.cpu().numpy())
                trues_A.append(yA_b.numpy())
                trues_B.append(yB_b.numpy())

        pred_A = np.vstack(preds_A)
        pred_B = np.vstack(preds_B)
        true_A = np.vstack(trues_A)
        true_B = np.vstack(trues_B)
        if parity_data is not None:
            parity_data = Path(parity_data)
            parity_data.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                parity_data,
                pred_A=pred_A,
                pred_B=pred_B,
                true_A=true_A,
                true_B=true_B,
                epochs=epochs,
            )
        if parity_plot is not None:
            _write_parity_plot(
                pred_A, true_A, pred_B, true_B, parity_plot, epochs=epochs
            )

    return model


def _write_parity_plot(
    pred_A: np.ndarray,
    true_A: np.ndarray,
    pred_B: np.ndarray,
    true_B: np.ndarray,
    out_path: Path,
    epochs: int,
) -> None:
    """
    Save a single parity plot with A (blue) and B (red).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.scatter(true_A.ravel(), pred_A.ravel(), s=10, alpha=0.6, color="tab:blue", label="A")
    ax.scatter(true_B.ravel(), pred_B.ravel(), s=10, alpha=0.6, color="tab:red", label="B")

    all_vals = np.concatenate(
        [true_A.ravel(), pred_A.ravel(), true_B.ravel(), pred_B.ravel()]
    )
    vmin = float(all_vals.min())
    vmax = float(all_vals.max())
    ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1)

    ax.set_title(f"Parity Plot (epochs={epochs})")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.legend(frameon=False)

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(description="Train SAPTNet dual-head model.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale-power", type=float, default=3.0)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--split-mode", choices=["random", "geom_id"], default="random")
    parser.add_argument("--device", default=None)
    parser.add_argument("--parity-plot", type=Path, default=None)
    parser.add_argument("--parity-data", type=Path, default=None)
    args = parser.parse_args()

    train_model(
        csv_path=args.csv,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        scale_power=args.scale_power,
        val_frac=args.val_frac,
        seed=args.seed,
        split_mode=args.split_mode,
        device=args.device,
        parity_plot=args.parity_plot,
        parity_data=args.parity_data,
    )


if __name__ == "__main__":
    main()
