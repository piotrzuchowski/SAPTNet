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
    r_values = df["R"].to_numpy(dtype=np.float32)

    pair_vecs = pair_vecs.astype(np.float32)
    return pair_vecs, target_A, target_B, r_values, coords.shape[1]


def build_parallel_n2_scan(r_values: np.ndarray, bond_length: float = 1.1) -> np.ndarray:
    """
    Build coordinates for two parallel N2 molecules separated along z by R.
    """
    half = 0.5 * bond_length
    a1 = np.array([0.0, 0.0, -half])
    a2 = np.array([0.0, 0.0, half])
    coords = []
    for r in r_values:
        b1 = np.array([0.0, 0.0, r - half])
        b2 = np.array([0.0, 0.0, r + half])
        coords.append(np.stack([a1, a2, b1, b2], axis=0))
    return np.stack(coords, axis=0)


def smoothness_metrics(values: np.ndarray, r_values: np.ndarray) -> dict:
    """
    Compute finite-difference smoothness metrics for a vector-valued curve.
    """
    if len(r_values) < 3:
        raise ValueError("Need at least 3 points for second differences.")
    dr = float(r_values[1] - r_values[0])
    second_diff = values[2:] - 2.0 * values[1:-1] + values[:-2]
    second_diff /= dr * dr
    max_abs = np.max(np.abs(second_diff), axis=0)
    rms = np.sqrt(np.mean(second_diff**2, axis=0))
    return {"max_abs": max_abs, "rms": rms}


def run_smoothness_test(
    model: SAPTDualModel,
    device: str,
    out_csv: Path | None = None,
    r_min: float = 3.0,
    r_max: float = 8.0,
    n_points: int = 51,
    bond_length: float = 1.1,
) -> None:
    """
    Evaluate smoothness on a parallel N2 scan after training.
    """
    r_values = np.linspace(r_min, r_max, n_points)
    coords = build_parallel_n2_scan(r_values, bond_length=bond_length)
    pair_vecs, _ = compute_pairwise_vectors(coords)

    model.eval()
    with torch.no_grad():
        vecs_t = torch.tensor(pair_vecs, dtype=torch.float32, device=device)
        mu_A, mu_B, _, _ = model.forward_from_vecs(vecs_t)

    mu_A = mu_A.cpu().numpy()
    mu_B = mu_B.cpu().numpy()

    metrics_A = smoothness_metrics(mu_A, r_values)
    metrics_B = smoothness_metrics(mu_B, r_values)

    print("Smoothness metrics (second derivative):")
    print("A max_abs:", metrics_A["max_abs"])
    print("A rms    :", metrics_A["rms"])
    print("B max_abs:", metrics_B["max_abs"])
    print("B rms    :", metrics_B["rms"])

    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack([r_values, mu_A, mu_B])
        header = "R,muA_x,muA_y,muA_z,muB_x,muB_y,muB_z"
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="")


def write_dms_plot(
    model: SAPTDualModel,
    device: str,
    r_values: np.ndarray,
    true_A_z: np.ndarray,
    true_B_z: np.ndarray,
    out_png: Path,
    bond_length: float = 1.1,
    unscale_power: float = 3.0,
) -> None:
    """
    Plot N2-N2 DMS (z component) for propSAPT vs NN predictions.
    """
    import matplotlib.pyplot as plt

    coords = build_parallel_n2_scan(r_values, bond_length=bond_length)
    pair_vecs, _ = compute_pairwise_vectors(coords)

    model.eval()
    with torch.no_grad():
        vecs_t = torch.tensor(pair_vecs, dtype=torch.float32, device=device)
        mu_A, mu_B, _, _ = model.forward_from_vecs(vecs_t)

    pred_A_z = mu_A[:, 2].cpu().numpy()
    pred_B_z = mu_B[:, 2].cpu().numpy()
    if unscale_power is not None:
        # Convert from scaled targets (mu * R^3) back to physical dipoles.
        scale = np.power(r_values, unscale_power)
        pred_A_z = pred_A_z / scale
        pred_B_z = pred_B_z / scale

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    ax.plot(r_values, true_A_z, color="tab:blue", label="propSAPT A_z")
    ax.plot(r_values, true_B_z, color="tab:red", label="propSAPT B_z")
    ax.plot(r_values, pred_A_z, "--", color="tab:blue", label="NN A_z")
    ax.plot(r_values, pred_B_z, "--", color="tab:red", label="NN B_z")
    ax.set_xlabel("R (Angstrom)")
    ax.set_ylabel("DMS z-component")
    ax.set_title("N2-N2 DMS: reference vs NN")
    ax.legend(frameon=False)
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


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
    save_model: Path | None = None,
    smoothness_out: Path | None = None,
    dms_plot: Path | None = None,
    propsapt_csv: Path | None = None,
    propsapt_r_col: str = "R",
    propsapt_a_z_col: str = "A_z",
    propsapt_b_z_col: str = "B_z",
    bond_length: float = 1.1,
):
    """
    Train SAPTDualModel on pairwise vectors using MSE loss and Adam.
    """
    torch.manual_seed(seed)

    # Prepare inputs (pairwise vectors) and targets (X1_A/X1_B scaled by R^3).
    pair_vecs, target_A, target_B, r_values, n_atoms = build_dataset(
        csv_path, scale_power=scale_power
    )

    x = torch.from_numpy(pair_vecs)
    yA = torch.from_numpy(target_A)
    yB = torch.from_numpy(target_B)

    # Bundle into a dataset: each sample is (pair_vecs, target_A, target_B).
    r = torch.from_numpy(r_values)
    dataset = TensorDataset(x, yA, yB, r)
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
        for xb, yA_b, yB_b, _ in train_loader:
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
            for xb, yA_b, yB_b, _ in val_loader:
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
        r_vals = []
        with torch.no_grad():
            for xb, yA_b, yB_b, r_b in val_loader:
                xb = xb.to(device)
                pred_A, pred_B, _, _ = model.forward_from_vecs(xb)
                preds_A.append(pred_A.cpu().numpy())
                preds_B.append(pred_B.cpu().numpy())
                trues_A.append(yA_b.numpy())
                trues_B.append(yB_b.numpy())
                r_vals.append(r_b.numpy())

        pred_A = np.vstack(preds_A)
        pred_B = np.vstack(preds_B)
        true_A = np.vstack(trues_A)
        true_B = np.vstack(trues_B)
        r_vals = np.concatenate(r_vals)
        scale = np.power(r_vals, scale_power)[:, None]
        pred_A = pred_A / scale
        pred_B = pred_B / scale
        true_A = true_A / scale
        true_B = true_B / scale
        if parity_data is not None:
            parity_data = Path(parity_data)
            parity_data.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                parity_data,
                pred_A=pred_A,
                pred_B=pred_B,
                true_A=true_A,
                true_B=true_B,
                r_vals=r_vals,
                epochs=epochs,
            )
        if parity_plot is not None:
            _write_parity_plot(
                pred_A, true_A, pred_B, true_B, parity_plot, epochs=epochs
            )

    if save_model is not None:
        save_model.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_model)
        print(f"Saved model to {save_model}")

    if smoothness_out is not None:
        run_smoothness_test(model, device=device, out_csv=smoothness_out)

    if dms_plot is not None:
        if propsapt_csv is None:
            raise ValueError("--propsapt-csv is required when using --dms-plot.")
        import pandas as pd

        df = pd.read_csv(propsapt_csv)
        for col in (propsapt_r_col, propsapt_a_z_col, propsapt_b_z_col):
            if col not in df.columns:
                raise ValueError(f"Missing column in reference CSV: {col}")
        r_values = df[propsapt_r_col].to_numpy(dtype=float)
        true_A_z = df[propsapt_a_z_col].to_numpy(dtype=float)
        true_B_z = df[propsapt_b_z_col].to_numpy(dtype=float)

        write_dms_plot(
            model=model,
            device=device,
            r_values=r_values,
            true_A_z=true_A_z,
            true_B_z=true_B_z,
            out_png=dms_plot,
            bond_length=bond_length,
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
    Save per-component parity plots with A (blue) and B (red).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    components = ("x", "y", "z")
    for i, ax in enumerate(axes):
        ax.scatter(
            true_A[:, i],
            pred_A[:, i],
            s=14,
            alpha=0.6,
            color="tab:blue",
            marker="x",
            label="A",
        )
        ax.scatter(
            true_B[:, i],
            pred_B[:, i],
            s=14,
            alpha=0.6,
            color="tab:red",
            marker="o",
            label="B",
        )
        all_vals = np.concatenate([true_A[:, i], pred_A[:, i], true_B[:, i], pred_B[:, i]])
        vmin = float(all_vals.min())
        vmax = float(all_vals.max())
        ax.plot([vmin, vmax], [vmin, vmax], color="black", linewidth=1)
        ax.set_title(f"{components[i]} (epochs={epochs})")
        ax.set_xlabel("Target")
        if i == 0:
            ax.set_ylabel("Prediction")
        if i == 2:
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
    parser.add_argument("--load-model", type=Path, default=None)
    parser.add_argument("--csv", type=Path, default=None)
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
    parser.add_argument("--save-model", type=Path, default=None)
    parser.add_argument("--smoothness-out", type=Path, default=None)
    parser.add_argument("--dms-plot", type=Path, default=None)
    parser.add_argument("--propsapt-csv", type=Path, default=None)
    parser.add_argument("--propsapt-r-col", default="R")
    parser.add_argument("--propsapt-a-z-col", default="A_z")
    parser.add_argument("--propsapt-b-z-col", default="B_z")
    parser.add_argument("--bond-length", type=float, default=1.1)
    args = parser.parse_args()

    if args.load_model is not None and args.dms_plot is not None:
        model = SAPTDualModel(n_atoms=4)
        state = torch.load(args.load_model, map_location="cpu")
        model.load_state_dict(state)
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        if args.propsapt_csv is None:
            raise ValueError("--propsapt-csv is required when using --dms-plot.")
        import pandas as pd

        df = pd.read_csv(args.propsapt_csv)
        for col in (
            args.propsapt_r_col,
            args.propsapt_a_z_col,
            args.propsapt_b_z_col,
        ):
            if col not in df.columns:
                raise ValueError(f"Missing column in reference CSV: {col}")
        r_values = df[args.propsapt_r_col].to_numpy(dtype=float)
        true_A_z = df[args.propsapt_a_z_col].to_numpy(dtype=float)
        true_B_z = df[args.propsapt_b_z_col].to_numpy(dtype=float)

        write_dms_plot(
            model=model,
            device=device,
            r_values=r_values,
            true_A_z=true_A_z,
            true_B_z=true_B_z,
            out_png=args.dms_plot,
            bond_length=args.bond_length,
        )
        return

    if args.csv is None:
        raise ValueError("--csv is required for training.")

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
        save_model=args.save_model,
        smoothness_out=args.smoothness_out,
        dms_plot=args.dms_plot,
        propsapt_csv=args.propsapt_csv,
        propsapt_r_col=args.propsapt_r_col,
        propsapt_a_z_col=args.propsapt_a_z_col,
        propsapt_b_z_col=args.propsapt_b_z_col,
        bond_length=args.bond_length,
    )


if __name__ == "__main__":
    main()
