"""
Evaluate smoothness of model predictions along a 1D N2-N2 separation scan.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from SAPTNet_helper import compute_pairwise_vectors
from SAPTNet_model import SAPTDualModel


def build_parallel_n2_scan(
    r_values: np.ndarray, bond_length: float = 1.1
) -> np.ndarray:
    """
    Build coordinates for two parallel N2 molecules separated along z by R.
    """
    half = 0.5 * bond_length
    # Molecule A centered at origin, aligned along z.
    a1 = np.array([0.0, 0.0, -half])
    a2 = np.array([0.0, 0.0, half])
    coords = []
    for r in r_values:
        # Molecule B centered at (0, 0, R), aligned along z.
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


def main() -> None:
    """
    CLI entry point for smoothness testing.
    """
    parser = argparse.ArgumentParser(description="Smoothness test for N2-N2 scan.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--r-min", type=float, default=3.0)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--n-points", type=int, default=51)
    parser.add_argument("--bond-length", type=float, default=1.1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    r_values = np.linspace(args.r_min, args.r_max, args.n_points)
    coords = build_parallel_n2_scan(r_values, bond_length=args.bond_length)
    pair_vecs, _ = compute_pairwise_vectors(coords)

    model = SAPTDualModel(n_atoms=4)
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state)

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(args.device)
    model.eval()

    with torch.no_grad():
        vecs_t = torch.tensor(pair_vecs, dtype=torch.float32, device=args.device)
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

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        data = np.column_stack([r_values, mu_A, mu_B])
        header = "R,muA_x,muA_y,muA_z,muB_x,muB_y,muB_z"
        np.savetxt(args.out_csv, data, delimiter=",", header=header, comments="")


if __name__ == "__main__":
    main()
