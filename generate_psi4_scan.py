"""
Generate Psi4 geometry files for a parallel N2-N2 separation scan.
"""

import argparse
from pathlib import Path

import numpy as np


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


def psi4_geom_string(
    coords: np.ndarray,
    symbols_a: list[str],
    symbols_b: list[str],
    n_atoms_a: int,
    psi4_units: str = "angstrom",
    charge_mult_a: str = "0 1",
    charge_mult_b: str = "0 1",
) -> str:
    """
    Build a Psi4 geometry string with a monomer separator.
    """
    lines = [
        "symmetry c1",
        "no_com",
        "no_reorient",
        f"units {psi4_units}",
        charge_mult_a,
    ]
    for sym, (x, y, z) in zip(symbols_a, coords[:n_atoms_a]):
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    lines.append("--")
    lines.append(charge_mult_b)
    for sym, (x, y, z) in zip(symbols_b, coords[n_atoms_a:]):
        lines.append(f"{sym} {x:.10f} {y:.10f} {z:.10f}")
    return "\n".join(lines) + "\n"


def main() -> None:
    """
    CLI entry point.
    """
    parser = argparse.ArgumentParser(description="Generate Psi4 scan geometries.")
    parser.add_argument("--r-min", type=float, default=3.0)
    parser.add_argument("--r-max", type=float, default=8.0)
    parser.add_argument("--n-points", type=int, default=51)
    parser.add_argument("--bond-length", type=float, default=1.1)
    parser.add_argument("--out-dir", type=Path, default=Path("psi4_scan_geoms"))
    parser.add_argument("--psi4-units", default="angstrom")
    parser.add_argument("--charge-mult-a", default="0 1")
    parser.add_argument("--charge-mult-b", default="0 1")
    args = parser.parse_args()

    r_values = np.linspace(args.r_min, args.r_max, args.n_points)
    coords = build_parallel_n2_scan(r_values, bond_length=args.bond_length)

    symbols_a = ["N", "N"]
    symbols_b = ["N", "N"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for i, (r, geom) in enumerate(zip(r_values, coords)):
        geom_str = psi4_geom_string(
            geom,
            symbols_a=symbols_a,
            symbols_b=symbols_b,
            n_atoms_a=len(symbols_a),
            psi4_units=args.psi4_units,
            charge_mult_a=args.charge_mult_a,
            charge_mult_b=args.charge_mult_b,
        )
        filename = args.out_dir / f"N2_N2_{i:04d}_R{r:.3f}.psi4geom"
        filename.write_text(geom_str)


if __name__ == "__main__":
    main()
