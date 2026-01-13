"""
Compute X1_A and X1_B dipole components from a directory of Psi4 geometry files.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psi4
from prop_sapt import Dimer, calc_property


def setup_psi4_defaults():
    psi4.set_memory("2 GB")
    psi4.set_num_threads(2)
    psi4.set_options(
        {
            "basis": "aug-cc-pvdz",
            "scf_type": "direct",
            "save_jk": True,
            "DF_BASIS_SCF": "aug-cc-pvdz-jkfit",
            "DF_BASIS_SAPT": "aug-cc-pvdz-ri",
        }
    )


def compute_x1_components(geom_str: str) -> dict:
    dimer = Dimer(geom_str)
    dipole_df = calc_property(dimer, "dipole", results=os.devnull)

    def axis_vec(col: str) -> np.ndarray:
        return np.array(
            [
                float(dipole_df.loc["X", col]),
                float(dipole_df.loc["Y", col]),
                float(dipole_df.loc["Z", col]),
            ],
            dtype=float,
        )

    x1_a = axis_vec("x1_pol,r_A") + axis_vec("x1_exch,r_A")
    x1_b = axis_vec("x1_pol,r_B") + axis_vec("x1_exch,r_B")
    return {
        "X1_A_x": x1_a[0],
        "X1_A_y": x1_a[1],
        "X1_A_z": x1_a[2],
        "X1_B_x": x1_b[0],
        "X1_B_y": x1_b[1],
        "X1_B_z": x1_b[2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute X1 dipoles for psi4geom files.")
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    setup_psi4_defaults()

    geom_dir = args.directory
    files = sorted(geom_dir.glob("*.psi4geom"))
    if not files:
        raise ValueError(f"No .psi4geom files found in {geom_dir}")

    rows = []
    for path in files:
        geom_str = path.read_text()
        row = {
            "geom_file": path.name,
            **compute_x1_components(geom_str),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if args.out_csv is None:
        args.out_csv = geom_dir / "x1_dipoles.csv"
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
