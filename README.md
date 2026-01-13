# SAPTNet
Neural network utilities for propSAPT/SAPT dipole components and model training.

## Overview

SAPTNet predicts individual components of intermolecular interaction energies rather than total interaction energies. This component-based approach provides more detailed understanding of energy contributions and improves prediction accuracy.

### Current Focus

- Dipole moment components for dimer systems (e.g., N2–N2).
- Pairwise-vector models that predict monomer A/B induced dipoles.

## Installation

```bash
pip install -r requirements.txt
```

## Requirements

- Python 3.8+
- torch >= 2.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0
- psi4
- ase

## Usage

### Loading Data and Targets

```python
from SAPTNet_helper import (
    prepare_pairwise_vector_dataset,
    prepare_x1_targets,
    add_scaled_vector_columns,
)

# Load pairwise vectors and targets (X1_A/X1_B are sums of x1_pol,r + x1_exch,r)
pair_vecs, targets, info = prepare_pairwise_vector_dataset(
    "data/clean_propsapt.csv",
    target_prefixes=["X1_A", "X1_B"],
)

# Or create scaled targets for training stability (mu * R^3)
df = None  # load with pandas if you want to persist columns
prepare_x1_targets(df)
add_scaled_vector_columns(df, ["X1_A", "X1_B"], out_suffix="R3")
```

### Training

```bash
python train_saptnet.py --csv data/clean_propsapt.csv --split-mode geom_id --epochs 200
```

Parity plot (unscaled dipoles):

```bash
python train_saptnet.py --csv data/clean_propsapt.csv --parity-plot outputs/parity.png
```

### Geometry Scans

Generate Psi4 geometry files for an N2–N2 separation scan:

```bash
python generate_psi4_scan.py --r-min 3.0 --r-max 8.0 --n-points 51 --out-dir psi4_scan_geoms
```

Compute X1 dipoles for a directory of `.psi4geom` files:

```bash
python compute_dms_from_geoms.py --directory psi4_scan_geoms
```

### Data Format

**CSV Files:**
- Rows: individual molecular pairs
- Columns: propSAPT components (e.g., ES, EX, IND, DISP) + optional SAPT0 corrections

**NPZ Files:**
- Arrays in numpy archive with component names as keys

## Project Structure

```
SAPTNet/
├── SAPTNet_helper.py       # Data loading and preprocessing utilities
├── SAPTNet_model.py        # Pairwise-vector model definitions
├── train_saptnet.py         # Training + parity plot utilities
├── generate_psi4_scan.py    # Psi4 geometry scan generator
├── compute_dms_from_geoms.py # Compute X1 dipoles from psi4geom files
├── requirements.txt         # Project dependencies
├── README.md               # This file
└── .github/
    └── copilot-instructions.md  # Custom instructions for Copilot
```

## Development

See `copilot-instructions.md` for workspace-specific guidelines and development notes.
