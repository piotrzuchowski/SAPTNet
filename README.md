# SAPTNet
Neural network for propSAPT/SAPT components.

Neural network architecture for predicting intermolecular interaction energies using Symmetry-Adapted Perturbation Theory (SAPT).

## Overview

SAPTNet predicts individual components of intermolecular interaction energies rather than total interaction energies. This component-based approach provides more detailed understanding of energy contributions and improves prediction accuracy.

### Initial Validation

The project uses CO-CO interactions as the initial test case with:
- **propSAPT** components as dipole prediction
- **SAPT0** corrections as for the energy predictions

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

### Loading Data

```python
from SAPTNet_helper import load_data, prepare_features_and_targets

# Load data from CSV or NPZ
df, metadata = load_data('data.csv')

# Prepare features and targets
X, y, info = prepare_features_and_targets(
    df,
    sapt_columns=['ES', 'EX', 'IND', 'DISP'],  # propSAPT components
    sapt0_columns=['ES_CORR', 'EX_CORR'],      # optional SAPT0 corrections
    include_sapt0=True
)
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
├── SAPTNet_helper.py      # Data loading and preprocessing utilities
├── requirements.txt        # Project dependencies
├── README.md              # This file
└── .github/
    └── copilot-instructions.md  # Custom instructions for Copilot
```

## Development

See `copilot-instructions.md` for workspace-specific guidelines and development notes.
