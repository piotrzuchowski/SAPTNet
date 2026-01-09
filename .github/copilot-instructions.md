<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

## Workspace Setup Checklist

- [x] Created copilot-instructions.md file in .github directory
- [x] Clarify Project Requirements
- [x] Scaffold the Project
- [x] Customize the Project
- [x] Install Required Extensions
- [ ] Compile the Project
- [ ] Create and Run Task
- [ ] Launch the Project
- [ ] Ensure Documentation is Complete

## Project Information

**Project Name:** SAPTNet  
**Location:** /Users/piotrzuchowski/Work/SAPTNet  
**Status:** Python project for neural network predictions of intermolecular interaction energies

### Project Details

- **Language:** Python
- **Key Libraries:** PyTorch, pandas, numpy
- **Purpose:** Predict intermolecular interaction energy components using neural networks with SAPT components as targets
- **Data Source:** propSAPT with optional SAPT0 corrections
- **Initial Test Case:** CO-CO interactions

### Key Components

1. **SAPTNet_helper.py** - Data loading and preprocessing utilities
   - `load_data()`: Load CSV/NPZ files with auto-detection of SAPT columns
   - `prepare_features_and_targets()`: Prepare training data from dataframes
   - `validate_data()`: Validate required columns

### Dependencies

- torch >= 2.0.0
- pandas >= 2.0.0
- numpy >= 1.24.0

### Workspace-Specific Instructions

To install dependencies:
```bash
pip install -r requirements.txt
```

Working directory reference: `/Users/piotrzuchowski/Work/propSAPT`
