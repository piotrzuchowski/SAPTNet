"""
Helper functions for SAPTNet
Provides data loading and preprocessing utilities for intermolecular interaction energy predictions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union


def load_data(
    filepath: Union[str, Path],
    sapt_columns: Optional[list] = None,
    correction_columns: Optional[list] = None,
    sapt0_columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Load data from CSV or NPZ files.
    
    Args:
        filepath: Path to CSV or NPZ file
        sapt_columns: List of propSAPT component column names (default: auto-detect)
        correction_columns: List of correction column names (default: auto-detect)
        sapt0_columns: List of SAPT0 component column names (optional)
    
    Returns:
        Tuple of (dataframe, metadata_dict)
    """
    filepath = Path(filepath)
    
    if filepath.suffix == '.csv':
        df = pd.read_csv(filepath)
    elif filepath.suffix == '.npz':
        data = np.load(filepath)
        df = pd.DataFrame({key: data[key] for key in data.files})
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    metadata = {
        'source_file': str(filepath),
        'total_samples': len(df),
        'columns': df.columns.tolist(),
    }
    
    # Auto-detect SAPT columns if not provided
    if sapt_columns is None:
        sapt_columns = [col for col in df.columns if 'sapt' in col.lower() and 'sapt0' not in col.lower()]
    
    # Auto-detect SAPT0 columns if not provided
    if sapt0_columns is None:
        sapt0_columns = [col for col in df.columns if 'sapt0' in col.lower()]
    
    # Auto-detect correction columns if not provided
    if correction_columns is None:
        correction_columns = [col for col in df.columns if 'correction' in col.lower()]
    
    metadata['sapt_columns'] = sapt_columns
    metadata['sapt0_columns'] = sapt0_columns
    metadata['correction_columns'] = correction_columns
    
    return df, metadata


def prepare_features_and_targets(
    df: pd.DataFrame,
    sapt_columns: list,
    target_column: Optional[str] = None,
    sapt0_columns: Optional[list] = None,
    include_sapt0: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Prepare feature matrix and target vectors from dataframe.
    
    Args:
        df: Input dataframe
        sapt_columns: List of propSAPT component column names
        target_column: Name of target column (if None, sum of sapt_columns is used)
        sapt0_columns: List of SAPT0 component columns to include
        include_sapt0: Whether to include SAPT0 components in features
    
    Returns:
        Tuple of (features_array, targets_array, processing_info)
    """
    # Prepare features
    feature_cols = sapt_columns.copy()
    if include_sapt0 and sapt0_columns:
        feature_cols.extend(sapt0_columns)
    
    X = df[feature_cols].values
    
    # Prepare targets
    if target_column is not None:
        y = df[target_column].values
    else:
        # Sum propSAPT components as default target
        y = df[sapt_columns].sum(axis=1).values
    
    processing_info = {
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'feature_columns': feature_cols,
        'target_source': target_column or 'sum_of_sapt_components',
    }
    
    return X, y, processing_info


def validate_data(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that required columns exist in dataframe.
    
    Args:
        df: Input dataframe
        required_columns: List of required column names
    
    Returns:
        True if all required columns present, raises ValueError otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def infer_coordinate_columns(df: pd.DataFrame) -> Tuple[list, list]:
    """
    Infer coordinate columns from MLdms_general output.

    Looks for columns named x_<label>, y_<label>, z_<label> and preserves the
    atom order as they appear in the dataframe (x_ columns).
    """
    # Coordinate labels look like x_NA1, x_OB2, etc.
    x_cols = [
        col
        for col in df.columns
        if col.startswith("x_")
        and len(col) >= 5
        and col[2].isalpha()
        and col[3] in ("A", "B")
        and col[4:].isdigit()
    ]
    if not x_cols:
        raise ValueError("No coordinate columns found (expected x_<label> style).")

    labels = [col[2:] for col in x_cols]
    coord_cols = []
    for label in labels:
        for axis in ("x", "y", "z"):
            col = f"{axis}_{label}"
            if col not in df.columns:
                raise ValueError(f"Missing coordinate column: {col}")
            coord_cols.append(col)
    return labels, coord_cols


def extract_coordinates(
    df: pd.DataFrame, coord_cols: Optional[Sequence[str]] = None
) -> Tuple[np.ndarray, list]:
    """
    Extract coordinates as (n_samples, n_atoms, 3) from MLdms_general outputs.
    """
    if coord_cols is None:
        _, coord_cols = infer_coordinate_columns(df)
    coords = df[list(coord_cols)].to_numpy(dtype=float)
    n_atoms = len(coord_cols) // 3
    coords = coords.reshape(-1, n_atoms, 3)
    return coords, list(coord_cols)


def compute_pairwise_vectors(
    coords: np.ndarray,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute pairwise vectors r_j - r_i for all i < j.

    Args:
        coords: Array of shape (n_samples, n_atoms, 3)

    Returns:
        (pairwise_vectors, (idx_i, idx_j))
        pairwise_vectors has shape (n_samples, n_pairs, 3)
    """
    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError("coords must have shape (n_samples, n_atoms, 3)")
    n_atoms = coords.shape[1]
    idx_i, idx_j = np.triu_indices(n_atoms, k=1)
    vecs = coords[:, idx_j, :] - coords[:, idx_i, :]
    return vecs, (idx_i, idx_j)


def extract_vector_targets(
    df: pd.DataFrame, target_prefixes: Sequence[str]
) -> Dict[str, np.ndarray]:
    """
    Extract vector targets stored as <prefix>_x, <prefix>_y, <prefix>_z.
    """
    targets = {}
    for prefix in target_prefixes:
        cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns for {prefix}: {missing}")
        targets[prefix] = df[cols].to_numpy(dtype=float)
    return targets


def add_summed_vector_target(
    df: pd.DataFrame,
    out_prefix: str,
    prefix_a: str,
    prefix_b: str,
) -> pd.DataFrame:
    """
    Add a new vector target by summing two existing vector prefixes.
    """
    for axis in ("x", "y", "z"):
        col_out = f"{out_prefix}_{axis}"
        col_a = f"{prefix_a}_{axis}"
        col_b = f"{prefix_b}_{axis}"
        if col_a not in df.columns or col_b not in df.columns:
            raise ValueError(f"Missing columns for sum: {col_a}, {col_b}")
        df[col_out] = df[col_a].to_numpy(dtype=float) + df[col_b].to_numpy(dtype=float)
    return df


def prepare_x1_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build X1_A and X1_B targets as (x1_pol,r + x1_exch,r) per axis.
    """
    add_summed_vector_target(df, "X1_A", "x1_pol,r_A", "x1_exch,r_A")
    add_summed_vector_target(df, "X1_B", "x1_pol,r_B", "x1_exch,r_B")
    targets = extract_vector_targets(df, ["X1_A", "X1_B"])
    return targets["X1_A"], targets["X1_B"]


def scale_vector_targets_by_r(
    df: pd.DataFrame, target_prefixes: Sequence[str], r_column: str = "R", power: float = 3.0
) -> Dict[str, np.ndarray]:
    """
    Scale vector targets by R**power (e.g., power=3 for dipole induction decay).
    """
    if r_column not in df.columns:
        raise ValueError(f"Missing R column: {r_column}")
    r = df[r_column].to_numpy(dtype=float)
    if np.any(r <= 0):
        raise ValueError("R must be positive for scaling.")
    scale = r ** power

    scaled = {}
    for prefix in target_prefixes:
        cols = [f"{prefix}_x", f"{prefix}_y", f"{prefix}_z"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing target columns for {prefix}: {missing}")
        vec = df[cols].to_numpy(dtype=float)
        scaled[prefix] = vec * scale[:, None]
    return scaled


def add_scaled_vector_columns(
    df: pd.DataFrame,
    target_prefixes: Sequence[str],
    out_suffix: str = "scaled",
    r_column: str = "R",
    power: float = 3.0,
) -> pd.DataFrame:
    """
    Add scaled vector columns to the dataframe.
    """
    scaled = scale_vector_targets_by_r(
        df, target_prefixes=target_prefixes, r_column=r_column, power=power
    )
    for prefix, vec in scaled.items():
        for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
            df[f"{prefix}_{out_suffix}_{axis}"] = vec[:, idx]
    return df


class PairwiseVectorDataset:
    """
    Minimal dataset: returns pairwise vectors and selected targets.
    """

    def __init__(
        self,
        filepath: Union[str, Path],
        target_prefixes: Sequence[str],
        coord_cols: Optional[Sequence[str]] = None,
    ) -> None:
        self.df, self.metadata = load_data(filepath)
        self.coords, self.coord_cols = extract_coordinates(self.df, coord_cols=coord_cols)
        self.pairwise_vecs, self.pair_indices = compute_pairwise_vectors(self.coords)
        self.targets = extract_vector_targets(self.df, target_prefixes)

    def __len__(self) -> int:
        return self.pairwise_vecs.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        return self.pairwise_vecs[idx], {k: v[idx] for k, v in self.targets.items()}

def prepare_pairwise_vector_dataset(
    filepath: Union[str, Path],
    target_prefixes: Sequence[str],
    coord_cols: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], dict]:
    """
    Load MLdms_general (batch) output and return pairwise vectors + targets.
    """
    df, metadata = load_data(filepath)
    coords, coord_cols = extract_coordinates(df, coord_cols=coord_cols)
    pairwise_vecs, pairwise_indices = compute_pairwise_vectors(coords)
    targets = extract_vector_targets(df, target_prefixes)

    info = {
        "n_samples": coords.shape[0],
        "n_atoms": coords.shape[1],
        "n_pairs": pairwise_vecs.shape[1],
        "coord_columns": coord_cols,
        "pair_indices": pairwise_indices,
        "targets": list(targets.keys()),
        **metadata,
    }
    return pairwise_vecs, targets, info
