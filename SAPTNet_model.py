"""
Neural network models for predicting induced dipole vectors.
"""

from typing import Tuple

import torch
import torch.nn as nn


class PairwiseVectorNet(nn.Module):
    """
    Predict a vector as a weighted sum of all pairwise atom vectors.
    """

    def __init__(self, n_atoms: int, hidden_dim: int = 128) -> None:
        super().__init__()
        if n_atoms < 2:
            raise ValueError("n_atoms must be >= 2")

        self.n_atoms = n_atoms
        self.n_pairs = (n_atoms * (n_atoms - 1)) // 2

        tri_idx = torch.triu_indices(n_atoms, n_atoms, offset=1)
        self.register_buffer("idx_i", tri_idx[0])
        self.register_buffer("idx_j", tri_idx[1])

        self.mlp = nn.Sequential(
            nn.Linear(self.n_pairs, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_pairs),
        )

    def forward(self, coords: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: Tensor of shape (batch, n_atoms, 3)
        Returns:
            mu: (batch, 3)
            s: (batch, n_pairs)
        """
        vecs = coords[:, self.idx_j, :] - coords[:, self.idx_i, :]
        return self.forward_from_vecs(vecs)

    def forward_from_vecs(self, vecs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vecs: Tensor of shape (batch, n_pairs, 3)
        Returns:
            mu: (batch, 3)
            s: (batch, n_pairs)
        """
        dists = torch.linalg.norm(vecs, dim=-1)
        s = self.mlp(dists)
        mu = torch.sum(s.unsqueeze(-1) * vecs, dim=1)
        return mu, s


class SAPTDualModel(nn.Module):
    """
    Two independent heads for predicting induced dipoles on monomers A and B.
    """

    def __init__(self, n_atoms: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net_A = PairwiseVectorNet(n_atoms=n_atoms, hidden_dim=hidden_dim)
        self.net_B = PairwiseVectorNet(n_atoms=n_atoms, hidden_dim=hidden_dim)

    def forward(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_A, s_A = self.net_A(coords)
        mu_B, s_B = self.net_B(coords)
        return mu_A, mu_B, s_A, s_B

    def forward_from_vecs(
        self, vecs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu_A, s_A = self.net_A.forward_from_vecs(vecs)
        mu_B, s_B = self.net_B.forward_from_vecs(vecs)
        return mu_A, mu_B, s_A, s_B
