"""Controlled ablation models for the GDSC2 drug/cell-line benchmark."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class DrugIdExpressionModel(nn.Module):
    """Drug-ID embeddings + cell-line expression encoder; no graph propagation."""

    def __init__(
        self,
        num_drugs: int,
        num_expression_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if min(num_drugs, num_expression_features, hidden_dim) <= 0:
            raise ValueError("All model dimensions must be positive.")
        self.drug_embedding = nn.Embedding(num_drugs, hidden_dim)
        nn.init.xavier_uniform_(self.drug_embedding.weight)
        self.cell_encoder = nn.Sequential(
            nn.Linear(num_expression_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        expression: torch.Tensor,
        drug_indices: torch.Tensor,
        cell_indices: torch.Tensor,
    ) -> torch.Tensor:
        drug = self.drug_embedding(drug_indices)
        cell = self.cell_encoder(expression[cell_indices])
        pair = torch.cat((drug, cell, drug * cell, torch.abs(drug - cell)), dim=1)
        return self.pair_head(pair)


class ExpressionOnlyModel(nn.Module):
    """Cell-line expression predictor; deliberately ignores drug identity."""

    def __init__(
        self,
        num_expression_features: int,
        hidden_dim: int = 64,
        dropout: float = 0.20,
    ) -> None:
        super().__init__()
        if min(num_expression_features, hidden_dim) <= 0:
            raise ValueError("All model dimensions must be positive.")
        self.cell_encoder = nn.Sequential(
            nn.Linear(num_expression_features, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        expression: torch.Tensor,
        drug_indices: torch.Tensor,
        cell_indices: torch.Tensor,
    ) -> torch.Tensor:
        # drug_indices is accepted to keep the training interface identical,
        # but is intentionally unused in this ablation.
        del drug_indices
        return self.head(self.cell_encoder(expression[cell_indices]))
