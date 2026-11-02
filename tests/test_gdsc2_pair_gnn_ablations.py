import numpy as np
import torch

from scripts.ai.run_gdsc2_pair_gnn_ablations import fit_target_scaler
from src.training.gdsc2_pair_ablations import (
    DrugIdExpressionModel,
    ExpressionOnlyModel,
)


def test_target_scaler_is_endpoint_specific_and_invertible():
    targets = np.array(
        [[1.0, 0.1], [3.0, 0.3], [5.0, 0.5]],
        dtype=np.float32,
    )
    mean, scale = fit_target_scaler(targets)
    assert np.allclose(mean, [3.0, 0.3])
    assert np.allclose(scale, np.std(targets, axis=0))
    normalized = (targets - mean) / scale
    restored = normalized * scale + mean
    assert np.allclose(restored, targets)


def test_drug_id_expression_model_forward():
    model = DrugIdExpressionModel(
        num_drugs=4,
        num_expression_features=12,
        hidden_dim=16,
    )
    predictions = model(
        torch.randn(5, 12),
        torch.tensor([0, 1, 2]),
        torch.tensor([0, 2, 4]),
    )
    assert predictions.shape == (3, 2)
    assert torch.isfinite(predictions).all()


def test_expression_only_model_ignores_drug_identity():
    model = ExpressionOnlyModel(
        num_expression_features=12,
        hidden_dim=16,
        dropout=0.0,
    )
    model.eval()
    expression = torch.randn(5, 12)
    cells = torch.tensor([0, 2, 4])
    first = model(expression, torch.tensor([0, 1, 2]), cells)
    second = model(expression, torch.tensor([3, 2, 1]), cells)
    assert torch.allclose(first, second)
    assert first.shape == (3, 2)
