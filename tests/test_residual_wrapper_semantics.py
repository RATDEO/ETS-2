from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.tsm import ResidualWrapper


class _ZeroResidualModel(torch.nn.Module):
    def __init__(self, pred_len: int):
        super().__init__()
        self.pred_len = pred_len

    def forward(self, x_enc, x_dec):
        return torch.zeros((x_enc.shape[0], self.pred_len, 1), dtype=x_enc.dtype, device=x_enc.device)


def test_residual_wrapper_last_value_strategy_repeats_last_observation():
    wrapper = ResidualWrapper(
        _ZeroResidualModel(pred_len=3),
        pred_len=3,
        alpha_init=0.0,
        baseline_strategy="last_value",
    )
    x_enc = torch.tensor([[[1.0], [2.5], [4.0]]], dtype=torch.float32)

    out = wrapper(x_enc, None)

    assert torch.allclose(out.squeeze(0), torch.tensor([[4.0], [4.0], [4.0]]))


def test_residual_wrapper_zero_strategy_uses_flat_zero_return_baseline():
    wrapper = ResidualWrapper(
        _ZeroResidualModel(pred_len=4),
        pred_len=4,
        alpha_init=0.0,
        baseline_strategy="zero",
    )
    x_enc = torch.tensor([[[0.3], [-0.1], [0.2]]], dtype=torch.float32)

    out = wrapper(x_enc, None)

    assert torch.allclose(out, torch.zeros((1, 4, 1), dtype=torch.float32))
