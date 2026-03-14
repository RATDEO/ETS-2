from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.tsm import DLinear


def _set_identity_channel_forecast(model: DLinear) -> None:
    with torch.no_grad():
        for layer in model.Linear_Trend:
            layer.weight.fill_(1.0)
            layer.bias.zero_()
        for layer in model.Linear_Seasonal:
            layer.weight.zero_()
            layer.bias.zero_()


def test_dlinear_target_only_mixer_ignores_exogenous_channels():
    model = DLinear(
        seq_len=1,
        pred_len=1,
        enc_in=2,
        individual=True,
        kernel_size=1,
        channel_mixer="target_only",
    )
    _set_identity_channel_forecast(model)
    x = torch.tensor([[[1.0, 5.0]]], dtype=torch.float32)

    out = model(x)

    assert out.shape == (1, 1, 1)
    assert torch.allclose(out.squeeze(), torch.tensor(1.0))


def test_dlinear_linear_mixer_can_use_exogenous_channels():
    model = DLinear(
        seq_len=1,
        pred_len=1,
        enc_in=2,
        individual=True,
        kernel_size=1,
        channel_mixer="linear",
    )
    _set_identity_channel_forecast(model)
    with torch.no_grad():
        model.channel_projection.weight[:] = torch.tensor([[0.0, 1.0]])
        model.channel_projection.bias.zero_()
    x = torch.tensor([[[1.0, 5.0]]], dtype=torch.float32)

    out = model(x)

    assert out.shape == (1, 1, 1)
    assert torch.allclose(out.squeeze(), torch.tensor(5.0))


def test_dlinear_residual_linear_starts_from_target_only_forecast():
    model = DLinear(
        seq_len=1,
        pred_len=1,
        enc_in=2,
        individual=True,
        kernel_size=1,
        channel_mixer="residual_linear",
    )
    _set_identity_channel_forecast(model)
    x = torch.tensor([[[1.0, 5.0]]], dtype=torch.float32)

    out = model(x)

    assert out.shape == (1, 1, 1)
    assert torch.allclose(out.squeeze(), torch.tensor(1.0))


def test_dlinear_residual_linear_can_add_exogenous_correction():
    model = DLinear(
        seq_len=1,
        pred_len=1,
        enc_in=2,
        individual=True,
        kernel_size=1,
        channel_mixer="residual_linear",
    )
    _set_identity_channel_forecast(model)
    with torch.no_grad():
        model.channel_projection.weight[:] = torch.tensor([[0.0, 1.0]])
        model.channel_projection.bias.zero_()
        model.channel_gate.copy_(torch.tensor(10.0))
    x = torch.tensor([[[1.0, 5.0]]], dtype=torch.float32)

    out = model(x)

    assert out.shape == (1, 1, 1)
    assert torch.allclose(out.squeeze(), torch.tensor(6.0), atol=1e-3)


def test_dlinear_residual_horizon_linear_starts_from_target_only_forecast():
    model = DLinear(
        seq_len=1,
        pred_len=2,
        enc_in=2,
        individual=True,
        kernel_size=1,
        channel_mixer="residual_horizon_linear",
    )
    _set_identity_channel_forecast(model)
    x = torch.tensor([[[1.0, 5.0]]], dtype=torch.float32)

    out = model(x)

    assert out.shape == (1, 2, 1)
    assert torch.allclose(out.squeeze(-1), torch.tensor([[1.0, 1.0]]))


def test_dlinear_residual_horizon_linear_can_use_horizon_specific_weights():
    model = DLinear(
        seq_len=1,
        pred_len=2,
        enc_in=2,
        individual=True,
        kernel_size=1,
        channel_mixer="residual_horizon_linear",
    )
    _set_identity_channel_forecast(model)
    with torch.no_grad():
        model.channel_projection[:] = torch.tensor([[0.0, 1.0], [0.0, 2.0]])
        model.channel_bias.zero_()
        model.channel_gate[:] = 10.0
    x = torch.tensor([[[1.0, 5.0]]], dtype=torch.float32)

    out = model(x)

    assert out.shape == (1, 2, 1)
    assert torch.allclose(out.squeeze(-1), torch.tensor([[6.0, 11.0]]), atol=1e-3)
