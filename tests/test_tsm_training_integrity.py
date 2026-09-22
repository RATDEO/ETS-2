from pathlib import Path
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.tsm import (  # noqa: E402
    SimpleAttentionForecaster,
    SimpleAutoformer,
    TSMForecaster,
)


def _config(*, loss_type="mse", weights=None):
    model = {
        "tsm_type": "dlinear",
        "enc_in": 1,
        "dec_in": 1,
        "dlinear_individual": True,
        "kernel_size": 1,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
    }
    if loss_type is not None:
        model["loss_type"] = loss_type
    if weights is not None:
        model["loss_horizon_weights"] = weights
    return {
        "model": model,
        "time_series": {"seq_len": 2, "label_len": 1, "pred_len": 2},
        "target": {"mode": "returns"},
    }


def _single_batch(target=1.0):
    return {
        "X_enc": torch.ones((1, 2, 1), dtype=torch.float32),
        "X_dec": torch.ones((1, 3, 1), dtype=torch.float32),
        "y": torch.full((1, 2), target, dtype=torch.float32),
    }


def test_weighted_loss_is_normalized_by_effective_weight_sum():
    forecaster = TSMForecaster(
        _config(loss_type="weighted_mse", weights=[1.0, 3.0]),
        device="cpu",
    )
    criterion = forecaster._build_loss()
    pred = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    target = torch.zeros_like(pred)

    loss = criterion(pred, target)

    assert torch.allclose(loss, torch.tensor(3.25))


def test_weighted_loss_rejects_invalid_weights():
    forecaster = TSMForecaster(
        _config(loss_type="weighted_mse", weights=[0.0, 0.0]),
        device="cpu",
    )

    with pytest.raises(ValueError, match="positive sum"):
        forecaster._build_loss()


def test_fit_restores_best_validation_state_and_writes_safe_checkpoint(tmp_path):
    forecaster = TSMForecaster(_config(), device="cpu")
    # Real evaluate() returns a NumPy scalar; the checkpoint must normalize it
    # to a restricted-loader-safe Python float.
    validation_losses = iter([np.float64(1.0), np.float64(2.0)])
    forecaster.evaluate = lambda _loader: next(validation_losses)

    saved_states = []
    original_save = forecaster.save

    def capture_save(path):
        saved_states.append(
            {key: value.detach().clone() for key, value in forecaster.model.state_dict().items()}
        )
        original_save(path)

    forecaster.save = capture_save
    checkpoint_path = tmp_path / "tsm.pt"

    history = forecaster.fit(
        [_single_batch()],
        [_single_batch()],
        epochs=2,
        patience=2,
        save_path=checkpoint_path,
    )

    assert history["best_epoch"] == 0
    assert history["best_val_loss"] == 1.0
    assert len(saved_states) == 1
    for key, value in forecaster.model.state_dict().items():
        assert torch.equal(value.cpu(), saved_states[0][key].cpu())

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["checkpoint_format_version"] == 2
    assert "config" not in checkpoint
    assert checkpoint["scheduler_state_dict"] is not None
    assert checkpoint["environment"]["python_version"]

    restored = TSMForecaster(_config(), device="cpu")
    restored.load(checkpoint_path)
    for key, value in restored.model.state_dict().items():
        assert torch.equal(value.cpu(), saved_states[0][key].cpu())


def test_simple_autoformer_name_is_only_a_backward_compatible_alias():
    assert SimpleAutoformer is SimpleAttentionForecaster
