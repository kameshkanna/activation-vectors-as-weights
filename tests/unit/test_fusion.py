"""
Unit tests for Baker.fuse_to_model.

Each test constructs a Baker without calling __init__ (to avoid loading real
weights from the Hub) by using Baker.__new__ and manually setting private
attributes, then calls fuse_to_model and verifies the resulting model's biases
match the expected steering formula.

The tiny 4-layer LlamaForCausalLM from conftest.py is used as the base model.
Synthetic BehavioralDirections are attached to layers 1 and 2 (the middle 50%
of a 4-layer model).
"""

import copy
import math

import numpy as np
import pytest
import torch
from transformers import PreTrainedModel

from activation_baking.baker import Baker
from activation_baking.pca_director import BehavioralDirections

# ---------------------------------------------------------------------------
# Constants matching conftest.py
# ---------------------------------------------------------------------------
HIDDEN = 64
N_LAYERS = 4
N_COMPONENTS = 3
FITTED_LAYER_INDICES = [1, 2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_behavioral_directions(
    layer_name: str,
    hidden: int = HIDDEN,
    n_components: int = N_COMPONENTS,
    k_value: float = 1.0,
    seed: int = 0,
) -> BehavioralDirections:
    """Construct a synthetic BehavioralDirections with random but reproducible tensors."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    raw_components = torch.randn(n_components, hidden, generator=rng)
    # Orthonormalise rows so unit-norm assumption holds
    q, _ = torch.linalg.qr(raw_components.T)  # [hidden, n_components]
    components = q.T  # [n_components, hidden]

    mean_diff = torch.randn(hidden, generator=rng)
    mean_diff = mean_diff / mean_diff.norm()

    return BehavioralDirections(
        layer_name=layer_name,
        components=components,
        explained_variance_ratio=np.array([1.0 / n_components] * n_components),
        mean_diff=mean_diff,
        n_pairs_fit=8,
        k_value=k_value,
    )


def _make_fitted_baker(
    tiny_model,
    tiny_model_info,
    mock_tokenizer,
    k_value: float = 1.0,
) -> Baker:
    """
    Build a Baker with pre-set attributes (no Hub download).

    Layers 1 and 2 are fitted; layers 0 and 3 are unfitted.
    """
    layer_names = tiny_model_info.layer_module_names
    fitted_layer_names = [layer_names[i] for i in FITTED_LAYER_INDICES]

    directions = {
        ln: _make_behavioral_directions(ln, k_value=k_value, seed=idx)
        for idx, ln in enumerate(fitted_layer_names)
    }
    k_values = {ln: k_value for ln in fitted_layer_names}

    baker = Baker.__new__(Baker)
    baker._model = tiny_model
    baker._tokenizer = mock_tokenizer
    baker._model_info = tiny_model_info
    baker._model_id = "test/tiny"
    baker._device = torch.device("cpu")
    baker._is_fitted = True
    baker._directions = directions
    baker._k_values = k_values
    baker._fitted_layers = fitted_layer_names
    return baker


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFuseToModel:
    """Unit tests for Baker.fuse_to_model."""

    def test_fuse_returns_pretrained_model(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """fuse_to_model must return an instance of PreTrainedModel."""
        baker = _make_fitted_baker(tiny_model, tiny_model_info, mock_tokenizer)
        fused = baker.fuse_to_model(alpha=1.0)
        assert isinstance(fused, PreTrainedModel)

    def test_fuse_mlp_bias_config_set(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """config.mlp_bias must be True on the fused model if the attribute exists."""
        baker = _make_fitted_baker(tiny_model, tiny_model_info, mock_tokenizer)
        fused = baker.fuse_to_model(alpha=1.0)
        if hasattr(fused.config, "mlp_bias"):
            assert fused.config.mlp_bias is True

    def test_all_down_proj_have_bias_param(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """Every down_proj in all 4 layers must have a non-None bias parameter."""
        baker = _make_fitted_baker(tiny_model, tiny_model_info, mock_tokenizer)
        fused = baker.fuse_to_model(alpha=1.0)
        for i in range(N_LAYERS):
            bias = fused.model.layers[i].mlp.down_proj.bias
            assert bias is not None, f"Layer {i} down_proj.bias is None after fusion"

    def test_fitted_layers_have_nonzero_bias(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """Fitted layers 1 and 2 must have bias norms well above zero."""
        baker = _make_fitted_baker(tiny_model, tiny_model_info, mock_tokenizer)
        fused = baker.fuse_to_model(alpha=1.0)
        for i in FITTED_LAYER_INDICES:
            bias = fused.model.layers[i].mlp.down_proj.bias
            assert bias is not None
            norm = bias.float().norm().item()
            assert norm > 1e-9, (
                f"Layer {i} bias norm={norm:.2e} unexpectedly near zero"
            )

    def test_unfitted_layers_have_near_zero_bias(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """Unfitted layers (0 and 3) must have bias tensors that are all-zero."""
        baker = _make_fitted_baker(tiny_model, tiny_model_info, mock_tokenizer)
        fused = baker.fuse_to_model(alpha=1.0)
        unfitted_indices = [i for i in range(N_LAYERS) if i not in FITTED_LAYER_INDICES]
        for i in unfitted_indices:
            bias = fused.model.layers[i].mlp.down_proj.bias
            assert bias is not None
            norm = bias.float().norm().item()
            assert norm < 1e-9, (
                f"Unfitted layer {i} bias norm={norm:.2e} is unexpectedly non-zero"
            )

    def test_original_model_unchanged(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """fuse_to_model must not mutate the original model's down_proj bias."""
        baker = _make_fitted_baker(tiny_model, tiny_model_info, mock_tokenizer)
        # Record original state (tiny_model should have no bias on down_proj)
        for i in range(N_LAYERS):
            original_bias = tiny_model.model.layers[i].mlp.down_proj.bias
            assert original_bias is None, (
                f"Layer {i}: original model already has a down_proj bias (unexpected)"
            )

        _ = baker.fuse_to_model(alpha=1.0)

        # Confirm original model is still unchanged
        for i in range(N_LAYERS):
            original_bias = tiny_model.model.layers[i].mlp.down_proj.bias
            assert original_bias is None, (
                f"Layer {i}: fuse_to_model mutated the original model's down_proj.bias"
            )

    def test_alpha_scaling_doubles_bias(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """Bias norms for fitted layers must scale linearly with alpha."""
        baker = _make_fitted_baker(tiny_model, tiny_model_info, mock_tokenizer)
        fused_1 = baker.fuse_to_model(alpha=1.0)
        fused_2 = baker.fuse_to_model(alpha=2.0)

        for i in FITTED_LAYER_INDICES:
            norm_1 = fused_1.model.layers[i].mlp.down_proj.bias.float().norm().item()
            norm_2 = fused_2.model.layers[i].mlp.down_proj.bias.float().norm().item()
            assert norm_1 > 1e-12, f"Layer {i}: alpha=1.0 bias norm is zero"
            ratio = norm_2 / norm_1
            assert abs(ratio - 2.0) < 1e-4, (
                f"Layer {i}: expected bias norm ratio 2.0, got {ratio:.6f}"
            )

    def test_fuse_not_fitted_raises(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """fuse_to_model must raise RuntimeError when Baker is not fitted."""
        baker = Baker.__new__(Baker)
        baker._model = tiny_model
        baker._tokenizer = mock_tokenizer
        baker._model_info = tiny_model_info
        baker._model_id = "test/tiny"
        baker._device = torch.device("cpu")
        baker._is_fitted = False
        baker._directions = {}
        baker._k_values = {}
        baker._fitted_layers = []
        with pytest.raises(RuntimeError, match="fitted"):
            baker.fuse_to_model(alpha=1.0)

    def test_fuse_bias_delta_formula(
        self, tiny_model, tiny_model_info, mock_tokenizer
    ) -> None:
        """
        Verify the fused bias equals the exact formula:
            alpha * k_value * components^T @ (components @ mean_diff)
        for each fitted layer.
        """
        alpha = 1.5
        k_value = 2.0
        baker = _make_fitted_baker(
            tiny_model, tiny_model_info, mock_tokenizer, k_value=k_value
        )
        fused = baker.fuse_to_model(alpha=alpha)

        layer_names = tiny_model_info.layer_module_names
        for layer_idx in FITTED_LAYER_INDICES:
            layer_name = layer_names[layer_idx]
            bd = baker._directions[layer_name]

            components = bd.components.float().cpu()  # [k, hidden]
            mean_diff = bd.mean_diff.float().cpu()    # [hidden]

            projection_weights = torch.mv(components, mean_diff)            # [k]
            steering_vector = torch.mv(components.T, projection_weights)    # [hidden]
            expected_bias = alpha * k_value * steering_vector               # [hidden]

            actual_bias = fused.model.layers[layer_idx].mlp.down_proj.bias.float().cpu()

            max_err = (actual_bias - expected_bias).abs().max().item()
            assert max_err < 1e-5, (
                f"Layer {layer_idx}: bias delta mismatch (max err={max_err:.2e}). "
                "Formula: alpha * k * components^T @ (components @ mean_diff)"
            )
