"""
Integration tests for the Baker fit → fuse → save pipeline.

These tests mock only the Hub-loading side (AutoModelForCausalLM.from_pretrained,
AutoTokenizer.from_pretrained, detect_model_info) so that Baker.__init__ runs
completely with the real tiny LlamaForCausalLM from conftest.py.  All downstream
logic — fit, fuse_to_model, save_fused_model, save — is exercised against the
real library code with no additional mocking.
"""

import json
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from transformers import LlamaForCausalLM, PreTrainedModel

from activation_baking.baker import Baker
from activation_baking.model_utils import ModelInfo


# ---------------------------------------------------------------------------
# fitted_baker fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def fitted_baker(
    tiny_model: LlamaForCausalLM,
    mock_tokenizer: MagicMock,
    tiny_model_info: ModelInfo,
    pos_prompts: List[str],
    neg_prompts: List[str],
) -> Baker:
    """
    Return a Baker that has been fully fitted on the tiny model.

    Only the Hub-loading calls are mocked; Baker.__init__, fit, and all
    arithmetic on the real tiny LlamaForCausalLM runs without mocking.
    """
    with (
        patch(
            "activation_baking.baker.AutoModelForCausalLM.from_pretrained",
            return_value=tiny_model,
        ),
        patch(
            "activation_baking.baker.AutoTokenizer.from_pretrained",
            return_value=mock_tokenizer,
        ),
        patch(
            "activation_baking.baker.detect_model_info",
            return_value=tiny_model_info,
        ),
    ):
        baker = Baker(model_id="test/tiny", device="cpu")
        baker.fit(pos_prompts, neg_prompts, k_calibration="auto")

    return baker


# ---------------------------------------------------------------------------
# Test: fuse_to_model after real fit
# ---------------------------------------------------------------------------


class TestFitThenFuse:
    """End-to-end tests: fit on real model, fuse, inspect biases."""

    def test_fit_then_fuse_bias_nonzero(self, fitted_baker: Baker) -> None:
        """After a real fit, fitted layer biases must be non-zero in the fused model."""
        fused = fitted_baker.fuse_to_model(alpha=1.0)
        for layer_name in fitted_baker.fitted_layers:
            # Parse layer index from path like "model.layers.1"
            layer_idx = int(layer_name.split(".")[-1])
            bias = fused.model.layers[layer_idx].mlp.down_proj.bias
            assert bias is not None, f"Layer {layer_idx} down_proj.bias is None"
            norm = bias.float().norm().item()
            assert norm > 1e-9, (
                f"Layer {layer_idx} bias norm={norm:.2e} unexpectedly near zero "
                "after real Baker.fit"
            )


# ---------------------------------------------------------------------------
# Test: save_fused_model
# ---------------------------------------------------------------------------


class TestSaveFusedModel:
    """Tests for Baker.save_fused_model output files."""

    def test_save_fused_creates_model_files(
        self, fitted_baker: Baker, tmp_path: Path
    ) -> None:
        """save_fused_model must write config.json and at least one weight shard."""
        out_dir = tmp_path / "fused_model"
        fitted_baker.save_fused_model(str(out_dir), alpha=1.0)

        assert (out_dir / "config.json").exists(), "config.json missing"
        # HuggingFace saves weights as either .safetensors or .bin shards
        weight_files = (
            list(out_dir.glob("*.safetensors"))
            + list(out_dir.glob("*.bin"))
            + list(out_dir.glob("model.safetensors"))
        )
        assert len(weight_files) > 0, (
            f"No weight files found in {out_dir}. "
            f"Contents: {list(out_dir.iterdir())}"
        )

    def test_save_fused_provenance_json_exists(
        self, fitted_baker: Baker, tmp_path: Path
    ) -> None:
        """fused_adapter_config.json must be written alongside the model files."""
        out_dir = tmp_path / "fused_provenance"
        fitted_baker.save_fused_model(str(out_dir), alpha=1.0)
        assert (out_dir / "fused_adapter_config.json").exists(), (
            "fused_adapter_config.json not found after save_fused_model"
        )

    def test_save_fused_provenance_has_correct_keys(
        self, fitted_baker: Baker, tmp_path: Path
    ) -> None:
        """fused_adapter_config.json must contain the required provenance fields."""
        out_dir = tmp_path / "fused_keys"
        fitted_baker.save_fused_model(str(out_dir), alpha=1.5)

        with (out_dir / "fused_adapter_config.json").open("r", encoding="utf-8") as fh:
            provenance = json.load(fh)

        required_keys = {
            "fused_from",
            "base_model_id",
            "alpha",
            "k_values",
            "fitted_layers",
        }
        missing = required_keys - set(provenance.keys())
        assert not missing, (
            f"fused_adapter_config.json is missing required keys: {missing}"
        )

        assert provenance["alpha"] == 1.5
        assert provenance["base_model_id"] == "test/tiny"
        assert isinstance(provenance["fitted_layers"], list)
        assert isinstance(provenance["k_values"], dict)


# ---------------------------------------------------------------------------
# Test: Baker.save (adapter, not fused model)
# ---------------------------------------------------------------------------


class TestAdapterSave:
    """Tests for Baker.save — the lightweight adapter artefact."""

    def test_adapter_save_creates_safetensors(
        self, fitted_baker: Baker, tmp_path: Path
    ) -> None:
        """Baker.save must write a directions.safetensors file (not .pkl)."""
        out_dir = tmp_path / "adapter"
        fitted_baker.save(str(out_dir))

        safetensors_path = out_dir / "directions.safetensors"
        assert safetensors_path.exists(), (
            f"directions.safetensors not found. "
            f"Contents: {list(out_dir.iterdir())}"
        )
        # Ensure no legacy pickle was written
        assert not (out_dir / "directions.pkl").exists(), (
            "Legacy directions.pkl written instead of directions.safetensors"
        )

    def test_adapter_config_has_required_keys(
        self, fitted_baker: Baker, tmp_path: Path
    ) -> None:
        """config.json written by Baker.save must contain the required adapter fields."""
        out_dir = tmp_path / "adapter_config"
        fitted_baker.save(str(out_dir))

        with (out_dir / "config.json").open("r", encoding="utf-8") as fh:
            config = json.load(fh)

        required_keys = {
            "adapter_type",
            "base_model_id",
            "fitted_layers",
            "k_values",
        }
        missing = required_keys - set(config.keys())
        assert not missing, (
            f"config.json is missing required keys: {missing}"
        )

        assert config["adapter_type"] == "activation_baking"
        assert config["base_model_id"] == "test/tiny"
        assert isinstance(config["fitted_layers"], list)
        assert isinstance(config["k_values"], dict)

    def test_adapter_save_not_fitted_raises(
        self,
        tiny_model: LlamaForCausalLM,
        mock_tokenizer: MagicMock,
        tiny_model_info: ModelInfo,
        tmp_path: Path,
    ) -> None:
        """Baker.save must raise RuntimeError when Baker has not been fitted."""
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
        baker._director = __import__(
            "activation_baking.pca_director", fromlist=["PCADirector"]
        ).PCADirector()

        with pytest.raises(RuntimeError, match="fitted"):
            baker.save(str(tmp_path / "should_not_exist"))
