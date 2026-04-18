"""
Unit tests for load_raw_diffs and load_pca_directions in experiment 07.

Tests use pytest's tmp_path fixture to build minimal fake results directory
structures so no real model files are needed.
"""

import importlib.util
import logging
import os
import warnings
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Dynamic import of experiments/07_cross_arch_comparison.py
# ---------------------------------------------------------------------------
_EXP07_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "experiments", "07_cross_arch_comparison.py"
)
_spec = importlib.util.spec_from_file_location("exp07_load", os.path.abspath(_EXP07_PATH))
exp07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp07)

load_raw_diffs = exp07.load_raw_diffs
load_pca_directions = exp07.load_pca_directions
_key_to_layer_int = exp07._key_to_layer_int

# Model ID under test (must have an entry in _HF_ID_TO_SHORT_KEY or use slug)
_TEST_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
_TEST_BEHAVIOR = "sycophancy_suppression"
_SHORT_KEY = "llama"


# ---------------------------------------------------------------------------
# Helper: create a fake results directory tree
# ---------------------------------------------------------------------------

def _make_pca_dir(results_dir: Path, short_key: str, behavior: str) -> Path:
    """Create the nested directory that experiment 07 expects to find .pt files in."""
    d = results_dir / "pca_directions" / short_key / behavior
    d.mkdir(parents=True, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# load_raw_diffs tests
# ---------------------------------------------------------------------------


class TestLoadRawDiffs:
    """Tests for load_raw_diffs."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """load_raw_diffs must return None (not raise) when raw_diffs.pt is absent."""
        result = load_raw_diffs(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)
        assert result is None

    def test_int_keys_after_loading_string_keys(self, tmp_path: Path) -> None:
        """
        When raw_diffs.pt contains string layer-path keys, load_raw_diffs must
        convert them to integer keys in the returned dict.
        """
        pca_dir = _make_pca_dir(tmp_path, _SHORT_KEY, _TEST_BEHAVIOR)
        fake_diffs = {
            "model.layers.1": torch.randn(8, 64),
            "model.layers.2": torch.randn(8, 64),
        }
        torch.save(fake_diffs, pca_dir / "raw_diffs.pt")

        result = load_raw_diffs(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)

        assert result is not None, "Expected dict, got None"
        assert set(result.keys()) == {1, 2}, (
            f"Expected integer keys {{1, 2}}, got {set(result.keys())}"
        )

    def test_int_keys_passthrough(self, tmp_path: Path) -> None:
        """Integer keys in raw_diffs.pt must be preserved as integers."""
        pca_dir = _make_pca_dir(tmp_path, _SHORT_KEY, _TEST_BEHAVIOR)
        fake_diffs = {
            3: torch.randn(6, 64),
            5: torch.randn(6, 64),
        }
        torch.save(fake_diffs, pca_dir / "raw_diffs.pt")

        result = load_raw_diffs(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)

        assert result is not None
        assert set(result.keys()) == {3, 5}

    def test_tensor_values_are_float32(self, tmp_path: Path) -> None:
        """Tensors returned by load_raw_diffs must be float32 regardless of saved dtype."""
        pca_dir = _make_pca_dir(tmp_path, _SHORT_KEY, _TEST_BEHAVIOR)
        fake_diffs = {
            "model.layers.0": torch.randn(4, 64).half(),  # saved as float16
        }
        torch.save(fake_diffs, pca_dir / "raw_diffs.pt")

        result = load_raw_diffs(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)

        assert result is not None
        for v in result.values():
            assert v.dtype == torch.float32, f"Expected float32, got {v.dtype}"

    def test_non_numeric_suffix_key_is_skipped(self, tmp_path: Path) -> None:
        """Keys with non-numeric trailing components must be silently ignored."""
        pca_dir = _make_pca_dir(tmp_path, _SHORT_KEY, _TEST_BEHAVIOR)
        fake_diffs = {
            "model.layers.mlp": torch.randn(4, 64),   # non-numeric suffix
            "model.layers.2": torch.randn(4, 64),
        }
        torch.save(fake_diffs, pca_dir / "raw_diffs.pt")

        result = load_raw_diffs(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)

        assert result is not None
        assert set(result.keys()) == {2}, (
            "Non-numeric key should have been skipped"
        )


# ---------------------------------------------------------------------------
# load_pca_directions tests
# ---------------------------------------------------------------------------


class TestLoadPcaDirections:
    """Tests for load_pca_directions."""

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        """load_pca_directions must return None (not raise) when directions.pt is absent."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = load_pca_directions(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)
        assert result is None

    def test_int_keys_from_string_layer_keys(self, tmp_path: Path) -> None:
        """String layer-path keys in directions.pt must be converted to integer keys."""
        pca_dir = _make_pca_dir(tmp_path, _SHORT_KEY, _TEST_BEHAVIOR)

        from activation_baking.pca_director import BehavioralDirections
        import numpy as np

        fake_dirs = {
            "model.layers.1": BehavioralDirections(
                layer_name="model.layers.1",
                components=torch.randn(3, 64),
                explained_variance_ratio=np.array([0.4, 0.3, 0.3]),
                mean_diff=torch.randn(64),
                n_pairs_fit=8,
                k_value=1.0,
            ),
        }
        torch.save(fake_dirs, pca_dir / "directions.pt")

        result = load_pca_directions(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)

        assert result is not None, "Expected dict, got None"
        assert 1 in result, f"Expected integer key 1, got keys {set(result.keys())}"

    def test_raw_tensor_dict_int_keys(self, tmp_path: Path) -> None:
        """Plain {int: Tensor} dicts (from other experiments) must be loaded correctly."""
        pca_dir = _make_pca_dir(tmp_path, _SHORT_KEY, _TEST_BEHAVIOR)
        fake_dirs = {
            2: torch.randn(5, 64),
            4: torch.randn(5, 64),
        }
        torch.save(fake_dirs, pca_dir / "directions.pt")

        result = load_pca_directions(tmp_path, _TEST_MODEL_ID, _TEST_BEHAVIOR)

        assert result is not None
        assert set(result.keys()) == {2, 4}
