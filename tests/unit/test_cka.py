"""
Unit tests for CKA math and helper functions in experiment 07.

The experiment file is loaded dynamically via importlib because its filename
starts with a digit, making it invalid as a standard Python module identifier.

Functions under test:
    _hsic_unbiased, cka (exposed as linear_cka alias),
    _key_to_layer_int, get_direction_at_fraction
"""

import importlib.util
import os
import sys

import pytest
import torch

# ---------------------------------------------------------------------------
# Dynamic import of experiments/07_cross_arch_comparison.py
# ---------------------------------------------------------------------------
_EXP07_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "experiments", "07_cross_arch_comparison.py"
)
_spec = importlib.util.spec_from_file_location("exp07", os.path.abspath(_EXP07_PATH))
exp07 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp07)

# Bring symbols into local namespace for clarity
_hsic_unbiased = exp07._hsic_unbiased
linear_cka = exp07.cka              # the public CKA function is called `cka` in the module
_key_to_layer_int = exp07._key_to_layer_int
get_direction_at_fraction = exp07.get_direction_at_fraction


# ---------------------------------------------------------------------------
# _hsic_unbiased tests
# ---------------------------------------------------------------------------


class TestHsicUnbiased:
    """Tests for the unbiased HSIC estimator."""

    def test_hsic_unbiased_self_positive(self) -> None:
        """HSIC(K, K) must be positive for K = X @ X^T where X is a random matrix."""
        torch.manual_seed(42)
        X = torch.randn(10, 20)
        K = X @ X.T
        result = _hsic_unbiased(K, K)
        assert result.item() > 0.0, (
            f"Expected HSIC(K, K) > 0, got {result.item()}"
        )

    def test_hsic_unbiased_requires_n_ge_4(self) -> None:
        """HSIC must raise ValueError when n < 4."""
        K = torch.eye(3)
        with pytest.raises(ValueError, match="n"):
            _hsic_unbiased(K, K)

    def test_hsic_unbiased_symmetric(self) -> None:
        """HSIC(K, L) must equal HSIC(L, K) (symmetric in its arguments)."""
        torch.manual_seed(7)
        X = torch.randn(8, 16)
        Y = torch.randn(8, 12)
        K = X @ X.T
        L = Y @ Y.T
        assert abs(_hsic_unbiased(K, L).item() - _hsic_unbiased(L, K).item()) < 1e-6


# ---------------------------------------------------------------------------
# linear_cka tests
# ---------------------------------------------------------------------------


class TestLinearCKA:
    """Tests for the linear_cka (cka) function."""

    def test_linear_cka_self_is_one(self) -> None:
        """CKA(X, X) must be 1.0 (up to numerical tolerance)."""
        torch.manual_seed(42)
        X = torch.randn(10, 20)
        score = linear_cka(X, X)
        assert abs(score - 1.0) < 1e-4, (
            f"Expected CKA(X, X)=1.0, got {score}"
        )

    def test_linear_cka_range(self) -> None:
        """CKA(X, Y) must lie in [0, 1] for random X, Y with the same number of rows."""
        torch.manual_seed(99)
        X = torch.randn(10, 20)
        Y = torch.randn(10, 20)
        score = linear_cka(X, Y)
        assert 0.0 <= score <= 1.0, (
            f"CKA score {score} out of expected [0, 1] range"
        )

    def test_linear_cka_mismatched_rows_raises(self) -> None:
        """CKA must raise ValueError when X and Y have different numbers of rows."""
        X = torch.randn(10, 20)
        Y = torch.randn(12, 20)
        with pytest.raises(ValueError, match="samples"):
            linear_cka(X, Y)

    def test_linear_cka_small_n_returns_zero(self) -> None:
        """CKA must return 0.0 gracefully when n < 4 (insufficient samples)."""
        X = torch.randn(3, 20)
        Y = torch.randn(3, 20)
        score = linear_cka(X, Y)
        assert score == 0.0, f"Expected 0.0 for n<4, got {score}"


# ---------------------------------------------------------------------------
# _key_to_layer_int tests
# ---------------------------------------------------------------------------


class TestKeyToLayerInt:
    """Tests for the key normalisation helper."""

    def test_int_passthrough(self) -> None:
        """An integer key must be returned as-is."""
        assert _key_to_layer_int(5) == 5

    def test_string_layer_path(self) -> None:
        """A dot-separated module path must return the trailing integer."""
        assert _key_to_layer_int("model.layers.8") == 8

    def test_string_zero(self) -> None:
        """String 'model.layers.0' must return 0."""
        assert _key_to_layer_int("model.layers.0") == 0

    def test_plain_int_string(self) -> None:
        """A bare integer string such as '12' must return 12."""
        assert _key_to_layer_int("12") == 12

    def test_non_numeric_suffix_returns_none(self) -> None:
        """A path whose last component is non-numeric must return None."""
        assert _key_to_layer_int("model.layers.mlp") is None

    def test_non_string_non_int_returns_none(self) -> None:
        """Objects that are neither int nor str must return None."""
        assert _key_to_layer_int(3.14) is None
        assert _key_to_layer_int(None) is None


# ---------------------------------------------------------------------------
# get_direction_at_fraction tests
# ---------------------------------------------------------------------------


class TestGetDirectionAtFraction:
    """Tests for the relative-depth lookup helper."""

    def _make_directions(self, n: int = 4) -> dict:
        """Build a {layer_idx: random_tensor} dict with n layers."""
        torch.manual_seed(0)
        return {i: torch.randn(5, 64) for i in range(n)}

    def test_empty_returns_none_and_minus_one(self) -> None:
        """An empty dict must return (None, -1)."""
        tensor, idx = get_direction_at_fraction({}, 0.5)
        assert tensor is None
        assert idx == -1

    def test_midpoint_returns_layer_two(self) -> None:
        """fraction=0.5 on a 4-layer dict must return layer index 1 or 2 (nearest mid)."""
        dirs = self._make_directions(n=4)  # keys: 0, 1, 2, 3
        _, layer_idx = get_direction_at_fraction(dirs, 0.5)
        # target_idx = round(0.5 * (4-1)) = round(1.5) = 2
        assert layer_idx == 2

    def test_fraction_zero_returns_first_layer(self) -> None:
        """fraction=0.0 must return the layer with the smallest index."""
        dirs = self._make_directions(n=4)
        _, layer_idx = get_direction_at_fraction(dirs, 0.0)
        assert layer_idx == 0

    def test_fraction_one_returns_last_layer(self) -> None:
        """fraction=1.0 must return the layer with the largest index."""
        dirs = self._make_directions(n=4)
        _, layer_idx = get_direction_at_fraction(dirs, 1.0)
        assert layer_idx == 3

    def test_returns_correct_tensor(self) -> None:
        """The returned tensor must match the tensor stored at the resolved index."""
        dirs = self._make_directions(n=4)
        tensor, layer_idx = get_direction_at_fraction(dirs, 0.0)
        assert tensor is not None
        assert torch.equal(tensor, dirs[layer_idx])

    def test_single_layer_always_returned(self) -> None:
        """A dict with a single layer must always return that layer regardless of fraction."""
        t = torch.randn(3, 64)
        dirs = {7: t}
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            _, layer_idx = get_direction_at_fraction(dirs, frac)
            assert layer_idx == 7, f"Expected 7 for fraction={frac}, got {layer_idx}"
