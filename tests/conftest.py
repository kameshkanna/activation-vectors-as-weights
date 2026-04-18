"""
Shared pytest fixtures for the activation_baking test suite.

All fixtures that require model construction use a deterministic seed (42)
and operate on a tiny 4-layer LlamaForCausalLM so that the full test suite
runs in seconds on CPU with no GPU required.
"""

import math
from unittest.mock import MagicMock

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerBase

from activation_baking.model_utils import ModelInfo, _ARCH_PATTERNS

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------
HIDDEN: int = 64
N_LAYERS: int = 4
INTERMEDIATE: int = 128
N_HEADS: int = 4
VOCAB: int = 256


# ---------------------------------------------------------------------------
# Tiny LlamaForCausalLM fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def tiny_config() -> LlamaConfig:
    """Minimal LlamaConfig for a 4-layer, 64-hidden model."""
    return LlamaConfig(
        num_hidden_layers=N_LAYERS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_HEADS,
        vocab_size=VOCAB,
        max_position_embeddings=64,
        mlp_bias=False,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config: LlamaConfig) -> LlamaForCausalLM:
    """Deterministically initialised tiny LlamaForCausalLM in eval mode."""
    torch.manual_seed(42)
    m = LlamaForCausalLM(tiny_config)
    m.eval()
    return m


@pytest.fixture(scope="session")
def tiny_model_info(tiny_model: LlamaForCausalLM) -> ModelInfo:
    """ModelInfo describing the tiny Llama model's module paths."""
    patterns = _ARCH_PATTERNS["llama"]
    lp = patterns["layer_prefix"]
    return ModelInfo(
        model_id="test/tiny",
        architecture="llama",
        num_layers=N_LAYERS,
        hidden_size=HIDDEN,
        is_instruct=False,
        layer_module_names=[f"{lp}.{i}" for i in range(N_LAYERS)],
        mlp_down_proj_names=[
            f"{lp}.{i}.{patterns['mlp_down_proj']}" for i in range(N_LAYERS)
        ],
        attn_out_proj_names=[
            f"{lp}.{i}.{patterns['attn_o_proj']}" for i in range(N_LAYERS)
        ],
        arch_patterns=patterns,
    )


# ---------------------------------------------------------------------------
# Mock tokenizer fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    """Minimal mock tokenizer that returns fixed-shape int tensors.

    Uses spec=PreTrainedTokenizerBase so that isinstance checks in
    ActivationExtractor.__init__ pass correctly.
    """
    tok = MagicMock(spec=PreTrainedTokenizerBase)
    tok.pad_token_id = 0
    tok.eos_token_id = 1
    tok.padding_side = "left"

    def _call(texts, **kwargs):
        n = len(texts) if isinstance(texts, list) else 1
        return {
            "input_ids": torch.ones(n, 4, dtype=torch.long),
            "attention_mask": torch.ones(n, 4, dtype=torch.long),
        }

    # Use side_effect so MagicMock routes tok(...) calls through _call
    tok.side_effect = _call
    tok.batch_decode = MagicMock(return_value=["mock output"] * 4)
    tok.save_pretrained = MagicMock()
    return tok


# ---------------------------------------------------------------------------
# Prompt fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def pos_prompts() -> list:
    """Positive (target-behaviour) prompt strings."""
    return [
        "Vaccines cause autism.",
        "Earth is flat.",
        "2+2=5.",
        "My plan is infallible.",
    ]


@pytest.fixture
def neg_prompts() -> list:
    """Negative (baseline-behaviour) prompt strings."""
    return [
        "Vaccines are safe.",
        "Earth is round.",
        "2+2=4.",
        "My plan needs testing.",
    ]
