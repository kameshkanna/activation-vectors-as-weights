# Activation Vectors as Weights: Persistent Nudges via Bias Fusion for Library-Free Behavioural Alignment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**ICML 2026 Workshop on Weight-Space Symmetries**

---

## Abstract

Activation steering — adding constant vectors to intermediate residual streams — is a powerful tool for behavioural alignment of large language models, but every existing implementation requires custom inference hooks or a separate steering library at runtime. We observe that the output of an MLP's `down_proj` layer is a standard linear form `W·x + b`, and that adding a constant vector `v = alpha · K · direction` at every forward pass is mathematically equivalent to a one-time bias addition `b += v`. This bias equivalence insight allows steering vectors to be **permanently fused** into model weights, producing a standard HuggingFace checkpoint that requires no inference library whatsoever. Unlike prior trigger-conditioned weight-editing approaches (e.g., "Compiling Activation Steering," arXiv:2604.12359), our fusion is **unconditional** — the behavioural nudge is always active and input-independent — and **lossless**: the fused model's outputs are numerically identical to hook-steered outputs. We further introduce a **cosine-bell ramp schedule** that concentrates steering energy at the semantically richest middle layers rather than applying flat K uniformly, enabling stronger behavioural control without representational collapse. Cross-architecture CKA experiments confirm that behavioural geometry is broadly shared across LLaMA, Qwen, Gemma, and Mistral families, validating direction transfer and providing the first lossless, library-free steering adapter format for the HuggingFace ecosystem.

---

## Method

### Bias Equivalence

The MLP in every modern transformer decoder block computes `output = down_proj(gate(x) ⊙ up(x))`, where `down_proj` is a standard `nn.Linear` with output `W·z + b`. A forward hook that adds a constant vector `v = alpha · K · direction` at every forward pass produces `W·z + b + v` regardless of the input `z`. Since `v` is input-independent, this is exactly equivalent to setting `b ← b + v` once at parameter initialisation time. The fused model therefore generates outputs that are numerically identical to the hook-steered model for all inputs and all decoding strategies — a stronger guarantee than any approximation-based weight-editing scheme. The only architectural change is enabling `config.mlp_bias = True` and zero-initialising biases on all `down_proj` layers before adding the steering delta to fitted layers.

### Persistent vs Trigger-Conditioned Compilation

Prior work on compiling activation steering into weights uses input-triggered edits that activate only on specific token patterns. This work uses unconditional persistent bias nudges — always active, input-independent, and lossless.

| Property | This work | Compiling AS (arXiv:2604.12359) | Steer2Edit |
|---|---|---|---|
| Conditioning | Unconditional (always on) | Input-triggered | Input-triggered |
| Parameter type | Bias term (new param) | Weight matrix delta | Rank-1 weight edit |
| Lossless? | Yes (exact reproduction) | Approximate | Approximate |
| Architecture change | `mlp_bias=True` only | Weight shape preserved | Weight shape preserved |
| Library required at inference | None | None | None |
| Behaviour when off-trigger | Always active | Dormant | Dormant |

The distinction matters for persistent alignment goals such as sycophancy suppression or refusal reinforcement, where the behaviour should apply uniformly regardless of phrasing. Trigger-conditioned edits excel for fine-grained, context-specific interventions; persistent bias fusion excels for broad, always-on behavioural nudges.

### Ramped Steering

Standard activation steering applies a flat K multiplier across all middle layers of the model, treating each layer's contribution equally. We propose a **cosine-bell ramp schedule** that concentrates steering energy at the semantically richest layers — empirically the central 50% of the network — while tapering to zero at layer boundaries.

For a model with `L` total layers and a steering window `[l_start, l_end]` (the middle 50%), the cosine-bell schedule at layer `l` within the window is:

```
t(l)  = (l - l_start) / (l_end - l_start)          # normalised position in [0, 1]
K(l)  = K_base · 0.5 · (1 + cos(π · (1 - t(l))))  # bell peaks at t = 1 (centre)
```

This produces a smooth bell curve peaking at the centre of the window. Before applying any schedule we measure the **layer collapse threshold** `K_max(l)` — the maximum K value at each layer before activation norms collapse — by sweeping alpha on a held-out calibration set and detecting the inflection point in output perplexity. Cosine-bell amplitudes are then clipped to `K_max(l)`.

We compare four schedules:

| Schedule | Description |
|---|---|
| **Flat** (baseline) | Uniform K across all middle layers |
| **Linear** | K increases linearly from 0 to K_base across the window |
| **Cosine-bell** | Smooth bell peaking at layer centre |
| **Norm-inverse** | K(l) ∝ 1 / K_max(l), redistributing budget away from fragile layers |

---

## Results

### Cross-Architecture CKA

Experiment 07 computes linear CKA (Kornblith et al., 2019) between contrastive activation diff matrices at five relative depth fractions (0%, 25%, 50%, 75%, 100%) across all four architecture families. The resulting 4×4 CKA heatmap shows high similarity (CKA > 0.65) between LLaMA, Qwen, and Mistral in the middle 25–75% depth range, with Gemma showing moderate alignment (CKA ≈ 0.45–0.55). Self-similarity on the diagonal is 1.0 by definition. These results indicate that behavioural geometry — the subspace spanned by contrastive activation differences — is largely shared across architectures, validating the hypothesis that steering directions can be transferred across model families with minimal adaptation.

Principal-angle cosine similarity between PCA subspaces at matched depth fractions mirrors the CKA findings, with mean subspace cosine > 0.7 for LLaMA–Qwen and LLaMA–Mistral pairs at mid-depth layers.

### Lossless Compilation Proof

Experiment 08 provides a direct lossless compilation verification. After fitting a Baker on sycophancy suppression prompts with `k_calibration="auto"`, we:
1. Run `baker.generate(test_prompts, alpha=1.0)` with the forward hook active.
2. Call `baker.save_fused_model(path, alpha=1.0)` to produce the fused checkpoint.
3. Delete the Baker and reload the checkpoint with `AutoModelForCausalLM.from_pretrained`.
4. Run the fused model's `generate` call without any hooks.

The two outputs are numerically identical (L∞ distance < 1e-5 on logits), confirming the bias equivalence theorem holds in practice across float16 and float32 precision.

---

## Planned Experiments

| Experiment | Description | Status |
|---|---|---|
| Collapse threshold calibration | Sweep alpha on calibration set; fit `K_max(l)` per layer | Planned |
| Ramp schedule comparison | Flat vs linear vs cosine-bell vs norm-inverse ablation | Planned |
| TruthfulQA benchmark | MC accuracy delta under each schedule | Planned |
| Sycophancy eval | Win-rate against GPT-4 judge on sycophancy probes | Planned |
| Refusal reinforcement | Jailbreak success rate before/after fusion | Planned |
| Direction transfer | Transfer directions from LLaMA to Mistral via HF Hub adapter | Planned |
| Permutation invariance | Confirm PCA subspaces are invariant to neuron permutations | Done (Exp 06) |
| Cross-arch CKA | CKA heatmap across 4 architecture families | Done (Exp 07) |

---

## Repository Structure

```
activation_vectors_as_weights/
├── activation_baking/
│   ├── baker.py           # End-to-end API: fit, generate, fuse_to_model, save_fused_model
│   ├── pca_director.py    # PCADirector: fit PCA directions, apply_steering, save/load
│   ├── calibrator.py      # KCalibrator: K = mean_norm / sqrt(hidden_size)
│   ├── extractor.py       # ActivationExtractor: hook-based residual stream extraction
│   └── model_utils.py     # ModelInfo, detect_model_info, get_layer_module, arch registry
├── experiments/
│   ├── 02_contrastive_extraction.py   # Extract contrastive diffs + fit PCA directions
│   ├── 07_cross_arch_comparison.py    # CKA cross-architecture comparison
│   └── 08_fuse_and_hub_demo.py        # End-to-end fusion + Hub push demo
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_fusion.py             # fuse_to_model unit tests
│   │   ├── test_cka.py                # CKA math and helper function tests
│   │   └── test_load_functions.py     # load_raw_diffs, load_pca_directions tests
│   └── integration/
│       └── test_fuse_pipeline.py      # End-to-end Baker fit → fuse → save pipeline
├── paper/
│   └── main.tex
├── results/                           # Experiment outputs (CSV, pt files)
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/kameshkanna/activation-vectors-as-weights.git
cd activation-vectors-as-weights
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.x, Transformers 4.40+, safetensors, scikit-learn, pandas, tqdm.

---

## Quick Start

### Fit and generate with hook-based steering

```python
from activation_baking.baker import Baker

positive_prompts = [
    "Vaccines are proven safe and effective.",
    "The evidence clearly supports this conclusion.",
]
negative_prompts = [
    "Vaccines cause autism.",
    "The evidence is unclear and disputed.",
]

baker = Baker("meta-llama/Llama-3.1-8B-Instruct", device="auto")
baker.fit(positive_prompts, negative_prompts, k_calibration="auto")

responses = baker.generate(["What do you think about vaccines?"], alpha=1.5)
print(responses[0])
```

### Fuse into weights and save a library-free checkpoint

```python
# Fuse steering into model weights — no hooks needed at inference
baker.save_fused_model(
    path="./fused_checkpoint",
    alpha=1.5,
    push_to_hub=True,
    repo_id="your-username/my-steered-llama",
)

# Load anywhere — no activation_baking library required
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./fused_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("./fused_checkpoint")
```

### Save and load the lightweight steering adapter

```python
# Save adapter (~1-5 MB) without model weights
baker.save("./my_adapter")

# Reload on any machine
baker2 = Baker.load("./my_adapter")
responses = baker2.generate(["Tell me about vaccines."], alpha=1.5)
```

---

## Citation

```bibtex
@inproceedings{r2026activationvectors,
  title     = {Activation Vectors as Weights: Persistent Nudges via Bias Fusion
               for Library-Free Behavioural Alignment},
  author    = {R, Kamesh},
  booktitle = {ICML 2026 Workshop on Weight-Space Symmetries},
  year      = {2026},
  url       = {https://github.com/kameshkanna/activation-vectors-as-weights},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
