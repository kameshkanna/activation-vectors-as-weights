# Activation Vectors as Weights: Persistent Nudges via Bias Fusion for Library-Free Behavioural Alignment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Activation steering adds constant vectors to transformer residual streams at inference time — effective for behavioural alignment, but every existing implementation requires custom inference hooks or a separate steering library at runtime. This paper shows that for input-independent steering vectors, the runtime hook is mathematically equivalent to a one-time bias modification, enabling permanent fusion into model weights.

After fusion, the result is a standard HuggingFace checkpoint that loads and runs without any activation_baking library, hooks, or special inference setup.

---

## Method

### Bias Equivalence

Every modern transformer MLP computes `output = down_proj(gate(x) ⊙ up(x))`, where `down_proj` is a standard `nn.Linear` with output `W·z + b`. A forward hook that adds a constant vector `v = alpha · K · direction` at every forward pass produces `W·z + b + v`. Since `v` is input-independent, this is identical to setting `b ← b + v` once at initialization.

The fused model's outputs are numerically identical to the hook-steered model for all inputs and all decoding strategies — a lossless guarantee by construction, not approximation. The only architectural change is enabling `config.mlp_bias = True` and zero-initializing biases on all `down_proj` layers before adding the steering delta.

### Unconditional vs Trigger-Conditioned Fusion

Prior work on compiling activation steering into weights (arXiv:2604.12359, Steer2Edit) uses input-triggered edits that activate only on specific token patterns. This work uses unconditional persistent bias fusion — always active, input-independent, and lossless.

| Property | This work | Compiling AS (arXiv:2604.12359) | Steer2Edit |
|---|---|---|---|
| Conditioning | Unconditional (always on) | Input-triggered | Input-triggered |
| Parameter type | Bias term | Weight matrix delta | Rank-1 weight edit |
| Lossless? | Yes (exact by construction) | Approximate | Approximate |
| Library at inference | None | None | None |

Unconditional fusion is preferable for persistent alignment goals (sycophancy suppression, refusal reinforcement) where the nudge should apply uniformly regardless of phrasing.

### Cosine-Bell Ramp Schedule

Standard steering applies a flat K multiplier across all middle layers. We propose a cosine-bell ramp that concentrates steering energy at the semantically richest layers — empirically the central 50% of the network — while tapering to zero at boundaries:

```
t(l)  = (l - l_start) / (l_end - l_start)
K(l)  = K_base · 0.5 · (1 + cos(π · (1 - t(l))))
```

Before applying any schedule, the layer collapse threshold `K_max(l)` is measured by sweeping alpha on a held-out calibration set and detecting the perplexity inflection point. Cosine-bell amplitudes are clipped to `K_max(l)`.

Four schedules are compared:

| Schedule | Description |
|---|---|
| Flat (baseline) | Uniform K across all middle layers |
| Linear | K increases linearly from 0 to K_base across the window |
| Cosine-bell | Smooth bell peaking at layer centre |
| Norm-inverse | K(l) ∝ 1 / K_max(l), redistributing budget from fragile layers |

---

## Experiments

All experiments are pending a GPU rerun. A prior downstream evaluation run (LLM-as-judge) was found to have a bug and all results from that run were discarded.

| # | Experiment | Status |
|---|---|---|
| 01 | Cross-architecture CKA (4×4 heatmap, 5 depth fractions) | Ready to run |
| 02 | Lossless compilation verification (L∞ < 1e-5 on logits) | Ready to run |
| 03 | Collapse threshold K_max(l) sweep | Planned |
| 04 | Ramp schedule ablation (flat / linear / cosine-bell / norm-inverse) | Planned |
| 05 | TruthfulQA MC accuracy under each schedule | Planned |
| 06 | Sycophancy suppression win-rate (GPT-4o judge) | Planned |
| 07 | Refusal reinforcement — jailbreak success rate | Planned |
| 08 | Direction transfer across architectures via Hub adapter | Planned |

---

## Repository Structure

```
activation_vectors_as_weights/
├── activation_baking/
│   ├── baker.py           # End-to-end API: fit, generate, fuse_to_model, save_fused_model
│   ├── pca_director.py    # PCADirector: fit PCA directions, apply_steering, save/load
│   ├── calibrator.py      # KCalibrator: K = mean_norm / sqrt(hidden_size)
│   ├── extractor.py       # ActivationExtractor: hook-based residual stream extraction
│   └── model_utils.py     # ModelInfo, detect_model_info, arch registry
├── experiments/
│   ├── 01_cross_arch_comparison.py    # CKA cross-architecture comparison
│   └── 02_fuse_and_hub_demo.py        # End-to-end fusion + Hub push demo
├── tests/
│   ├── unit/              # Fusion, CKA math, load-function tests
│   └── integration/       # End-to-end Baker fit → fuse → save pipeline
├── paper/
│   └── main.tex
├── results/
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

**Requirements:** Python 3.10+, PyTorch 2.x, Transformers 4.40+, safetensors, scikit-learn.

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

### Fuse into weights — no library required at inference

```python
baker.save_fused_model(
    path="./fused_checkpoint",
    alpha=1.5,
    push_to_hub=True,
    repo_id="your-username/my-steered-llama",
)

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./fused_checkpoint")
tokenizer = AutoTokenizer.from_pretrained("./fused_checkpoint")
```

### Save and reload the lightweight adapter

```python
baker.save("./my_adapter")

baker2 = Baker.load("./my_adapter")
responses = baker2.generate(["Tell me about vaccines."], alpha=1.5)
```

---

## Citation

```bibtex
@article{r2026activationvectors,
  title   = {Activation Vectors as Weights: Persistent Nudges via Bias Fusion
             for Library-Free Behavioural Alignment},
  author  = {R, Kamesh},
  year    = {2026},
  url     = {https://github.com/kameshkanna/activation-vectors-as-weights},
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
