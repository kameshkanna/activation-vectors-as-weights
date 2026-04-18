# Activation Vectors as Weights: Persistent Nudges via Bias Fusion

Activation steering vectors are mathematically equivalent to a **constant bias addition at `down_proj`** — the final MLP projection into the residual stream. This equivalence enables permanent weight fusion: the steered model becomes a standard HuggingFace checkpoint requiring no hooks or library at inference time.

## Key insight: persistent vs trigger-conditioned steering

Prior work ("Compiling Activation Steering into Weights", arXiv:2604.12359) compiles steering as **trigger-conditioned** weight edits — the behaviour activates only for specific input patterns. Our approach uses **unconditional persistent bias nudges**: the steering is always active, input-independent, and exactly reproduces hook-based steering output (lossless compilation).

For behavioural alignment targets (sycophancy suppression, uncertainty expression), persistent nudges are strictly preferable — the behaviour is desired universally, not conditionally.

## Key results

- Fused model output **exactly matches** hook-steered Baker output (lossless compilation)
- PEFT-compatible adapter format: `config.json` + `directions.safetensors` + `directions_meta.json`
- Cross-architecture CKA shows behavioural directions are structurally similar across Llama, Qwen, Mistral, Gemma

## Structure

```
activation_baking/     core library — includes Baker.fuse_to_model() and Baker.save_fused_model()
experiments/
  07_cross_arch_comparison.py   unbiased CKA cross-architecture analysis
  08_fuse_and_hub_demo.py       end-to-end: fit → hook-steer → fuse → save → reload → verify
results/cross_arch/    CKA matrices and layer-depth similarity CSVs
```

## Usage

```python
from activation_baking.baker import Baker

baker = Baker(model_id="meta-llama/Llama-3.1-8B-Instruct", device="cuda")
baker.fit(positive_prompts, negative_prompts)

# Hook-based (requires library)
steered_outputs = baker.generate(prompts)

# Fuse into weights → standard HF checkpoint, no library needed at inference
baker.save_fused_model("results/fused/demo")

# Reload with plain transformers — no activation_baking import
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("results/fused/demo")
```

## Install

```bash
pip install -r requirements.txt
```
