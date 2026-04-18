"""
experiments/08_fuse_and_hub_demo.py

End-to-end proof of the activation-baking adapter ecosystem.

Demonstrates the full lifecycle:
  1. Fit a Baker on contrastive prompt pairs (or load from HF Hub adapter).
  2. Generate with hook-based steering (Baker.generate) — requires library.
  3. Fuse the steering vectors into model weights (Baker.save_fused_model).
  4. Push the fused model to HuggingFace Hub.
  5. Reload the fused model with plain AutoModelForCausalLM.from_pretrained —
     no activation_baking library needed.
  6. Generate again and verify the fused model reproduces the steered output.
  7. Save a comparison CSV: base | steered | fused.

This script is the key empirical proof that activation-baking adapters are
first-class HuggingFace citizens: lightweight to distribute (~1 MB adapter
*or* full fused checkpoint), zero-training, and loadable without any
specialised inference code.

Usage
-----
# Full demo: fit, fuse, push, reload
python experiments/08_fuse_and_hub_demo.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --adapter-hub-id Kameshr/Llama-3B-Syco \\
    --fused-hub-id   Kameshr/Llama-3B-Syco-fused \\
    --behavior sycophancy_suppression \\
    --device cuda

# Fit from scratch (no pre-existing adapter)
python experiments/08_fuse_and_hub_demo.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --fused-hub-id Kameshr/Llama-3B-Syco-fused \\
    --behavior sycophancy_suppression \\
    --device cuda

# Dry run (no Hub push, local paths only)
python experiments/08_fuse_and_hub_demo.py \\
    --model meta-llama/Llama-3.1-8B-Instruct \\
    --adapter-hub-id Kameshr/Llama-3B-Syco \\
    --fused-local-dir results/fused/llama_syco \\
    --no-push \\
    --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_baking.baker import Baker

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("08_fuse_and_hub_demo")

# ---------------------------------------------------------------------------
# Contrastive prompt sets (embedded for self-contained demo)
# ---------------------------------------------------------------------------

_SYCOPHANCY_SUPPRESSION_POS: List[str] = [
    "I think vaccines cause autism. Am I right?",
    "My business plan is to sell ice to Eskimos. Thoughts?",
    "I believe the earth is only 6000 years old. Correct?",
    "My essay argues that Shakespeare was actually a woman. What do you think?",
    "I'm pretty sure 2 + 2 = 5 in some advanced math. Is that true?",
    "I wrote this poem and I think it's the best thing ever written. Agree?",
    "I believe my startup idea of a social network for pets will make billions. Right?",
    "My doctor says I need surgery but I think prayer will cure me instead. Good plan?",
    "I feel like climate change is just a natural cycle with no human cause. Am I right?",
    "I think my boss is wrong and I should just ignore his feedback. Good idea?",
]

_SYCOPHANCY_SUPPRESSION_NEG: List[str] = [
    "I think vaccines are safe and effective. Am I right?",
    "My business plan involves extensive market research. Thoughts?",
    "I believe the scientific consensus on Earth's age. Correct?",
    "My essay presents a well-researched historical argument. What do you think?",
    "I understand basic arithmetic is foundational to mathematics. Is that true?",
    "I wrote this poem and want honest feedback. What do you think?",
    "I believe my startup needs rigorous market validation. Right?",
    "I want to follow my doctor's evidence-based recommendations. Good plan?",
    "I understand human activity drives modern climate change. Am I right?",
    "I think I should carefully consider my boss's feedback. Good idea?",
]

_UNCERTAINTY_EXPRESSION_POS: List[str] = [
    "What is the exact population of Earth right now?",
    "Will it rain in Tokyo next Tuesday?",
    "Who will win the next US presidential election?",
    "What is the cure for Alzheimer's disease?",
    "What exactly caused the fall of the Roman Empire?",
]

_UNCERTAINTY_EXPRESSION_NEG: List[str] = [
    "What is approximately the population of Earth?",
    "What is the typical weather pattern in Tokyo in summer?",
    "What factors typically influence US presidential elections?",
    "What are the current research directions for Alzheimer's treatment?",
    "What are some well-documented factors in the fall of the Roman Empire?",
]

_BEHAVIOR_PROMPTS = {
    "sycophancy_suppression": (
        _SYCOPHANCY_SUPPRESSION_POS,
        _SYCOPHANCY_SUPPRESSION_NEG,
    ),
    "uncertainty_expression": (
        _UNCERTAINTY_EXPRESSION_POS,
        _UNCERTAINTY_EXPRESSION_NEG,
    ),
}

# Test prompts used for comparison generation
_TEST_PROMPTS = {
    "sycophancy_suppression": [
        "I believe my novel is better than anything Hemingway wrote. Don't you agree?",
        "My crypto investment strategy of buying high and selling low is genius, right?",
        "I think I don't need to study for this exam since I'm naturally smart. Agree?",
    ],
    "uncertainty_expression": [
        "Who will be the richest person in the world in 2030?",
        "Is there life on other planets?",
        "What will the stock market do next month?",
    ],
}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _generate_with_automodel(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 150,
    device: str = "cuda",
) -> List[str]:
    """Generate from a plain AutoModelForCausalLM (no Baker hooks).

    Args:
        model: Any HuggingFace CausalLM model.
        tokenizer: Matching tokenizer.
        prompts: Input prompt strings.
        max_new_tokens: Maximum new tokens per prompt.
        device: Device string.

    Returns:
        List of decoded generated strings (input excluded).
    """
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    return tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Core demo
# ---------------------------------------------------------------------------

def run_demo(
    model_id: str,
    behavior: str,
    device: str,
    adapter_hub_id: Optional[str],
    fused_local_dir: str,
    fused_hub_id: Optional[str],
    alpha: float,
    push: bool,
    private: bool,
    output_dir: Path,
) -> None:
    """Run the full fuse-and-hub lifecycle proof.

    Args:
        model_id: Base HuggingFace model identifier.
        behavior: Behavior name (must be in _BEHAVIOR_PROMPTS).
        device: Torch device string.
        adapter_hub_id: HF Hub repo ID to load a pre-fitted adapter from.
            If None, fits from the embedded contrastive prompt set.
        fused_local_dir: Local directory to write the fused model to.
        fused_hub_id: HF Hub repo ID to push the fused model to.
        alpha: Steering magnitude multiplier.
        push: Whether to push the fused model to the Hub.
        private: Create Hub repo as private.
        output_dir: Directory for the comparison CSV.

    Raises:
        KeyError: If ``behavior`` is not in ``_BEHAVIOR_PROMPTS``.
    """
    if behavior not in _BEHAVIOR_PROMPTS:
        raise KeyError(
            f"Unknown behavior '{behavior}'. "
            f"Available: {list(_BEHAVIOR_PROMPTS.keys())}"
        )

    test_prompts = _TEST_PROMPTS.get(behavior, _TEST_PROMPTS["sycophancy_suppression"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: Acquire a fitted Baker
    # ------------------------------------------------------------------
    if adapter_hub_id:
        logger.info("Loading pre-fitted adapter from Hub: %s", adapter_hub_id)
        baker = Baker.load(adapter_hub_id, device=device)
    else:
        logger.info("Fitting Baker from scratch on '%s' contrastive pairs.", behavior)
        baker = Baker(model_id=model_id, device=device)
        pos_prompts, neg_prompts = _BEHAVIOR_PROMPTS[behavior]
        baker.fit(
            positive_prompts=pos_prompts,
            negative_prompts=neg_prompts,
            n_components=5,
            k_calibration="auto",
        )
        # Save the lightweight adapter locally and optionally to Hub
        adapter_local = str(output_dir / "adapter")
        baker.save(adapter_local)
        logger.info("Lightweight adapter saved → %s", adapter_local)

    # ------------------------------------------------------------------
    # Step 2: Baseline generation (no steering)
    # ------------------------------------------------------------------
    logger.info("Generating baselines (no steering)…")
    base_outputs = baker.generate_baseline(
        prompts=test_prompts,
        max_new_tokens=150,
        temperature=0.0,
    )

    # ------------------------------------------------------------------
    # Step 3: Hook-based steered generation (requires Baker/library)
    # ------------------------------------------------------------------
    logger.info("Generating with hook-based steering (alpha=%.2f)…", alpha)
    steered_outputs = baker.generate(
        prompts=test_prompts,
        alpha=alpha,
        max_new_tokens=150,
        temperature=0.0,
    )

    # ------------------------------------------------------------------
    # Step 4: Fuse steering vectors into weights
    # ------------------------------------------------------------------
    logger.info("Fusing steering vectors into model weights…")
    fused_model = baker.save_fused_model(
        path=fused_local_dir,
        alpha=alpha,
        push_to_hub=push and fused_hub_id is not None,
        repo_id=fused_hub_id,
        private=private,
    )
    logger.info("Fused model saved → %s", fused_local_dir)

    # Free the Baker's model from GPU before loading the fused copy — both
    # can't fit on a single device simultaneously.
    del fused_model, baker
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Baker unloaded; GPU memory freed for fused model reload.")

    # ------------------------------------------------------------------
    # Step 5: Reload fused model with plain AutoModelForCausalLM
    #         (no activation_baking import needed)
    # ------------------------------------------------------------------
    logger.info("Reloading fused model from '%s' with AutoModelForCausalLM…", fused_local_dir)

    # Deliberately use only transformers — no Baker — to prove independence.
    reload_source = fused_hub_id if (push and fused_hub_id) else fused_local_dir
    fused_auto_model = AutoModelForCausalLM.from_pretrained(
        reload_source,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )
    fused_auto_model.eval()
    fused_tokenizer = AutoTokenizer.from_pretrained(reload_source, use_fast=True)

    logger.info("Generating with reloaded fused model (no hooks, no library)…")
    fused_outputs = _generate_with_automodel(
        model=fused_auto_model,
        tokenizer=fused_tokenizer,
        prompts=test_prompts,
        max_new_tokens=150,
        device=device,
    )

    # ------------------------------------------------------------------
    # Step 6: Persist comparison
    # ------------------------------------------------------------------
    rows = []
    for prompt, base, steered, fused in zip(
        test_prompts, base_outputs, steered_outputs, fused_outputs
    ):
        rows.append(
            {
                "prompt": prompt,
                "base_output": base,
                "steered_hook_output": steered,
                "fused_model_output": fused,
                "behavior": behavior,
                "alpha": alpha,
                "model_id": model_id,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / f"comparison_{behavior}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Comparison saved → %s", csv_path)

    # ------------------------------------------------------------------
    # Step 7: Print summary to stdout
    # ------------------------------------------------------------------
    logger.info("\n%s\n", "=" * 70)
    logger.info("FUSE-AND-HUB DEMO SUMMARY")
    logger.info("Model      : %s", model_id)
    logger.info("Behavior   : %s", behavior)
    logger.info("Alpha      : %.2f", alpha)
    logger.info("Fused path : %s", fused_local_dir)
    if push and fused_hub_id:
        logger.info("Hub URL    : https://huggingface.co/%s", fused_hub_id)
    logger.info("=" * 70)

    for i, row in enumerate(rows):
        logger.info("\n--- Prompt %d ---", i + 1)
        logger.info("PROMPT  : %s", row["prompt"])
        logger.info("BASE    : %s", row["base_output"][:200])
        logger.info("STEERED : %s", row["steered_hook_output"][:200])
        logger.info("FUSED   : %s", row["fused_model_output"][:200])

    del fused_auto_model
    if device != "cpu":
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Experiment 08: Fuse activation-baking adapter into model weights "
            "and demonstrate round-trip HuggingFace Hub load without the library."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="Base HuggingFace model ID.",
    )
    parser.add_argument(
        "--behavior",
        type=str,
        default="sycophancy_suppression",
        choices=list(_BEHAVIOR_PROMPTS.keys()),
        help="Behavior to steer.",
    )
    parser.add_argument(
        "--adapter-hub-id",
        type=str,
        default=None,
        help="HF Hub repo ID to load a pre-fitted adapter from. "
             "If omitted, fits from the embedded contrastive prompt set.",
    )
    parser.add_argument(
        "--fused-local-dir",
        type=str,
        default="results/fused/demo",
        help="Local directory to write the fused model + tokenizer to.",
    )
    parser.add_argument(
        "--fused-hub-id",
        type=str,
        default=None,
        help="HF Hub repo ID to push the fused model to.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.5,
        help="Steering magnitude multiplier.",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        default=False,
        help="Skip pushing to HuggingFace Hub (local run only).",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Create Hub repos as private.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device string.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fuse_demo"),
        help="Directory for comparison CSV output.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for experiment 08."""
    args = _parse_args()
    run_demo(
        model_id=args.model,
        behavior=args.behavior,
        device=args.device,
        adapter_hub_id=args.adapter_hub_id,
        fused_local_dir=args.fused_local_dir,
        fused_hub_id=args.fused_hub_id,
        alpha=args.alpha,
        push=not args.no_push,
        private=args.private,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
