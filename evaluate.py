"""
Evaluate base vs. fine-tuned Llama 3 models on perplexity and BLEU.

Usage:
    python evaluate.py
"""

import math
import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from unsloth import FastLanguageModel
from config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    DTYPE,
    LOAD_IN_4BIT,
    OUTPUT_DIR,
    EVAL_PROMPTS,
)


def download_nltk_data():
    """Download required NLTK tokeniser data."""
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)


def load_base_model():
    """Load the original (non-fine-tuned) model."""
    print("[*] Loading base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_finetuned_model():
    """Load the fine-tuned LoRA adapter model."""
    print("[*] Loading fine-tuned model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=OUTPUT_DIR,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of the model on the given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss.item()
    return math.exp(loss)


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the model given a prompt."""
    formatted = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{prompt}\n\n### Response:\n"
    )
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
    # Decode only the new tokens
    response = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    return response.strip()


def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute BLEU score between reference and hypothesis strings."""
    ref_tokens = nltk.word_tokenize(reference.lower())
    hyp_tokens = nltk.word_tokenize(hypothesis.lower())
    smoothie = SmoothingFunction().method1
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)


def evaluate():
    """Run full evaluation comparing base vs. fine-tuned model."""
    download_nltk_data()

    # Load both models
    base_model, base_tokenizer = load_base_model()
    ft_model, ft_tokenizer = load_finetuned_model()

    print("\n" + "=" * 70)
    print("  EVALUATION: Base vs. Fine-Tuned Llama 3")
    print("=" * 70)

    base_perplexities = []
    ft_perplexities = []
    bleu_scores = []

    for i, prompt in enumerate(EVAL_PROMPTS, 1):
        print(f"\n{'─' * 60}")
        print(f"  Prompt {i}: {prompt[:60]}...")
        print(f"{'─' * 60}")

        # Generate responses
        base_response = generate_response(base_model, base_tokenizer, prompt)
        ft_response = generate_response(ft_model, ft_tokenizer, prompt)

        # Compute perplexity on the prompt
        base_ppl = compute_perplexity(base_model, base_tokenizer, prompt)
        ft_ppl = compute_perplexity(ft_model, ft_tokenizer, prompt)
        base_perplexities.append(base_ppl)
        ft_perplexities.append(ft_ppl)

        # Compute BLEU (using base response as reference)
        bleu = compute_bleu(base_response, ft_response)
        bleu_scores.append(bleu)

        print(f"  Base PPL       : {base_ppl:.2f}")
        print(f"  Fine-Tuned PPL : {ft_ppl:.2f}")
        print(f"  BLEU Score     : {bleu:.4f}")
        print(f"\n  [Base]       {base_response[:120]}...")
        print(f"  [Fine-Tuned] {ft_response[:120]}...")

    # Summary
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    avg_base_ppl = sum(base_perplexities) / len(base_perplexities)
    avg_ft_ppl = sum(ft_perplexities) / len(ft_perplexities)
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    print(f"  Avg Base Perplexity       : {avg_base_ppl:.2f}")
    print(f"  Avg Fine-Tuned Perplexity : {avg_ft_ppl:.2f}")
    print(f"  Perplexity Improvement    : {((avg_base_ppl - avg_ft_ppl) / avg_base_ppl * 100):.1f}%")
    print(f"  Avg BLEU Score            : {avg_bleu:.4f}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    evaluate()
