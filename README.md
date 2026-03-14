# 🧠 LLM Fine-Tuning

Fine-tuning Llama 3 (8B) on instruction-following data using [Unsloth](https://github.com/unslothai/unsloth) and QLoRA. The setup cuts training time roughly in half and drops GPU memory usage by ~60% compared to full fine-tuning, making it feasible to run on a single consumer GPU.

## 💡 What this does

- Loads Llama 3 in 4-bit quantisation via Unsloth
- Applies QLoRA adapters (rank-16) to attention + MLP layers
- Trains on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) instruction dataset using SFTTrainer
- Evaluates base vs. fine-tuned model on **perplexity** and **BLEU** to quantify the improvement

## 📁 Project structure

```
├── config.py           # All hyperparameters and paths
├── fine_tune.py        # Training pipeline (load → LoRA → train → save)
├── evaluate.py         # Base vs. fine-tuned comparison (perplexity, BLEU)
├── requirements.txt
└── .gitignore
```

## 🚀 Quick start

```bash
pip install -r requirements.txt

# Train
python fine_tune.py

# Evaluate
python evaluate.py
```

> ⚡ Requires a CUDA GPU. Tested on a T4 (16 GB) — runs fine with 4-bit quantisation.

## 📊 Results

The fine-tuned model shows consistent gains over the base model across evaluation prompts:

| Metric | Base | Fine-Tuned |
|--------|------|------------|
| Perplexity | Higher | Lower (better) |
| BLEU | — | Improved |

Training completes in ~30 min on a T4 with `max_steps=500`.

## 🔧 Key dependencies

- `unsloth` — 2x faster LoRA fine-tuning
- `trl` — SFTTrainer for instruction tuning
- `transformers`, `torch`, `datasets`
- `nltk` — BLEU scoring during evaluation
