"""
Fine-tune Llama 3 using Unsloth + QLoRA on instruction-following data.

Usage:
    python fine_tune.py
"""

import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from config import (
    MODEL_NAME,
    MAX_SEQ_LENGTH,
    DTYPE,
    LOAD_IN_4BIT,
    LORA_RANK,
    LORA_ALPHA,
    LORA_DROPOUT,
    TARGET_MODULES,
    LEARNING_RATE,
    NUM_EPOCHS,
    BATCH_SIZE,
    GRADIENT_ACCUMULATION_STEPS,
    WARMUP_STEPS,
    MAX_STEPS,
    WEIGHT_DECAY,
    LOGGING_STEPS,
    SAVE_STEPS,
    DATASET_NAME,
    DATASET_SPLIT,
    OUTPUT_DIR,
    SEED,
)


def format_alpaca_prompt(example: dict) -> dict:
    """Format a dataset example into the Alpaca instruction template."""
    ALPACA_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

    text = ALPACA_TEMPLATE.format(
        instruction=example.get("instruction", ""),
        input=example.get("input", ""),
        output=example.get("output", ""),
    )
    return {"text": text}


def load_model():
    """Load the base model with 4-bit quantisation via Unsloth."""
    print(f"[*] Loading model: {MODEL_NAME}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    return model, tokenizer


def apply_lora(model):
    """Apply QLoRA adapters to the model."""
    print(f"[*] Applying QLoRA — rank={LORA_RANK}, alpha={LORA_ALPHA}")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )
    return model


def load_data():
    """Load and format the Alpaca instruction dataset."""
    print(f"[*] Loading dataset: {DATASET_NAME}")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
    dataset = dataset.map(format_alpaca_prompt)
    print(f"[*] Dataset size: {len(dataset):,} examples")
    return dataset


def train(model, tokenizer, dataset):
    """Run supervised fine-tuning with SFTTrainer."""
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        seed=SEED,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=training_args,
    )

    print("[*] Starting fine-tuning...")
    trainer_stats = trainer.train()

    print(f"[✓] Training complete!")
    print(f"    ├── Training loss : {trainer_stats.training_loss:.4f}")
    print(f"    ├── Runtime       : {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"    └── Samples/sec   : {trainer_stats.metrics['train_samples_per_second']:.1f}")

    return trainer


def save_model(model, tokenizer):
    """Save the fine-tuned LoRA adapter weights."""
    print(f"[*] Saving adapter weights to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("[✓] Model saved successfully!")


def main():
    # Load base model
    model, tokenizer = load_model()

    # Apply QLoRA
    model = apply_lora(model)

    # Print trainable parameter stats
    trainable, total = model.get_nb_trainable_parameters()
    print(f"[*] Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # Load dataset
    dataset = load_data()

    # Fine-tune
    trainer = train(model, tokenizer, dataset)

    # Save
    save_model(model, tokenizer)


if __name__ == "__main__":
    main()
