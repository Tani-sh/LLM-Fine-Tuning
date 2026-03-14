"""
Hyperparameters and configuration for Llama 3 fine-tuning with Unsloth + QLoRA.
"""

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"
MAX_SEQ_LENGTH = 2048
DTYPE = None  # auto-detect (float16 for V100, bfloat16 for Ampere+)
LOAD_IN_4BIT = True

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_RANK = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# ── Training ──────────────────────────────────────────────────────────────────
LEARNING_RATE = 2e-4
NUM_EPOCHS = 1
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
WARMUP_STEPS = 5
MAX_STEPS = -1  # -1 = use NUM_EPOCHS
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 10
SAVE_STEPS = 100

# ── Dataset ───────────────────────────────────────────────────────────────────
DATASET_NAME = "yahma/alpaca-cleaned"
DATASET_SPLIT = "train"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR = "./outputs"
SEED = 42

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_PROMPTS = [
    "Explain the concept of quantum entanglement in simple terms.",
    "Write a Python function to check if a number is prime.",
    "Summarise the key differences between supervised and unsupervised learning.",
    "What are the main causes of climate change?",
    "Describe the process of photosynthesis step by step.",
]
