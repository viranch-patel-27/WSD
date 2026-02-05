"""
Configuration for Hybrid WSD Model Training
============================================
Centralized configuration for training, inference, and Wikipedia integration.
"""

import os
import torch

# ============== DEVICE SETUP ==============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============== TRAINING CONFIG ==============
TRAINING_CONFIG = {
    # Data
    "max_instances": 50000,          # Training instances from SemCor
    "train_split": 0.9,              # 90% train, 10% validation
    "max_negatives": 4,              # Negative samples per positive
    
    # Model
    "base_model": "bert-base-uncased",
    "max_length": 128,               # Max sequence length
    
    # Training
    "epochs": 5,
    "batch_size": 32,                # Good for 8GB GPU (RTX 3070)
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    
    # Checkpointing
    "save_dir": "wsd_hybrid_wikipedia_v1",
    "checkpoint_dir": "checkpoints",
    "save_best_only": False,         # Save every epoch
    
    # Mixed precision
    "use_fp16": True,
}

# ============== HYBRID SCORING WEIGHTS ==============
# These weights control the contribution of each component
# during inference: final_score = α*BERT + β*WordNet + γ*Wikipedia
SCORING_WEIGHTS = {
    "bert": 0.5,        # α - Neural contextual understanding
    "wordnet": 0.3,     # β - Lexical knowledge (definitions, relations)
    "wikipedia": 0.2,   # γ - Encyclopedic knowledge (real-world context)
}

# Number of top candidates to consider
TOP_K_CANDIDATES = 6

# ============== WIKIPEDIA CONFIG ==============
WIKIPEDIA_CONFIG = {
    "cache_dir": "./wiki_cache",
    "max_summary_sentences": 3,
    "max_summary_chars": 200,        # Max chars to include in training
    "timeout": 5,                    # API timeout in seconds
    "prefetch": True,                # Pre-fetch Wikipedia during data loading
}

# ============== MODEL PATHS ==============
MODEL_PATHS = {
    "baseline": "wsd_hybrid_bert_wordnet_v2",       # Current model
    "wikipedia_hybrid": "wsd_hybrid_wikipedia_v1",  # New model
}

# ============== INFERENCE CONFIG ==============
INFERENCE_CONFIG = {
    "model_dir": MODEL_PATHS["wikipedia_hybrid"],
    "use_wikipedia": True,
    "use_wordnet": True,
    "top_k": TOP_K_CANDIDATES,
}


def get_checkpoint_path(epoch: int) -> str:
    """Get path for epoch checkpoint."""
    checkpoint_dir = os.path.join(
        TRAINING_CONFIG["save_dir"], 
        TRAINING_CONFIG["checkpoint_dir"]
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")


def get_best_model_path() -> str:
    """Get path for best model checkpoint."""
    checkpoint_dir = os.path.join(
        TRAINING_CONFIG["save_dir"], 
        TRAINING_CONFIG["checkpoint_dir"]
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    return os.path.join(checkpoint_dir, "best.pt")


def print_config():
    """Print current configuration."""
    print("=" * 60)
    print("HYBRID WSD MODEL CONFIGURATION")
    print("=" * 60)
    print(f"\nDevice: {DEVICE}")
    print(f"GPU: {torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'N/A'}")
    
    print(f"\n{'='*30} TRAINING {'='*30}")
    for key, value in TRAINING_CONFIG.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*30} SCORING {'='*30}")
    for key, value in SCORING_WEIGHTS.items():
        print(f"  {key}: {value}")
    
    print(f"\n{'='*30} WIKIPEDIA {'='*30}")
    for key, value in WIKIPEDIA_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
