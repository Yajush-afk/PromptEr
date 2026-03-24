"""
PromptEr Configuration Module

Contains centralized constants, model names, and default parameters for generation and similarity scoring.
"""

# ==========================================
# text generation configuration
# ==========================================

# Use a small, CPU-friendly model by default for local testing
DEFAULT_GENERATION_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Text generation parameters
GENERATION_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9
}

# Number of prompt variations to generate default
DEFAULT_NUM_VARIATIONS = 5

# ==========================================
# embedding & similarity configuration
# ==========================================

# A fast, lightweight embedding model for CPU inference
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Similarity Thresholds (Cosine similarity is between -1 and 1)
# These represent the boundaries for "Robustness" labels.
SIMILARITY_THRESHOLDS = {
    "HIGH": 0.85,    # > 0.85 average similarity is considered highly robust
    "MEDIUM": 0.70,  # 0.70 - 0.85 is medium robustness
    "LOW": 0.0       # < 0.70 is low robustness
}
