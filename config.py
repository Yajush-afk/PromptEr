"""
PromptEr Configuration Module
Contains centralized constants, model names, and default parameters for generation and similarity scoring.
"""

DEFAULT_GENERATION_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GENERATION_PARAMS = {
    "max_new_tokens": 150,
    "temperature": 0.7,
    "do_sample": True,
    "top_p": 0.9
}

DEFAULT_NUM_VARIATIONS = 5
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLDS = {
    "HIGH": 0.85,
    "MEDIUM": 0.70,
    "LOW": 0.0
}
