"""
Similarity Computation

Calculates pairwise cosine similarity between dense vector embeddings.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Given an array of embeddings of shape (N, D), computes the (N, N) cosine similarity matrix.

    Args:
        embeddings: A 2D numpy array where each row is an embedding vector.
        
    Returns:
        A 2D numpy array containing the pairwise cosine similarities.
    """
    if len(embeddings) == 0:
        return np.array([])
        
    # cosine_similarity expects 2D array, it returns values from -1 to 1
    matrix = cosine_similarity(embeddings)
    return matrix
