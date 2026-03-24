import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) == 0:
        return np.array([])
    matrix = cosine_similarity(embeddings)
    return matrix