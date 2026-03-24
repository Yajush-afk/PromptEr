"""
Scorer Module

Analyzes the similarity matrix to generate a consistency score and robustness metrics.
"""

import numpy as np
from config import SIMILARITY_THRESHOLDS

def calculate_scores(similarity_matrix: np.ndarray, variations: list[str]) -> dict:
    """
    Calculates overall consistency score, finds the most divergent pair, and assigns a robustness label.
    """
    n = similarity_matrix.shape[0]
    if n <= 1:
        return {
            "consistency_score": 1.0,
            "divergent_pair": None,
            "robustness_label": "HIGH"
        }

    # Extract upper triangle indices (excluding the diagonal)
    # This prevents counting 1.0 self-similarities and duplicate pairs
    i_upper, j_upper = np.triu_indices(n, k=1)
    
    upper_tri_values = similarity_matrix[i_upper, j_upper]
    
    # 1. Consistency Score (Mean of off-diagonal pairwise similarities)
    consistency_score = float(np.mean(upper_tri_values))
    
    # 2. Most divergent pair (minimum similarity)
    min_idx = np.argmin(upper_tri_values)
    i_div = i_upper[min_idx]
    j_div = j_upper[min_idx]
    
    divergent_pair = {
        "index_1": int(i_div),
        "index_2": int(j_div),
        "prompt_1": variations[i_div],
        "prompt_2": variations[j_div],
        "similarity": float(upper_tri_values[min_idx])
    }
    
    # 3. Robustness Label based on thresholds
    if consistency_score >= SIMILARITY_THRESHOLDS["HIGH"]:
        label = "HIGH"
    elif consistency_score >= SIMILARITY_THRESHOLDS["MEDIUM"]:
        label = "MEDIUM"
    else:
        label = "LOW"
        
    return {
        "consistency_score": consistency_score,
        "divergent_pair": divergent_pair,
        "robustness_label": label
    }
