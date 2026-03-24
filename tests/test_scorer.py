import pytest
import numpy as np
from scorer import calculate_scores

def test_scorer_perfect_consistency():
    # 3 variations, perfect similarity (1.0 everywhere)
    matrix = np.array([
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]
    ])
    variations = [{"prompt": "A"}, {"prompt": "B"}, {"prompt": "C"}]
    
    results = calculate_scores(matrix, variations)
    
    assert results["consistency_score"] == 1.0
    assert results["robustness_label"] == "HIGH"
    
def test_scorer_zero_consistency():
    # 3 variations, complete divergence (0.0 off-diagonal)
    matrix = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    variations = [{"prompt": "A"}, {"prompt": "B"}, {"prompt": "C"}]
    
    results = calculate_scores(matrix, variations)
    
    assert results["consistency_score"] == 0.0
    assert results["robustness_label"] == "LOW"
    assert results["divergent_pair"]["similarity"] == 0.0

def test_scorer_single_variation():
    # Scorer should handle edge case of 1 variation safely
    matrix = np.array([[1.0]])
    variations = [{"prompt": "Only me"}]
    
    results = calculate_scores(matrix, variations)
    
    assert results["consistency_score"] == 1.0
    assert results["divergent_pair"] is None
    assert results["robustness_label"] == "HIGH"
