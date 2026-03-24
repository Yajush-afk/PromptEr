import pytest
from variation_generator import generate_variations

def test_generate_variations_count():
    base_prompt = "What is gravity?"
    num_vars = 5
    vars = generate_variations(base_prompt, num_vars)
    assert len(vars) == num_vars

def test_generate_variations_content():
    base_prompt = "Explain quantum computing."
    vars = generate_variations(base_prompt, 5)
    
    # The original should be the first entry
    assert vars[0]["prompt"] == base_prompt
    
    # All others should be strings containing the base prompt (in our current simplistic logic)
    for var in vars[1:]:
        assert isinstance(var["prompt"], str)
        assert len(var["prompt"]) > len(base_prompt)
        assert base_prompt in var["prompt"]
