"""
Prompt Variation Generator

Applies deterministic transformations to an input prompt to create variations.
This avoids needing an external LLM just to generate prompts, ensuring the tool is fast and independent.
"""

def generate_variations(base_prompt: str, num_variations: int = 5) -> list[str]:
    """
    Generates deterministic string-based variations of the input prompt.
    """
    base_prompt = base_prompt.strip()
    
    variations = [base_prompt]  # Ensure the original is always the first one
    
    # 1. Simplify
    variations.append(f"{base_prompt} Explain this in very simple terms.")
    
    # 2. Expand
    variations.append(f"Provide a comprehensive and detailed explanation for the following: {base_prompt}")
    
    # 3. Question format
    variations.append(f"What is the answer to: {base_prompt}?")
    
    # 4. Expert persona
    variations.append(f"Act as an expert. {base_prompt}")

    # 5. Direct
    variations.append(f"Direct answer only: {base_prompt}")

    # Return exactly num_variations (if the user configured fewer/more, we slice or repeat)
    # But since we have 6 including the original, we will return up to `num_variations`.
    # It's better to always return the requested amount padding with numbers if needed.
    
    results = variations[:num_variations]
    
    # If we need more than we have pre-defined
    while len(results) < num_variations:
        results.append(f"Variation {len(results)}: {base_prompt}")
        
    return results
