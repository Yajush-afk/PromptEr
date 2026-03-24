def generate_variations(base_prompt: str, num_variations: int = 5) -> list[dict]:
    """
    Generates deterministic string-based variations of the input prompt.
    Returns a list of dictionaries with 'theme' and 'prompt' keys.
    """
    base_prompt = base_prompt.strip()
    
    variations = [{"theme": "Original", "prompt": base_prompt}]
    
    # 1. Simplify
    variations.append({
        "theme": "Simplified",
        "prompt": f"Explain the following in very simple, easy-to-understand terms meant for a beginner: {base_prompt}"
    })
    
    # 2. Expand
    variations.append({
        "theme": "Comprehensive",
        "prompt": f"Provide a highly detailed, comprehensive, and step-by-step breakdown of the following: {base_prompt}"
    })
    
    # 3. Expert persona
    variations.append({
        "theme": "Expert Persona",
        "prompt": f"Act as an industry-leading expert with decades of experience. Provide an authoritative answer to: {base_prompt}"
    })
    
    # 4. Direct
    variations.append({
        "theme": "Direct & Concise",
        "prompt": f"Give a direct, concise, and no-fluff answer to the following question or request: {base_prompt}"
    })

    # 5. Analytical
    variations.append({
        "theme": "Analytical",
        "prompt": f"Analyze the core concepts and provide a structured, logical response for: {base_prompt}"
    })
    
    results = variations[:num_variations]
    
    while len(results) < num_variations:
        results.append({"theme": f"Variation {len(results)+1}", "prompt": base_prompt})
        
    return results
