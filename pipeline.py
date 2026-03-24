from variation_generator import generate_variations
from models.generator import TextGenerator
from models.embedder import ResponseEmbedder
from similarity import compute_similarity_matrix
from scorer import calculate_scores
from config import GENERATION_PARAMS

def run_evaluation(
    base_prompt: str,
    generator_model: TextGenerator,
    embedder_model: ResponseEmbedder,
    num_variations: int = 5
) -> dict:
    
    # 1. Generate variations
    variations = generate_variations(base_prompt, num_variations)
    
    # 2. Get text responses
    responses = []
    for var in variations:
        resp = generator_model.generate(var["prompt"], **GENERATION_PARAMS)
        responses.append(resp)
        
    # 3. Embed responses
    embeddings = embedder_model.embed(responses)
    
    # 4. Compute similarity matrix
    similarity_matrix = compute_similarity_matrix(embeddings)
    
    # 5. Calculate final consistency scores & find divergent pair
    scores = calculate_scores(similarity_matrix, variations)
    
    return {
        "base_prompt": base_prompt,
        "variations": variations,
        "responses": responses,
        "similarity_matrix": similarity_matrix,
        "scores": scores
    }
