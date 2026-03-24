import numpy as np
from sentence_transformers import SentenceTransformer

class ResponseEmbedder:

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(texts)
        return embeddings