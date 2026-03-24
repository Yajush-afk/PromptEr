# PromptEr

PromptEr is a simple evaluation tool to analyze the robustness of LLM prompts. By generating semantic variations of an initial prompt, running them through a Hugging Face text-generation model, and comparing the results via a sentence-transformer embedding model, PromptEr visualizes how sensitive a model is to minor prompting changes.

## Features
- **Prompt Variations:** Automatically generate N variations of an input prompt.
- **Text Generation:** (To be implemented via HF) Get responses for each variation.
- **Embedding Analysis:** (To be implemented via HF) Embed responses into vectors.
- **Consistency Scoring:** Calculate cosine similarity to derive a prompt consistency score.
- **Streamlit UI:** An interactive interface to visualize the results with similarity heatmaps.

## Setup Instructions

1. Ensure you have activated your conda environment:
   ```bash
   conda activate env
   ```

2. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit UI locally:
   ```bash
   streamlit run app.py
   ```

## Learning HF Models 
HF specific codes have been stubbed out as pseudocode. A separate `learning_hints.md` file (ignored by git) contains clues and reference code for you to practice text-generation and embedding using Hugging Face's libraries.
