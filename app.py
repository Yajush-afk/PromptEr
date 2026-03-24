import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_NUM_VARIATIONS,
    GENERATION_PARAMS,
)
from pipeline import run_evaluation
from models.generator import TextGenerator
from models.embedder import ResponseEmbedder

# ─────────────────────────────────────────────────────────────────────────────
# Cached Model Loaders
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Booting up Generation Model...")
def load_generator(model_name):
    return TextGenerator(model_name)

@st.cache_resource(show_spinner="Booting up Embedding Model...")
def load_embedder(model_name):
    return ResponseEmbedder(model_name)

st.set_page_config(
    page_title="PromptEr — Prompt Robustness Analyzer",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Background ─────────────────────────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1220 50%, #0f1525 100%);
    min-height: 100vh;
}

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.03);
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── Score card ──────────────────────────────────────────────────────────── */
.score-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    text-align: center;
    backdrop-filter: blur(12px);
}
.score-number {
    font-size: 3.5rem;
    font-weight: 700;
    line-height: 1;
    background: linear-gradient(135deg, #6366f1, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.score-label {
    font-size: 0.85rem;
    color: rgba(255,255,255,0.5);
    margin-top: 0.4rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

/* ── Robustness badge ────────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 0.35rem 1.1rem;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-HIGH   { background: rgba(34,197,94,0.15);  color: #4ade80; border: 1px solid rgba(34,197,94,0.3); }
.badge-MEDIUM { background: rgba(234,179,8,0.15);  color: #facc15; border: 1px solid rgba(234,179,8,0.3); }
.badge-LOW    { background: rgba(239,68,68,0.15);  color: #f87171; border: 1px solid rgba(239,68,68,0.3); }

/* ── Divergent alert ─────────────────────────────────────────────────────── */
.divergent-box {
    background: rgba(239,68,68,0.07);
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 12px;
    padding: 1rem 1.4rem;
}
.divergent-title {
    color: #f87171;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

/* ── Section title ───────────────────────────────────────────────────────── */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    color: rgba(255,255,255,0.9);
    border-left: 3px solid #6366f1;
    padding-left: 0.6rem;
    margin-bottom: 0.8rem;
}
</style>
""", unsafe_allow_html=True)


with st.sidebar:
    st.markdown("## Configuration")
    st.divider()

    gen_model_name = st.text_input(
        "Generation Model",
        value=DEFAULT_GENERATION_MODEL,
        help="Any HuggingFace instruction-tuned causal LM, e.g. TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    )

    emb_model_name = st.text_input(
        "Embedding Model",
        value=DEFAULT_EMBEDDING_MODEL,
        help="Any sentence-transformers model, e.g. sentence-transformers/all-MiniLM-L6-v2",
    )

    st.divider()
    st.markdown("**Generation Parameters**")

    num_variations = st.slider(
        "Number of Prompt Variations",
        min_value=2,
        max_value=8,
        value=DEFAULT_NUM_VARIATIONS,
    )

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=30,
        max_value=300,
        value=GENERATION_PARAMS["max_new_tokens"],
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.1,
        max_value=1.5,
        value=GENERATION_PARAMS["temperature"],
        step=0.05,
    )

    st.divider()
    with st.expander("❓ Help & Glossary"):
        st.markdown("""
        **Consistency Score**
        The average semantic similarity between all pairs of AI responses. 
        - **1.0**: Perfect consistency (the AI meant the exact same thing every time).
        - **0.0**: Complete divergence (the AI answers were totally unrelated).

        **Robustness Label**
        - **HIGH**: Score > 0.85 (Very stable prompt)
        - **MEDIUM**: Score 0.70 – 0.85 (Somewhat stable prompt)
        - **LOW**: Score < 0.70 (Unstable prompt, highly sensitive to phrasing)

        **Most Divergent Pair**
        Highlights the two prompt variations that caused the AI to give the most mathematically different answers.
        
        **Heatmap**
        A visual grid showing how every single answer compares to every other answer. Darker squares mean lower similarity.
        """)
        
    st.caption("PromptEr v0.1 — Prompt Robustness Analyzer")

st.markdown("""
<div style="padding: 2rem 0 1rem;">
  <h1 style="font-size:2.2rem; font-weight:700; margin:0;
             background:linear-gradient(135deg,#818cf8,#a78bfa,#c4b5fd);
             -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    PromptEr
  </h1>
  <p style="color:rgba(255,255,255,0.45); font-size:1rem; margin:0.3rem 0 0;">
    Measure how sensitive your LLM is to small prompt variations.
  </p>
</div>
""", unsafe_allow_html=True)

base_prompt = st.text_area(
    "Enter your base prompt",
    placeholder="e.g. Explain the theory of relativity",
    height=90,
    label_visibility="collapsed",
)

run_button = st.button("Run Analysis", use_container_width=True, type="primary")

if run_button:
    if not base_prompt.strip():
        st.warning("Please enter a prompt before running analysis.")
        st.stop()

    with st.spinner("Generating variations & running evaluation..."):
        # We load models using cache to prevent slow reloading on every UI interaction.
        generator = load_generator(gen_model_name)
        embedder  = load_embedder(emb_model_name)

        results = run_evaluation(
            base_prompt=base_prompt,
            generator_model=generator,
            embedder_model=embedder,
            num_variations=num_variations,
        )

    scores          = results["scores"]
    variations      = results["variations"]
    responses       = results["responses"]
    sim_matrix      = results["similarity_matrix"]
    consistency     = scores["consistency_score"]
    robustness      = scores["robustness_label"]
    divergent_pair  = scores["divergent_pair"]

    st.divider()

    col_score, col_label, col_div = st.columns([1, 1, 1], gap="medium")

    with col_score:
        st.markdown(f"""
        <div class="score-card">
          <div class="score-number">{consistency:.2f}</div>
          <div class="score-label">Consistency Score</div>
        </div>
        """, unsafe_allow_html=True)

    with col_label:
        st.markdown(f"""
        <div class="score-card">
          <div style="padding-top:0.4rem;">
            <span class="badge badge-{robustness}">{robustness} ROBUSTNESS</span>
          </div>
          <div class="score-label" style="margin-top:0.9rem;">Robustness Label</div>
        </div>
        """, unsafe_allow_html=True)

    with col_div:
        div_sim = divergent_pair["similarity"] if divergent_pair else 1.0
        st.markdown(f"""
        <div class="score-card">
          <div class="score-number" style="font-size:2.8rem;">{div_sim:.2f}</div>
          <div class="score-label">Most Divergent Pair</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Similarity heatmap ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">Similarity Heatmap</div>', unsafe_allow_html=True)
    st.info(
        "**How to read this:** This matrix shows how semantically similar the AI's answers were to each other. "
        "A score of **1.0** means the meaning was identical. **Lower scores** mean the AI changed its answer significantly "
        "when the prompt was rephrased. The darker the square, the more the answers diverged."
    )

    labels = [f"V{i+1}" for i in range(len(variations))]
    fig = go.Figure(go.Heatmap(
        z=sim_matrix,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#1e1b4b"],
            [0.5, "#4338ca"],
            [1.0, "#a78bfa"],
        ],
        zmin=0,
        zmax=1,
        text=np.round(sim_matrix, 2),
        texttemplate="%{text}",
        textfont={"size": 12, "color": "white"},
        showscale=True,
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)", family="Inter"),
        height=360,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Most divergent pair callout ────────────────────────────────────────
    if divergent_pair:
        st.markdown('<div class="section-title">Most Divergent Pair</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="divergent-box">
          <div class="divergent-title">Similarity: {divergent_pair['similarity']:.3f}</div>
          <div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem;">
            <div>
              <div style="font-size:0.7rem; color:rgba(167,139,250,0.7); text-transform:uppercase; letter-spacing:0.08em;">Variation {divergent_pair['index_1']+1}</div>
              <div style="color:rgba(255,255,255,0.8); font-size:0.88rem; margin-top:0.2rem;">{divergent_pair['prompt_1']}</div>
            </div>
            <div>
              <div style="font-size:0.7rem; color:rgba(167,139,250,0.7); text-transform:uppercase; letter-spacing:0.08em;">Variation {divergent_pair['index_2']+1}</div>
              <div style="color:rgba(255,255,255,0.8); font-size:0.88rem; margin-top:0.2rem;">{divergent_pair['prompt_2']}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Response Comparison</div>', unsafe_allow_html=True)

    # Use native Streamlit containers to provide strict styling and built-in text-copy functionality
    for i, (var, resp) in enumerate(zip(variations, responses)):
        with st.container(border=True):
            st.markdown(f"#### Variation {i+1}")
            st.caption(f"Theme: **{var['theme']}**")
            st.code(var['prompt'], language="text")
            st.markdown(f"**AI Response:**\n\n{resp}")

else:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 0; color:rgba(255,255,255,0.2);">
      <div style="margin-top:0.8rem; font-size:0.95rem;">Enter a prompt above and click <strong>Run Analysis</strong> to begin.</div>
    </div>
    """, unsafe_allow_html=True)
