import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os

# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(
    page_title="Research Paper Recommendation System",
    layout="wide"
)

# -----------------------------
# Helper paths (relative to project root)
# -----------------------------
DATA_PATH = os.path.join("data", "processed_data.csv")
EMBEDDING_PATH = os.path.join("models", "document_embeddings.npy")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# Load data & embeddings (cached)
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    embeddings = np.load(EMBEDDING_PATH)
    return df, embeddings

model = load_model()
df, embeddings = load_data()

# -----------------------------
# App UI
# -----------------------------
st.title("📄 Research Paper Recommendation System")
st.markdown(
    """
    This application performs **semantic search** over research papers using  **Sentence-BERT embeddings**, enabling context-aware paper recommendations.
    """
)

query = st.text_input(
    "Enter a research topic or query:",
    placeholder="e.g. transformer models for language understanding"
)

top_k = st.slider(
    "Number of recommendations",
    min_value=3,
    max_value=10,
    value=5
)

# -----------------------------
# Recommendation logic
# -----------------------------
def recommend_papers(query, top_k=5):
    query_embedding = model.encode([query])
    scores = cosine_similarity(query_embedding, embeddings)[0]
    
    top_indices = np.argsort(scores)[-top_k:][::-1]
    
    results = df.iloc[top_indices][["title", "categories"]].copy()
    results["similarity_score"] = scores[top_indices]
    
    return results

# -----------------------------
# Run search
# -----------------------------
if st.button("🔍 Recommend Papers"):

    if query.strip() == "":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Searching for relevant papers..."):
            results = recommend_papers(query, top_k)

        st.subheader("Recommended Papers")
        st.dataframe(results, use_container_width=True)

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown(
    """
    **Note:**  
    Similarity scores are cosine similarities computed in the embedding space.  
    Scores from different models (e.g., TF-IDF vs SBERT) are **not directly comparable**.
    """
)
