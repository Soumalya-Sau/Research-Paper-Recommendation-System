# 📄 Semantic Research Paper Recommendation System

A **semantic research paper recommendation system** built using **Sentence-BERT embeddings**, **TF-IDF baselines**, and **Streamlit**.  
The system mimics search behavior by understanding the **semantic intent** behind user queries and retrieving research papers that are meaningfully relevant — not just keyword matched.

This project demonstrates **semantic search design**, **engineer-level model comparison**, and an interactive **UI for real-world usage**, inspired by production-grade retrieval systems.

---

## 🚀 Key Features

- 🧠 Uses **Sentence-BERT (SBERT)** to generate semantic embeddings of research abstracts  
- 🔎 Computes **cosine similarity** between query and paper embeddings for ranking  
- 📊 Includes a **TF-IDF baseline** to show limitations of keyword-based retrieval  
- 📈 Manual relevance evaluation and explanation of similarity scores  
- 🖥️ **Streamlit UI** for interactive paper recommendations  
- ⚙️ Offline embedding computation for fast inference  
- 📂 Structured codebase with notebook prototyping and production code  

---

## 🧠 How It Works

The system processes research paper abstracts and converts them into dense vectors using a pre-trained SBERT model. During inference, a user enters a natural language query, which is also embedded. The system then ranks papers by cosine similarity between the query embedding and the precomputed paper embeddings.

To highlight the advantage of semantic representations, the system also implements a traditional **TF-IDF baseline**. While TF-IDF may produce higher raw cosine similarity due to keyword overlap, its results are often contextually irrelevant. In contrast, SBERT embeddings capture **meaning** and retrieve semantically aligned papers.

---

## 📂 Project Structure

```
paper-recommendation-system/
│
├── app/
│ └── app.py # Streamlit UI application
│
├── data/
│ └── processed_data.csv # Cleaned dataset for inference (ignored on GitHub)
│
├── models/
│ └── embeddings.npy # Precomputed SBERT embeddings (ignored on GitHub)
│
├── notebooks/
│ └── Paper_Recommendation.ipynb # Notebook for EDA & comparison
│
├── requirements.txt
├── .gitignore
└── README.md

```

> ⚠️ Large files such as processed data and embeddings are excluded from version control via `.gitignore` to keep the repository lightweight.

---

## 🧪 Dataset

- **Source:** arXiv research paper metadata  
- **Total papers:** ~1.7 million  
- **Subset used for experiments:** ~100,000  
- **Fields:** titles, abstracts, categories, metadata  

Due to memory and embedding computation constraints, a subset of approximately 100K papers was used. The design can be extended to the full dataset using scalable vector search techniques such as **FAISS**.

---

## 🖥️ Streamlit Demo

The Streamlit UI allows users to:

- Enter a free-form research query  
- Select the number of papers to recommend  
- View recommended paper titles, categories, and similarity scores  
- Interactively explore semantic retrieval results  

### Run Locally

```
pip install -r requirements.txt
streamlit run app/app.py

```

The application launches in your browser at:

```

http://localhost:8501

```
---

## 📊 Evaluation Metrics

| Metric                         | Observation                                  |
|--------------------------------|----------------------------------------------|
| Semantic Relevance (SBERT)     | High quality, contextually relevant results |
| Lexical Matching (TF-IDF)      | Often retrieves irrelevant papers            |
| Similarity Score Comparison    | Not directly comparable across models        |
| Ranking Quality                | SBERT consistently performs better           |

Evaluation is performed through **qualitative inspection and manual relevance analysis** due to the absence of labeled relevance data.

---

## 🧪 Notebook Usage

The Jupyter notebook is used for:

- Data preprocessing  
- Exploratory data analysis  
- Embedding generation  
- TF-IDF vs SBERT comparison  
- Result visualization  

> ⚠️ Production logic is implemented only in `.py` files.  
> The notebook is strictly for experimentation and analysis.

---

## 💡 Future Enhancements

- ⚡ Vector search using **FAISS** or other scalable indexes  
- 📦 API deployment using **FastAPI** or similar frameworks  
- 🔎 Advanced filtering by year, category, or author  
- 🌐 Hosted deployment (e.g., **Streamlit Cloud**)  
- 📈 Automated evaluation metrics with human feedback  

---

## 📌 Why This Project Matters

This project demonstrates:

- Applied use of NLP and semantic search  
- Understanding of embedding spaces and similarity metrics  
- Proper baseline comparison and evaluation reasoning  
- Awareness of real-world constraints and practical deployment  
- Transition from a research notebook to a usable interactive application  

---

## 👤 Author

**Soumalya Sau**  
*M.Tech, IIT Kharagpur*  
**Interests:** Data Science, NLP, Semantic Search, GenAI, ML Systems
