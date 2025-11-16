# 🧠 MSU FAQ Chatbot — Retrieval-Augmented Answering System for University Resources

**Goal:** Build an AI-powered FAQ chatbot for Montclair State University that answers student questions by grounding responses in official university documents (Registrar, Policies, Global Engagement, SoC, etc.).

## 🚀 Key Features

- **Automated Web Data Ingestion**
  - Crawls and fetches sitemap + manually added URLs from Montclair’s official sites.
  - Handles PDFs and HTML content uniformly.
  - Incrementally updates when pages change (using `Last-Modified`, `ETag`, or hash checks).

- **Document Cleaning & Structuring**
  - Converts HTML and PDFs into clean Markdown.
  - Splits long content into small, semantically meaningful chunks (RAG-ready).

- **Semantic Retrieval Engine**
  - Uses `sentence-transformers/all-MiniLM-L6-v2` to embed chunks.
  - Stores vectors in **ChromaDB** for fast, semantic similarity search.

- **RAG (Retrieval-Augmented Generation) Architecture**
  - Retrieves the most relevant policy passages.
  - Supplies them as context to an LLM (e.g., Mistral-7B or LLaMA 2-Chat) for accurate, grounded answers.

- **Extensible Design**
  - Supports new departments, sites, or changed URLs via YAML configuration.
  - Modular structure allows plugging in different embedding or LLM models later.

---

## 🏗️ System Architecture Overview

```text
                   ┌────────────────────┐
                   │  Student Question  │
                   └─────────┬──────────┘
                             │
                             ▼
                   ┌────────────────────┐
                   │ Embedding Model    │ (MiniLM-L6-v2)
                   └─────────┬──────────┘
                             │ vector
                             ▼
                ┌──────────────────────────┐
                │ Vector DB (ChromaDB)     │
                │ Stores all document chunks│
                └─────────┬────────────────┘
                             │ top-k results
                             ▼
                ┌──────────────────────────┐
                │ LLM (Mistral/LLaMA)      │
                │ Generates grounded answer│
                └──────────────────────────┘
```
---

## 🧩 Repository Structure
```text

faq_chatbot/
├── data/
│   ├── raw/                                         # Raw HTML/PDF fetched from sites (ignored in git)
│   ├── processed/                                   # Cleaned Markdown (ignored in git)
│   └── url_list.csv                                 # Master list of discovered URLs (ignored in git)
│
├── ingest/
│   ├── fetch_from_sitemap.py                        # Crawl + filter sitemap URLs
│   ├── discovery_links.py                           # Discover deep links and adapt to new sites
│   ├── pdf_to_markdown.py                           # Convert policy PDFs to text
│   ├── html_to_markdown.py                          # HTML → Markdown
│   ├── extract_tables.py                            # Extract tables into structured JSON/Markdown
│   ├── render_fetch.py                              # Rendered fetch (for JS-heavy pages, future)
│   ├── run_all.py                                   # End-to-end ingestion & conversion pipeline
│   └── __init__.py
│
├── kb/
│   ├── build_kb.py                                  # Generate Q/A pairs for FAQs
│   ├── faq.csv                                      # Seed FAQ questions (curated)
│   ├── faq.jsonl                                    # JSONL version of FAQ data
│   ├── seed_questions.yaml                          # Starter questions for evaluation
│   └── __init__.py
│
├── rag/
│   ├── chunker.py                                   # Split Markdown into smaller sections
│   ├── chunks.jsonl                                 # Chunked text ready for embeddings (ignored in git for new files)
│   └── __init__.py
│
├── vectorstore/
│   ├── build_index_retrieval.py                      # Build Chroma index + retrieval pipeline
│   ├── rerank.py                                     # Intent detection + reranking logic
│   ├── test_retrieval_query.py                       # Manual test harness for retrieval
│   ├── tests/
│   │   └── regression_testing_retrieval.py           # Automated regression tests
│   └── __init__.py
│
├── eval/
│   ├── policy_postfilter.py                          # Keep only relevant policy topics
│   ├── preview_chunks.py                             # Preview sample chunks for sanity checks
│   └── __init__.py
│
├── sources.yaml                                      # URL/bucket configuration for ingestion
├── SECURITY.md                                       # Security & commit guidelines
├── README.md                                         # Project overview and usage
└── requirements.txt                                  # Required dependencies
```

---

## ⚙️ Prerequisites

Python ≥ 3.9
Virtual environment (venv or conda)
Dependencies: 
```bash
pip install -r requirements.txt
```
---

## 🧰 Quick Start

```bash
# 1. Clone repository
git clone https://github.com/<your_username>/faq_chatbot.git
cd faq_chatbot

# 2. Activate virtual environment
python -m venv .venv
source .venv/bin/activate                               # Mac/Linux
.venv\Scripts\activate                                  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python -m ingest.run_all                                # Crawl + HTML → Markdown
python -m ingest.pdf_to_markdown                        # Extract PDFs → Markdown (if any)
python -m eval.policy_postfilter                        # Filter / focus on relevant policies
python -m rag.chunker                                   # Split Markdown into chunks
python -m eval.preview_chunks                           # Optional: sanity-check chunks
python -m kb.build_kb                                   # Optional: generate FAQ-style Q/A
python -m vectorstore.build_index_retrieval
python -m vectorstore.test_retrieval_query
python -m vectorstore.tests.regression_testing_retrieval

```

---

## 🧠 Configuration

- **Main configuration lives in sources.yaml:**
  - Add new sitemap URLs or manual pages here.
  - Edit regex rules under `buckets` to control which URLs are included/excluded.
  - The system automatically adapts to new external sites using discovery + auto-sitemap detection.

---

## 🧩 Deployment Options (Future)

- Local FastAPI server (for REST API)
- Streamlit Web UI (for chatbot interface)
- Cloud deployment via:
  - Docker container on Azure App Service or AWS Lightsail
  - Optional GPU inference with Hugging Face Inference API

---

## 💻 Development Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
 
- **Folder paths (inside YAML):**
  - data/raw → stores raw HTML/PDFs
  - data/processed → cleaned Markdown
  - rag/chunks.jsonl → chunked knowledge base
  - vectorstore/chroma/ → persistent ChromaDB index

---

## 🤝 Contribution Guidelines

- 1. Fork this repository.
- 2. Create a new branch:
```bash
     git checkout -b feature/my-feature
```
- 3. Commit your changes:
```bash
     git commit -m "Added new sitemap or fixed bug"
```
- 4. Push the branch and open a Pull Request.

---

## 🔒 Security & Commit Policy

📂 See [`SECURITY.md`](./SECURITY.md)

 - Do not commit API keys, credentials, or .env files.
 - .gitignore includes common sensitive paths (/data/, .venv/, /state/).
 - Never push raw student or internal documents.
 - All scraping follows robots.txt and university data-access policies.

---

## 🙏 Acknowledgments

- Montclair State University — School of Computing
- Faculty Advisors: Prof. Shang, Prof. Wang
- Contributors: Harshitha Ramakrishna, Imaduddin Syed, Lam Nguyen, Prudhvi Raj Kore
- Technologies: Python, ChromaDB, Sentence-Transformers, Mistral-7B, Streamlit

---

**📘 This repository is part of the MSU AI FAQ Chatbot project — a prototype system integrating RAG pipelines for student support and policy retrieval.**
