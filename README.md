# 📘 FAQ Chatbot – School of Computing

## 🎯 Project Overview
This project builds a **FAQ-style chatbot** to help students quickly get answers to curriculum, policy, administrative, and resource questions.  
- Knowledge source: **official university documents** (course catalogs, policies, registrar guides, resources).  
- Core pipeline: **Collect → Clean → Chunk → Embed → Retrieve → Answer**.  
- Prototype will use a **FastAPI backend** and a **Streamlit frontend**, always citing official sources.

---

## 🗂️ Repository Structure
faq-chatbot/
│
├── ingest/ # Data acquisition scripts (sitemap, API, feeds)
├── data/
│ ├── raw/ # Raw downloads (HTML, XML, JSON, PDFs)
│ └── processed/ # Clean Markdown files with metadata
├── rag/ # Retrieval pipeline (chunking, embeddings, vector DB)
├── api/ # FastAPI backend with /ask endpoint
├── ui/ # Streamlit chatbot interface
├── eval/ # Evaluation dataset and notebooks
├── infra/ # Deployment configs (Docker, Nginx) – later
├── docs/ # Notes, meeting records, designs
├── .env # Environment variables (local use, not committed)
├── requirements.txt # Python dependencies
└── README.md # Project overview and setup guide

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone <your_repo_url>
cd faq-chatbot


### 2. Set up virtual environment
```bash
# Create virtual environment
python3 -m venv .venv

# Activate the environment
# Mac/Linux
source .venv/bin/activate
# Windows
.\.venv\Scripts\activate


### 3. Install dependencies
pip install -r requirements.txt

### 4. Configure environment
#Create a .env file in the project root and add:

BASE_DOMAIN=https://www.montclair.edu
SOC_BASE=https://www.montclair.edu/school-of-computing
WP_API=https://www.montclair.edu/wp-json/wp/v2

### 5. Run ingestion
# Once ingestion scripts are ready, you’ll run:
python ingest/run_all.py

#This will:
Save raw files → data/raw/
Save cleaned Markdown files → data/processed/

