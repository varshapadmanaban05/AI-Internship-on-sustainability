# RePurpose AI

An AI-powered circular economy advisor built for the 1M1B IBM SkillsBuild AI for Sustainability Internship.

Describe any unwanted item and get a structured recommendation from the circular economy hierarchy:
**Reuse As-Is -> Repair -> Repurpose -> Donate -> Refurbish -> Recycle -> Responsible Disposal**

Recommendations are grounded in a curated knowledge base using RAG (Retrieval-Augmented Generation), so every answer cites its source.

**Primary SDG: SDG 12 -- Responsible Consumption & Production**

---

## Features

- Chat interface -- describe any item in natural language
- RAG pipeline -- answers grounded in real sustainability guidelines, not hallucinated
- 7-level circular hierarchy -- always recommends the highest-value option first
- Hazard detection -- flags e-waste, batteries, chemicals with a safety warning
- Source citations -- every recommendation shows which documents it used
- Multi-turn conversation -- ask follow-up questions without starting over

---

## Setup

### 1. Clone the repo

    git clone <your-repo-url>
    cd AI-Internship-on-sustainability

### 2. Install dependencies

    pip install -r requirements.txt

### 3. Add your OpenAI API key

    copy .env.example .env

Open .env and set:

    OPENAI_API_KEY=sk-...
    OPENAI_MODEL=gpt-3.5-turbo

Get a free API key at https://platform.openai.com

### 4. Build the knowledge base

    python ingest.py

This embeds the 6 included documents into a local ChromaDB vector store.

### 5. Run the app

    streamlit run app.py

Open http://localhost:8501 in your browser.

---

## Project Structure

    .
    app.py              # Streamlit chatbot (UI + RAG + LLM in one file)
    ingest.py           # Run once to load docs into vector store
    data/docs/          # Knowledge base documents (PDF or TXT)
    data/chroma_db/     # Auto-generated vector store
    requirements.txt
    .env.example

---

## How It Works

1. You describe an unwanted item
2. The app embeds your description with sentence-transformers
3. Retrieves relevant chunks from ChromaDB
4. Sends context + message to OpenAI
5. Returns structured recommendation: pathway, reasoning, citations, next steps, confidence
6. Hazardous keywords trigger a safety warning before the recommendation

---

## Tech Stack

| Component    | Technology                          |
|-------------|-------------------------------------|
| UI           | Streamlit                           |
| Embeddings   | sentence-transformers all-MiniLM-L6-v2 |
| Vector Store | ChromaDB (local)                    |
| LLM          | OpenAI gpt-3.5-turbo                |

---

## SDG Alignment

- **Primary: SDG 12** -- Responsible Consumption & Production
- **SDG 13** -- Climate Action
- **SDG 11** -- Sustainable Cities & Communities
