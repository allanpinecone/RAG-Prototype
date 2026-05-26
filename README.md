# RAG Prototype

A generic, full-stack RAG (Retrieval-Augmented Generation) app for querying PDF documents with source-grounded answers.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Pinecone](https://img.shields.io/badge/Pinecone-Integrated_Inference-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red)

## What It Does

This project lets you ingest a folder of PDFs, ask natural-language questions, and get answers with cited passages and page numbers.

**Upload documents -> Ask questions -> Get sourced answers.**

### Key Features

- Conversational Q&A over your full document corpus
- Compare mode for side-by-side comparisons
- Calculate mode for derived numeric answers with shown working
- Document explorer with optional document-level filter
- Source panel with metadata, relevance score, and excerpt
- Session memory across follow-up questions

## Architecture

```text
┌────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   Web Frontend     │────▶│   FastAPI Backend    │────▶│   Pinecone       │
│   (Vanilla JS)     │◀────│   (Python)           │◀────│   (Serverless)   │
│                    │     │                      │     │   llama-text-    │
│  • Chat UI         │     │  • /api/chat         │     │   embed-v2       │
│  • Document list   │     │  • /api/documents    │     │   2048 dims      │
│  • Source panel    │     │  • /api/health       │     │                  │
│  • Mode selector   │     │  • /api/suggest      │     └──────────────────┘
└────────────────────┘     │                      │
                           │  GPT-4o for answers  │
                           └──────────────────────┘
```

Pinecone Integrated Inference handles embeddings for both upsert and search using `llama-text-embed-v2` (2048 dimensions).

## Quick Start

### Prerequisites

- Python 3.10+
- [Pinecone](https://www.pinecone.io/) API key
- [OpenAI](https://platform.openai.com/) API key

### 1) Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2) Configure `.env`

Create `.env` in the repo root:

```env
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
PINECONE_INDEX_NAME=rag-prototype
```

### 3) Add Documents

```bash
mkdir -p Documents
```

Put PDFs under `Documents/` (subdirectories are supported).

### 4) Ingest

```bash
python backend/ingest.py
```

### 5) Run

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000).

## API

- `POST /api/chat` - query with retrieval + generation
- `GET /api/documents` - list ingested documents
- `GET /api/health` - index health and vector counts
- `GET /api/suggest` - starter prompts by mode

Example `POST /api/chat` body:

```json
{
  "query": "What are the key requirements in this document set?",
  "mode": "chat",
  "top_k": 8,
  "document_filter": null,
  "conversation_history": []
}
```

## Project Structure

```text
├── backend/
│   ├── app.py
│   ├── ingest.py
│   ├── generate_dummy_data.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── style.css
│   └── app.js
├── Documents/
├── PROMPT.md
└── README.md
```
