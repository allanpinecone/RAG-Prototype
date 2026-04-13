# VDR Copilot

An AI-powered document intelligence platform for querying financial documents from Virtual Data Rooms (VDRs). Automates the analysis of fund prospectuses, KIIDs, factsheets, and reports through a conversational interface.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Pinecone](https://img.shields.io/badge/Pinecone-Integrated_Inference-green) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-red)

## Reproducing from Scratch

The `PROMPT.md` file contains a self-contained AI coding prompt that describes the entire application. You can copy and paste it into any AI coding tool (Cursor, Copilot, Windsurf, etc.) to regenerate the full project from scratch.

## What It Does

Financial teams manually review thousands of pages of fund documents from deal rooms during investment workflows. This prototype demonstrates how RAG (Retrieval Augmented Generation) can automate that process through a conversational interface.

**Upload fund documents → Ask questions → Get sourced answers in seconds.**

### Key Features

- **Conversational Q&A** — Natural language questions over your entire document corpus with cited sources
- **Fund Comparison Mode** — Side-by-side comparison of fees, allocations, risks, and terms across funds
- **Calculation Mode** — Extract figures and perform derived calculations with step-by-step working
- **Document Explorer** — Browse all ingested documents by type and fund family
- **Source Citations** — Every answer links back to the exact document, page, and passage
- **Session Memory** — Follow-up questions maintain conversational context
- **Document Filtering** — Scope queries to a specific fund or document

### Architecture

```
┌────────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│   Web Frontend     │────▶│   FastAPI Backend     │────▶│   Pinecone       │
│   (Vanilla JS)     │◀────│   (Python)            │◀────│   (Serverless)   │
│                    │     │                       │     │   llama-text-    │
│  • Chat UI         │     │  • /api/chat          │     │   embed-v2       │
│  • Document list   │     │  • /api/documents     │     │   2048 dims      │
│  • Source panel     │     │  • /api/health        │     │                  │
│  • Mode selector   │     │  • /api/suggest       │     └──────────────────┘
└────────────────────┘     │                       │
                           │  GPT-4o for answers   │
                           └──────────────────────-┘
```

**Pinecone Integrated Inference** handles all embedding — no separate embedding API calls needed. Documents are embedded on upsert and queries are embedded on search, all server-side using NVIDIA's `llama-text-embed-v2` model at 2048 dimensions.

## Quick Start

### Prerequisites

- Python 3.10+
- A [Pinecone](https://www.pinecone.io/) API key
- An [OpenAI](https://platform.openai.com/) API key (for GPT-4o answer generation)

### 1. Clone & Install

```bash
git clone https://github.com/your-org/vdr-copilot.git
cd vdr-copilot

python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

### 2. Configure

Copy the example environment and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
PINECONE_INDEX_NAME=vdr-copilot
```

### 3. Add Documents

Create the `Documents/` directory and add your PDF files (subdirectories are supported):

```bash
mkdir -p Documents
```

Copy your fund PDFs into `Documents/`. For example:

```
Documents/
├── prospectus.pdf
├── factsheet.pdf
└── Vanguard fund 60-70/
    ├── annual-report.pdf
    └── kiid.pdf
```

### 4. Ingest Documents

Run the ingestion pipeline to extract, chunk, and index your documents:

```bash
python backend/ingest.py
```

This will:
- Create a Pinecone serverless index with `llama-text-embed-v2` (2048 dimensions)
- Extract text from all PDFs using PyMuPDF
- Chunk text with overlap for better retrieval
- Upsert all chunks with metadata (fund name, category, page number, source file)

### 5. Run the Application

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

### Chat Mode
Ask any question about your fund documents. The copilot retrieves relevant passages from Pinecone and generates a sourced answer.

> *"What is the investment objective of the Vanguard LifeStrategy 60% Equity Fund?"*

### Compare Mode
Compare multiple funds or documents side by side. Results are presented in structured tables.

> *"Compare the ongoing charges between the LifeStrategy 40, 60, and 80 equity funds"*

### Calculate Mode
Extract figures and perform calculations with shown working.

> *"What is the total expense ratio for the Franklin S&P 500 Paris Aligned Climate ETF?"*

### Document Filtering
Click any document in the sidebar to scope all queries to that specific fund. Click again to clear the filter.

## Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application & RAG pipeline
│   ├── ingest.py           # PDF extraction & Pinecone ingestion
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Single-page application
│   ├── style.css           # Dark theme UI
│   └── app.js              # Client-side logic
├── Documents/              # Create this folder and add your PDFs (not committed)
│   ├── *.pdf
│   └── subfolder/*.pdf
├── .env                    # API keys (not committed)
├── .env.example            # Template for environment variables
├── .gitignore
├── PROMPT.md               # AI coding prompt to regenerate the project
└── README.md
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send a query, get a RAG-powered answer with sources |
| `/api/documents` | GET | List all ingested documents |
| `/api/health` | GET | Check index status and vector count |
| `/api/suggest` | GET | Get suggested starter questions |

### POST `/api/chat`

```json
{
  "query": "What are the fund's key risks?",
  "mode": "chat",
  "top_k": 8,
  "fund_filter": null,
  "conversation_history": []
}
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Vector Database | Pinecone (Serverless) |
| Embedding Model | NVIDIA `llama-text-embed-v2` via Pinecone Integrated Inference |
| Embedding Dimensions | 2048 |
| LLM | OpenAI GPT-4o |
| Backend | FastAPI + Uvicorn |
| PDF Processing | PyMuPDF |
| Frontend | Vanilla HTML/CSS/JS |

## Configuration

| Environment Variable | Description | Default |
|---------------------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key | *required* |
| `OPENAI_API_KEY` | Your OpenAI API key | *required* |
| `PINECONE_INDEX_NAME` | Name of the Pinecone index | `vdr-copilot` |

## License

Prototype — internal use only.
