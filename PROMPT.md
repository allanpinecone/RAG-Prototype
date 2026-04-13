# VDR Copilot — AI Coding Prompt

Paste this entire prompt into your AI coding tool (Cursor, Copilot, Windsurf, etc.) to generate a complete RAG-powered document Q&A application using Pinecone and OpenAI.

---

## Prompt

Build me a full-stack RAG (Retrieval Augmented Generation) application called **"VDR Copilot"** — a chatbot-style interface for querying PDF documents from a virtual data room. The app should let users ask natural language questions and get sourced answers from their documents.

### Tech Stack (required)

- **Backend:** Python, FastAPI, Uvicorn
- **Vector Database:** Pinecone (serverless) using **Integrated Inference** with the `llama-text-embed-v2` embedding model (NVIDIA Llama) at **2048 dimensions**. This means no separate embedding API calls — Pinecone handles embedding on both upsert and search automatically.
- **LLM:** OpenAI GPT-4o for answer generation
- **PDF Processing:** PyMuPDF (`fitz`)
- **Frontend:** Vanilla HTML, CSS, and JavaScript (no build step, no framework)
- **Config:** python-dotenv for environment variables

### Project Structure

```
├── backend/
│   ├── app.py              # FastAPI application & RAG pipeline
│   ├── ingest.py           # PDF extraction, chunking, & Pinecone ingestion
│   └── requirements.txt    # Python dependencies
├── frontend/
│   ├── index.html          # Single-page application
│   ├── style.css           # Dark theme UI styles
│   └── app.js              # Client-side logic
├── Documents/              # User creates this folder and adds their PDFs
├── .env                    # API keys (gitignored)
├── .env.example            # Template for environment variables
├── .gitignore
└── README.md
```

### Environment Variables

```
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
PINECONE_INDEX_NAME=vdr-copilot
```

### Dependencies (requirements.txt)

```
fastapi>=0.115.0
uvicorn>=0.34.0
pinecone>=5.4.0
pymupdf>=1.25.0
python-dotenv>=1.0.0
openai>=1.60.0
python-multipart>=0.0.18
```

---

### Ingestion Pipeline (`backend/ingest.py`)

Build a script that:

1. Recursively collects all `.pdf` and `.PDF` files from a `Documents/` directory (including subdirectories).
2. Extracts text page-by-page using PyMuPDF.
3. Cleans the text (collapse whitespace, strip non-printable characters while preserving currency symbols like £€¥).
4. Chunks text into ~1000-character segments with ~200-character overlap, preferring sentence boundaries (split at `. ` when possible).
5. Creates a Pinecone index using `create_index_for_model()` with integrated inference if it doesn't already exist:
   - Model: `llama-text-embed-v2`
   - Dimension: 2048
   - Metric: cosine
   - Cloud: aws, Region: us-east-1
   - Field map: `{"text": "chunk_text"}`
6. Upserts records in batches of 50 using `index.upsert_records()` with namespace `"funds"`. Each record should have:
   - `_id`: a unique hash-based ID from the file path + chunk index
   - `chunk_text`: the text chunk (this is the field that gets embedded by Pinecone)
   - `fund_name`: a human-readable name derived from the filename (replace hyphens/underscores with spaces, strip common suffixes like "en", "gb", "uk", etc.)
   - `category`: auto-detected document type based on filename patterns (Prospectus, KIID/KID, Factsheet, Annual Report, Interim Report, Information Document, etc.)
   - `source_file`: the relative file path from the Documents directory
   - `page_number`: the page the chunk came from
   - `folder`: the parent subdirectory name (or "Root" if top-level)
7. Wait for the index to be ready before upserting (poll `describe_index` until `status.ready` is `True`).
8. Print progress as it processes each file.

---

### Backend API (`backend/app.py`)

Build a FastAPI application with these endpoints:

#### `POST /api/chat`
The main RAG endpoint. Accepts:
```json
{
  "query": "string",
  "conversation_history": [{"role": "user/assistant", "content": "..."}],
  "fund_filter": "optional fund name to scope search",
  "top_k": 8,
  "mode": "chat | compare | calculate"
}
```

This endpoint should:
1. Search Pinecone using `index.search()` with integrated inference (pass the query as `{"inputs": {"text": query}}`). Request fields: `chunk_text`, `fund_name`, `category`, `source_file`, `page_number`, `folder`. If `fund_filter` is provided, add a metadata filter.
2. Build a context string from the retrieved sources, including source numbers, document names, types, files, and page numbers.
3. Send the context + query to GPT-4o with a system prompt that instructs the copilot to:
   - Only answer from the provided context
   - Always cite source document and page number
   - Use structured tables for comparisons
   - Show step-by-step working for calculations
   - Be precise with numbers (don't round unless asked)
4. Include the last 6 messages from `conversation_history` for session memory.
5. Append mode-specific instructions:
   - **Compare mode**: instruct the LLM to use markdown tables and highlight differences
   - **Calculate mode**: instruct the LLM to show step-by-step working
6. Return the answer plus truncated source excerpts (max 300 chars each) with relevance scores.

#### `GET /api/documents`
List unique documents that have been ingested. Search Pinecone with a broad query, deduplicate by `source_file`, and return fund name, category, source file, and folder for each.

#### `GET /api/health`
Return index status including total vector count and namespace breakdown using `describe_index_stats()`.

#### `GET /api/suggest`
Return starter questions **grouped by mode** (`chat`, `compare`, `calculate`), with 6 suggestions per mode. Each suggestion has a `text` and an `icon` identifier. Make the suggestions generic enough to work with any set of financial PDF documents (fund prospectuses, factsheets, KIIDs, annual reports, etc.). Examples:
- Chat: "What are the key risks disclosed in the fund documents?", "Summarise the investment objectives across the funds"
- Compare: "Compare the ongoing charges across the funds", "How do the asset allocations differ between the funds?"
- Calculate: "What is the total expense ratio?", "If I invested £100,000, what would the annual charges be?"

#### Static File Serving
Mount the `frontend/` directory as static files at `/` with `html=True` so the frontend is served from the same origin.

#### CORS
Allow all origins (prototype).

---

### Frontend

Build a dark-themed, modern single-page application with the Inter font from Google Fonts. No framework — vanilla HTML/CSS/JS only.

#### Layout (three-panel)
1. **Left sidebar** (collapsible, 300px):
   - Logo area with app name "VDR Copilot" and a subtitle area
   - **Mode selector**: three buttons — Chat, Compare, Calculate — each with an SVG icon, a tooltip on hover explaining the mode, and visual active state. Clicking a mode should show the welcome screen with that mode's suggestions (even if a chat is open).
   - **Document explorer**: a filterable list of ingested documents loaded from `/api/suggest`. Each item shows the fund name, a color-coded category badge (Prospectus = purple, KIID = green, Factsheet = amber, Report = blue), and folder. Clicking a document scopes all queries to that fund (shows a filter bar above the input); clicking again clears the filter.
   - **Health indicator** at the bottom showing vector count from `/api/health`.

2. **Main content area** (center):
   - **Welcome screen** (shown by default and when switching modes): centered logo icon, heading, description, and a 2-column grid of suggestion cards for the current mode. Each card has an emoji icon with a colored background and the suggestion text. Clicking a card sends it as a query.
   - **Messages area** (shown after first query): alternating user/assistant messages with avatars, sender labels, and rendered markdown content (support bold, italic, code, headings, lists, and tables). Assistant messages include a "N sources" button.
   - **Input area** at the bottom: auto-resizing textarea, send button (disabled when empty), mode hint text, keyboard shortcuts (Enter to send, Shift+Enter for newline).

3. **Right source panel** (slides open, 380px):
   - Triggered by clicking the "sources" button on any assistant message.
   - Shows source cards with: fund name, relevance score as percentage, category badge, page number, source file, and an excerpt with a left border accent.
   - Close button to dismiss.

#### Styling Details
- Dark color scheme: backgrounds from `#0a0f1a` to `#1e293b`, blue accent `#3b82f6`, green/amber/purple for badges
- Smooth transitions (180ms ease) on hovers and state changes
- Custom thin scrollbar styling
- `fadeIn` animation on new messages
- Typing indicator with bouncing dots while waiting for response
- Responsive: sidebar collapses on mobile, source panel goes full-width
- Mode tooltips appear to the right of the buttons with an arrow/caret pointing left

#### JavaScript Behavior
- State object tracks: current mode, conversation history, fund filter, documents list, suggestions (grouped by mode), current sources
- Suggestions are fetched once from `/api/suggest` and stored; `renderSuggestions(mode)` re-renders the grid when the mode changes
- Basic markdown-to-HTML renderer supporting: tables (pipe syntax), bold, italic, inline code, headings (h2-h4), unordered lists, paragraph wrapping
- `escapeHtml()` for XSS prevention on all user-generated and API-returned content

---

### README

Write a comprehensive README covering:
- What the app does (VDR document Q&A for financial analysts)
- Architecture diagram (ASCII)
- Prerequisites (Python 3.10+, Pinecone key, OpenAI key)
- Step-by-step setup: clone, create venv, install deps, configure `.env`, create `Documents/` directory, add PDFs, run ingestion, start server
- Usage guide for each mode (Chat, Compare, Calculate) with example queries
- Project structure tree
- API endpoint reference table
- Tech stack table
- Configuration reference table

### .gitignore

Ignore: `.env`, `*.pem`, `Documents/`, `__pycache__/`, `*.pyc`, `.DS_Store`, `node_modules/`, `venv/`, `.venv/`, `*.egg-info/`, `dist/`, `build/`

### .env.example

```
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
PINECONE_INDEX_NAME=vdr-copilot
```
