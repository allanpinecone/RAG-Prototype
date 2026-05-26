# Generic RAG Prototype — AI Coding Prompt

Paste this prompt into an AI coding tool to generate a full-stack, generic document RAG application using Pinecone + OpenAI.

---

## Prompt

Build a full-stack application called **"RAG Prototype"** for querying PDF documents with a chatbot interface. The app should return grounded answers with source citations.

### Tech Stack (required)

- Backend: Python, FastAPI, Uvicorn
- Vector DB: Pinecone serverless with Integrated Inference using `llama-text-embed-v2` at 2048 dimensions
- LLM: OpenAI GPT-4o
- PDF processing: PyMuPDF (`fitz`)
- Frontend: Vanilla HTML/CSS/JS (no framework, no build step)
- Config: `python-dotenv`

### Environment Variables

```env
PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key
PINECONE_INDEX_NAME=rag-prototype
```

### Ingestion (`backend/ingest.py`)

Create a script that:

1. Recursively reads `Documents/**/*.pdf` and `Documents/**/*.PDF`.
2. Extracts page text via PyMuPDF.
3. Cleans text and chunks into ~1000 chars with ~200 overlap.
4. Creates a Pinecone index for integrated inference if missing.
5. Upserts to namespace `documents` in batches of 50.
6. Uses record schema:
   - `_id`
   - `chunk_text`
   - `document_name`
   - `document_type`
   - `source_file`
   - `page_number`
   - `folder`

### Backend (`backend/app.py`)

Implement endpoints:

- `POST /api/chat`
  - Input:
    ```json
    {
      "query": "string",
      "conversation_history": [{"role": "user|assistant", "content": "..."}],
      "document_filter": "optional",
      "top_k": 8,
      "mode": "chat|compare|calculate"
    }
    ```
  - Search Pinecone with integrated inference and fields: `chunk_text`, `document_name`, `document_type`, `source_file`, `page_number`, `folder`.
  - Build prompt context from retrieved excerpts.
  - Instruct GPT-4o to answer only from context, cite source + page, use tables for compare mode, and show working for calculate mode.
  - Return answer + truncated source excerpts with scores.

- `GET /api/documents` - return unique documents from index metadata.
- `GET /api/health` - return index health + vector counts.
- `GET /api/suggest` - return generic starter questions for `chat`, `compare`, `calculate`.

Mount `frontend/` at `/` and allow CORS for all origins (prototype).

### Frontend

Build a 3-panel dark UI:

1. Sidebar: app title, mode selector, searchable document list, health indicator.
2. Main area: welcome screen + suggestions, message history, input box.
3. Source panel: slide-out source cards with doc name/type/page/file/excerpt/score.

Client state should track:

- `mode`
- `conversationHistory`
- `documentFilter`
- `documents`
- `suggestions`
- `currentSources`

### README

Generate a README for a generic RAG prototype (not domain-specific), including setup, architecture, API usage, and configuration.
