"""
FastAPI backend for the VDR Copilot.
Provides RAG-powered Q&A over fund documents stored in Pinecone.
"""

import os
import json
import re
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "vdr-copilot")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
llm = OpenAI(api_key=OPENAI_API_KEY)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="VDR Copilot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are the VDR Copilot, an expert AI assistant for analysing private equity and investment fund documents from virtual data rooms.

You help analysts quickly find figures, compare funds, perform calculations, and extract key information from prospectuses, KIIDs, factsheets, annual reports, and other fund documents.

RULES:
- Answer ONLY based on the provided context from the retrieved documents. If the context doesn't contain enough information, say so clearly.
- When citing figures, always mention the source document name and page number.
- When asked to compare funds, present information in a structured format (tables when appropriate using markdown).
- When asked for calculations, show your working step by step.
- Use British English spelling conventions (e.g. "analyse" not "analyze").
- Format monetary values with appropriate currency symbols.
- Be precise with percentages and numerical data — do not round unless asked.
- If a question is ambiguous, ask for clarification rather than guessing.
"""


class ChatRequest(BaseModel):
    query: str
    conversation_history: list[dict] = []
    fund_filter: Optional[str] = None
    top_k: int = 8
    mode: str = "chat"  # "chat", "compare", "calculate"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    mode: str


class DocumentInfo(BaseModel):
    fund_name: str
    category: str
    source_file: str
    folder: str


def search_documents(query: str, top_k: int = 8, fund_filter: Optional[str] = None) -> list[dict]:
    """Search Pinecone using integrated inference."""
    search_params = {
        "namespace": "funds",
        "query": {
            "inputs": {"text": query},
            "top_k": top_k,
        },
        "fields": ["chunk_text", "fund_name", "category", "source_file", "page_number", "folder"],
    }

    if fund_filter:
        search_params["query"]["filter"] = {"fund_name": {"$eq": fund_filter}}

    results = index.search(**search_params)
    sources = []
    for hit in results.result.hits:
        fields = hit.fields or {}
        sources.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "text": fields.get("chunk_text", ""),
            "fund_name": fields.get("fund_name", "Unknown"),
            "category": fields.get("category", "Unknown"),
            "source_file": fields.get("source_file", ""),
            "page_number": fields.get("page_number", 0),
            "folder": fields.get("folder", ""),
        })
    return sources


def build_context(sources: list[dict]) -> str:
    """Build a context string from retrieved sources for the LLM."""
    context_parts = []
    for i, src in enumerate(sources, 1):
        context_parts.append(
            f"[Source {i}] Document: {src['fund_name']} | "
            f"Type: {src['category']} | "
            f"File: {src['source_file']} | "
            f"Page: {src['page_number']}\n"
            f"{src['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def get_mode_instruction(mode: str) -> str:
    if mode == "compare":
        return (
            "\n\nThe user wants to COMPARE funds or documents. "
            "Present your answer in a structured comparison format, using markdown tables where appropriate. "
            "Highlight key differences and similarities."
        )
    if mode == "calculate":
        return (
            "\n\nThe user wants you to CALCULATE or derive figures. "
            "Show your working step-by-step. Extract the relevant numbers from the context, "
            "state any assumptions, and present the final result clearly."
        )
    return ""


def generate_answer(query: str, sources: list[dict], conversation_history: list[dict], mode: str) -> str:
    """Generate an answer using GPT-4o with retrieved context."""
    context = build_context(sources)
    mode_instruction = get_mode_instruction(mode)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + mode_instruction}]

    for msg in conversation_history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = f"""Based on the following retrieved document excerpts, answer the user's question.

RETRIEVED CONTEXT:
{context}

USER QUESTION: {query}"""

    messages.append({"role": "user", "content": user_message})

    response = llm.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
    )
    return response.choices[0].message.content


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        sources = search_documents(
            query=request.query,
            top_k=request.top_k,
            fund_filter=request.fund_filter,
        )

        if not sources:
            return ChatResponse(
                answer="I couldn't find any relevant information in the fund documents for your query. Please try rephrasing or broadening your question.",
                sources=[],
                query=request.query,
                mode=request.mode,
            )

        answer = generate_answer(
            query=request.query,
            sources=sources,
            conversation_history=request.conversation_history,
            mode=request.mode,
        )

        clean_sources = [
            {
                "fund_name": s["fund_name"],
                "category": s["category"],
                "source_file": s["source_file"],
                "page_number": s["page_number"],
                "text": s["text"][:300] + "..." if len(s["text"]) > 300 else s["text"],
                "score": round(s["score"], 4),
            }
            for s in sources
        ]

        return ChatResponse(
            answer=answer,
            sources=clean_sources,
            query=request.query,
            mode=request.mode,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/documents")
async def list_documents():
    """List unique documents that have been ingested (sampled from the index)."""
    try:
        results = index.search(
            namespace="funds",
            query={"inputs": {"text": "fund overview prospectus factsheet"}, "top_k": 100},
            fields=["fund_name", "category", "source_file", "folder"],
        )
        seen = set()
        documents = []
        for hit in results.result.hits:
            fields = hit.fields or {}
            source_file = fields.get("source_file", "")
            if source_file and source_file not in seen:
                seen.add(source_file)
                documents.append({
                    "fund_name": fields.get("fund_name", "Unknown"),
                    "category": fields.get("category", "Unknown"),
                    "source_file": source_file,
                    "folder": fields.get("folder", "Root"),
                })
        documents.sort(key=lambda d: (d["folder"], d["fund_name"]))
        return {"documents": documents, "total": len(documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    try:
        stats = index.describe_index_stats()
        return {
            "status": "healthy",
            "index": INDEX_NAME,
            "total_vectors": stats.total_vector_count,
            "namespaces": {ns: info.vector_count for ns, info in stats.namespaces.items()},
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


@app.get("/api/suggest")
async def suggest_questions():
    """Return suggested starter questions grouped by mode."""
    return {
        "chat": [
            {
                "text": "What are the key risks associated with the Vanguard LifeStrategy funds?",
                "icon": "shield",
            },
            {
                "text": "Summarise the investment objectives of the Personal Portfolio funds",
                "icon": "target",
            },
            {
                "text": "What ESG or climate-related criteria do the ActiveLife Climate Aware funds use?",
                "icon": "leaf",
            },
            {
                "text": "What is the benchmark index for the Franklin S&P 500 Paris Aligned Climate ETF?",
                "icon": "search",
            },
            {
                "text": "What are the dealing and valuation arrangements for the OEIC funds?",
                "icon": "clock",
            },
            {
                "text": "Which funds have exposure to US equities and what are their top holdings?",
                "icon": "globe",
            },
        ],
        "compare": [
            {
                "text": "Compare the ongoing charges between the LifeStrategy 40, 60, and 80 equity funds",
                "icon": "scale",
            },
            {
                "text": "Compare the asset allocation of the PPF Balanced vs PPF Cautious funds",
                "icon": "pie-chart",
            },
            {
                "text": "How do the risk profiles differ between the Vanguard ActiveLife Climate Aware 40-50 and 80-90 funds?",
                "icon": "shield",
            },
            {
                "text": "Compare the investment objectives of the defensive vs adventurous Personal Portfolio funds",
                "icon": "target",
            },
            {
                "text": "What are the differences in geographic allocation between the LifeStrategy 20 and LifeStrategy 80 funds?",
                "icon": "globe",
            },
            {
                "text": "Compare the ESG factsheets of the balanced, cautious, and defensive funds",
                "icon": "leaf",
            },
        ],
        "calculate": [
            {
                "text": "What is the total expense ratio for the Franklin S&P 500 Paris Aligned Climate ETF?",
                "icon": "calculator",
            },
            {
                "text": "If I invested £100,000 in LifeStrategy 60, what would the annual charges be?",
                "icon": "coins",
            },
            {
                "text": "What is the total AUM across all LifeStrategy funds combined?",
                "icon": "trending-up",
            },
            {
                "text": "What is the percentage split between equities and fixed income in the LifeStrategy 40 fund?",
                "icon": "pie-chart",
            },
            {
                "text": "How much of the ActiveLife Climate Aware 60-70 fund is allocated to international equities?",
                "icon": "globe",
            },
            {
                "text": "What is the difference in ongoing charges between the PPF Ambitious and PPF Defensive funds?",
                "icon": "calculator",
            },
        ],
    }


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
