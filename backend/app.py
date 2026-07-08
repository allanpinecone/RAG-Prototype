"""
FastAPI backend for a generic RAG prototype.
Provides RAG-powered Q&A over documents stored in Pinecone.
"""

import os
import json
import re
import asyncio
import time
import math
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
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-prototype")

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)
llm = OpenAI(api_key=OPENAI_API_KEY)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="RAG Prototype", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """You are an expert retrieval-augmented AI assistant for document analysis.

You help users quickly locate facts, compare information, perform calculations, and extract key details from their uploaded documents.

RULES:
- Answer ONLY based on the provided context from the retrieved documents. If the context doesn't contain enough information, say so clearly.
- When citing figures, always mention the source document name and page number.
- When asked to compare items, present information in a structured format (tables when appropriate using markdown).
- When asked for calculations, show your working step by step.
- Use British English spelling conventions (e.g. "analyse" not "analyze").
- Format values in a way that matches the user's context (including currency symbols when relevant).
- Be precise with percentages and numerical data — do not round unless asked.
- If a question is ambiguous, ask for clarification rather than guessing.
"""


class ChatRequest(BaseModel):
    query: str
    conversation_history: list[dict] = []
    document_filter: Optional[str] = None
    top_k: int = 8
    mode: str = "chat"  # "chat", "compare", "calculate"


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    mode: str


class DocumentInfo(BaseModel):
    document_name: str
    document_type: str
    source_file: str
    folder: str


class LoadTestRequest(BaseModel):
    query: str = "document summary"
    qps: float = 1.0
    duration_seconds: int = 10
    top_k: int = 8
    document_filter: Optional[str] = None


class LoadTestResponse(BaseModel):
    query: str
    qps_target: float
    qps_achieved: float
    duration_seconds: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_latency_ms: Optional[float]
    p50_latency_ms: Optional[float]
    p90_latency_ms: Optional[float]
    p99_latency_ms: Optional[float]


def search_documents(query: str, top_k: int = 8, document_filter: Optional[str] = None) -> list[dict]:
    """Search Pinecone using integrated inference."""
    search_params = {
        "namespace": "documents",
        "query": {
            "inputs": {"text": query},
            "top_k": top_k,
        },
        "fields": ["chunk_text", "document_name", "document_type", "source_file", "page_number", "folder"],
    }

    if document_filter:
        search_params["query"]["filter"] = {"document_name": {"$eq": document_filter}}

    results = index.search(**search_params)
    sources = []
    for hit in results.result.hits:
        fields = hit.fields or {}
        sources.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "text": fields.get("chunk_text", ""),
            "document_name": fields.get("document_name", "Unknown"),
            "document_type": fields.get("document_type", "Unknown"),
            "source_file": fields.get("source_file", ""),
            "page_number": fields.get("page_number", 0),
            "folder": fields.get("folder", ""),
        })
    return sources


def compute_percentile(values: list[float], percentile: float) -> Optional[float]:
    """Return percentile with linear interpolation."""
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    lower_value = sorted_values[lower]
    upper_value = sorted_values[upper]
    return lower_value + (upper_value - lower_value) * (position - lower)


def pinecone_search_latency_ms(query: str, top_k: int, document_filter: Optional[str]) -> float:
    """Execute one Pinecone search call and return latency in milliseconds."""
    search_params = {
        "namespace": "documents",
        "query": {
            "inputs": {"text": query},
            "top_k": top_k,
        },
        "fields": ["source_file"],
    }

    if document_filter:
        search_params["query"]["filter"] = {"document_name": {"$eq": document_filter}}

    started = time.perf_counter()
    index.search(**search_params)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return elapsed_ms


def build_context(sources: list[dict]) -> str:
    """Build a context string from retrieved sources for the LLM."""
    context_parts = []
    for i, src in enumerate(sources, 1):
        context_parts.append(
            f"[Source {i}] Document: {src['document_name']} | "
            f"Type: {src['document_type']} | "
            f"File: {src['source_file']} | "
            f"Page: {src['page_number']}\n"
            f"{src['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def get_mode_instruction(mode: str) -> str:
    if mode == "compare":
        return (
            "\n\nThe user wants to COMPARE documents or entities. "
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
            document_filter=request.document_filter,
        )

        if not sources:
            return ChatResponse(
                answer="I couldn't find any relevant information in the indexed documents for your query. Please try rephrasing or broadening your question.",
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
                "document_name": s["document_name"],
                "document_type": s["document_type"],
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
            namespace="documents",
            query={"inputs": {"text": "document summary report reference"}, "top_k": 100},
            fields=["document_name", "document_type", "source_file", "folder"],
        )
        seen = set()
        documents = []
        for hit in results.result.hits:
            fields = hit.fields or {}
            source_file = fields.get("source_file", "")
            if source_file and source_file not in seen:
                seen.add(source_file)
                documents.append({
                    "document_name": fields.get("document_name", "Unknown"),
                    "document_type": fields.get("document_type", "Unknown"),
                    "source_file": source_file,
                    "folder": fields.get("folder", "Root"),
                })
        documents.sort(key=lambda d: (d["folder"], d["document_name"]))
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
                "text": "What are the key goals and constraints described in these documents?",
                "icon": "shield",
            },
            {
                "text": "Summarise the main points covered across the document set",
                "icon": "target",
            },
            {
                "text": "Which sections describe compliance, policy, or governance requirements?",
                "icon": "leaf",
            },
            {
                "text": "Which document defines the core terms and key definitions?",
                "icon": "search",
            },
            {
                "text": "What deadlines, timelines, or milestone dates are specified?",
                "icon": "clock",
            },
            {
                "text": "What are the major risks, assumptions, or dependencies called out?",
                "icon": "globe",
            },
        ],
        "compare": [
            {
                "text": "Compare the scope and objectives of documents A and B",
                "icon": "scale",
            },
            {
                "text": "How do the requirement sections differ between these documents?",
                "icon": "pie-chart",
            },
            {
                "text": "Compare the risk statements across the selected documents",
                "icon": "shield",
            },
            {
                "text": "What changes between version 1 and version 2 of this policy?",
                "icon": "target",
            },
            {
                "text": "Compare responsibilities and ownership across the process documents",
                "icon": "globe",
            },
            {
                "text": "How do the acceptance criteria vary between the proposals?",
                "icon": "leaf",
            },
        ],
        "calculate": [
            {
                "text": "Add up the budget figures listed across these sections",
                "icon": "calculator",
            },
            {
                "text": "Calculate the percentage change between the two reported totals",
                "icon": "coins",
            },
            {
                "text": "What is the combined total count of items mentioned in these tables?",
                "icon": "trending-up",
            },
            {
                "text": "Compute the average value for the metrics listed in the document",
                "icon": "pie-chart",
            },
            {
                "text": "Estimate the monthly run rate based on the provided quarterly numbers",
                "icon": "globe",
            },
            {
                "text": "What is the difference between the minimum and maximum values reported?",
                "icon": "calculator",
            },
        ],
    }


@app.post("/api/load-test", response_model=LoadTestResponse)
async def load_test(request: LoadTestRequest):
    """Run Pinecone search load test at requested QPS and return latency stats."""
    if request.qps <= 0:
        raise HTTPException(status_code=400, detail="qps must be greater than 0")
    if request.duration_seconds <= 0:
        raise HTTPException(status_code=400, detail="duration_seconds must be greater than 0")
    if request.top_k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be greater than 0")

    total_requests = max(1, int(round(request.qps * request.duration_seconds)))
    interval_seconds = 1.0 / request.qps
    latencies: list[float] = []
    failed_requests = 0
    tasks: list[asyncio.Task] = []

    async def run_single_call() -> Optional[float]:
        try:
            return await asyncio.to_thread(
                pinecone_search_latency_ms,
                request.query,
                request.top_k,
                request.document_filter,
            )
        except Exception:
            return None

    started = time.perf_counter()

    for i in range(total_requests):
        target_time = started + (i * interval_seconds)
        now = time.perf_counter()
        sleep_for = target_time - now
        if sleep_for > 0:
            await asyncio.sleep(sleep_for)
        tasks.append(asyncio.create_task(run_single_call()))

    results = await asyncio.gather(*tasks)
    elapsed_seconds = max(time.perf_counter() - started, 1e-9)

    for latency in results:
        if latency is None:
            failed_requests += 1
        else:
            latencies.append(latency)

    successful_requests = len(latencies)
    qps_achieved = total_requests / elapsed_seconds
    avg_latency = (sum(latencies) / successful_requests) if successful_requests else None

    return LoadTestResponse(
        query=request.query,
        qps_target=request.qps,
        qps_achieved=qps_achieved,
        duration_seconds=request.duration_seconds,
        total_requests=total_requests,
        successful_requests=successful_requests,
        failed_requests=failed_requests,
        avg_latency_ms=avg_latency,
        p50_latency_ms=compute_percentile(latencies, 0.50),
        p90_latency_ms=compute_percentile(latencies, 0.90),
        p99_latency_ms=compute_percentile(latencies, 0.99),
    )


app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
