"""
AgriRAG FastAPI Backend
=======================
Serves the full precision agriculture RAG pipeline as a REST API.

Endpoints:
  GET  /health        — Model + service health check
  POST /analyze       — Full pipeline: image → advisory report
  POST /evaluate      — Trigger RAGAS evaluation on golden dataset
  POST /admin/ingest  — Re-trigger knowledge base ingestion (admin)

Design principle: all heavy models (BLIP-2, CrossEncoder, embedding model)
are loaded ONCE at application startup via the lifespan context manager and
held as module-level singletons. Per-request loading would add 5-30 seconds
of latency per call — unacceptable for a demo system.

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import io
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field
from transformers import BlipForConditionalGeneration, BlipProcessor

import anthropic
from retrieval import HybridAgriculturalRetriever, RetrievedChunk
from synthesis import AdvisoryReport, synthesize_advisory


# ── Model Registry (singletons) ────────────────────────────────────────────────

class _ModelRegistry:
    """Holds all loaded models as class attributes. Avoids global variables."""
    blip_processor: Optional[BlipProcessor]               = None
    blip_model:     Optional[BlipForConditionalGeneration] = None
    retriever:      Optional[HybridAgriculturalRetriever]  = None
    anthropic_client: Optional[anthropic.Anthropic]        = None
    device: str = "cpu"
    ready:  bool = False


registry = _ModelRegistry()


# ── Lifespan: startup / shutdown ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads all models at startup. FastAPI calls this before serving requests.
    Using lifespan (not @app.on_event) is the recommended approach in FastAPI 0.93+.
    """
    t0 = time.perf_counter()
    registry.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Startup] Device: {registry.device}")

    # ── BLIP image captioning model ──────────────────────────────────────────
    blip_id = os.getenv("BLIP_MODEL_ID", "Salesforce/blip-image-captioning-large")
    print(f"[Startup] Loading BLIP from hub: {blip_id}")
    registry.blip_processor = BlipProcessor.from_pretrained(blip_id)
    registry.blip_model = BlipForConditionalGeneration.from_pretrained(
        blip_id,
        torch_dtype=torch.float16 if registry.device == "cuda" else torch.float32,
    ).to(registry.device)
    registry.blip_model.eval()
    print("[Startup] BLIP loaded.")

    # ── Hybrid retriever (loads Qdrant client + BM25 index + CrossEncoder) ───
    print("[Startup] Initialising retriever...")
    registry.retriever = HybridAgriculturalRetriever(
        qdrant_url    = os.getenv("QDRANT_URL",    "http://localhost:6333"),
        bm25_index_dir= os.getenv("BM25_INDEX_DIR","data/bm25_index"),
        dense_top_k   = int(os.getenv("DENSE_TOP_K",   "50")),
        bm25_top_k    = int(os.getenv("BM25_TOP_K",    "50")),
        rerank_top_k  = int(os.getenv("RERANK_TOP_K",  "20")),
        final_top_k   = int(os.getenv("FINAL_TOP_K",    "5")),
    )

    registry.ready = True
    elapsed = round((time.perf_counter() - t0) * 1000)
    print(f"[Startup] All models ready in {elapsed}ms.")
    yield
    print("[Shutdown] Releasing resources.")
    registry.ready = False


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title       = "AgriRAG — Precision Agriculture Intelligence",
    description = "Geo-temporal hybrid RAG for crop disease diagnosis and advisory.",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Pydantic response models ───────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    models_loaded: dict[str, bool]
    device:        str

class CitedClaimOut(BaseModel):
    claim:          str
    source_indices: list[int]

class SourceDocOut(BaseModel):
    index:            int
    title:            str
    source:           str
    region:           Optional[str]
    publication_date: Optional[str]
    geo_weight:       float
    temporal_weight:  float
    ce_normalized:    float
    final_score:      float
    content_snippet:  str

class AnalysisResponse(BaseModel):
    caption:                   str
    summary:                   str
    diagnosis:                 str
    confidence_level:          str
    treatment_recommendations: list[str]
    follow_up_actions:         list[str]
    cited_claims:              list[CitedClaimOut]
    sources:                   list[SourceDocOut]
    processing_metadata:       dict


# ── BLIP Caption Generation ────────────────────────────────────────────────────

def generate_caption(pil_image: Image.Image) -> str:
    """
    Runs BLIP-2 conditional image captioning.

    The prompt prefix "a photograph of a crop showing" steers the model toward
    agricultural descriptions (symptoms, colours, texture) rather than generic
    scene descriptions.
    """
    inputs = registry.blip_processor(
        images = pil_image,
        text   = "a photograph of a crop showing",
        return_tensors = "pt",
    ).to(registry.device)

    with torch.no_grad():
        output_ids = registry.blip_model.generate(
            **inputs,
            max_new_tokens = 128,
            num_beams      = 5,
            early_stopping = True,
        )
    return registry.blip_processor.decode(output_ids[0], skip_special_tokens=True)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status        = "healthy" if registry.ready else "loading",
        models_loaded = {
            "blip":      registry.blip_model is not None,
            "retriever": registry.retriever   is not None,
        },
        device = registry.device,
    )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_crop_image(
    image:            UploadFile = File(..., description="Crop photograph (JPG/PNG/WebP)"),
    latitude:         float      = Form(..., description="GPS latitude of image capture"),
    longitude:        float      = Form(..., description="GPS longitude of image capture"),
    observation_date: str        = Form(default="", description="ISO 8601 date string"),
    crop_type:        str        = Form(default="",  description="Known crop type (optional)"),
):
    """
    Full analysis pipeline:
    Image → BLIP-2 Caption → Hybrid RAG → Geo-Temporal Rerank → Claude Synthesis
    """
    if not registry.ready:
        raise HTTPException(status_code=503, detail="Models are still loading. Retry in a moment.")

    # ── Validate image ─────────────────────────────────────────────────────────
    if image.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(status_code=400, detail=f"Unsupported image type: {image.content_type}")

    raw_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not open image: {e}")

    timings: dict[str, int] = {}

    # ── Stage 1: BLIP-2 captioning ────────────────────────────────────────────
    t = time.perf_counter()
    caption = generate_caption(pil_image)
    timings["caption_ms"] = round((time.perf_counter() - t) * 1000)
    print(f"[Pipeline] Caption ({timings['caption_ms']}ms): {caption}")

    # ── Parse observation date ─────────────────────────────────────────────────
    if not observation_date:
        observation_date = datetime.now().isoformat()
    try:
        obs_datetime = datetime.fromisoformat(observation_date)
    except ValueError:
        obs_datetime = datetime.now()
        observation_date = obs_datetime.isoformat()

    crop_filter = crop_type.strip() or None

    # ── Stage 2–5: Hybrid RAG + geo-temporal reranking ────────────────────────
    t = time.perf_counter()
    retrieved: list[RetrievedChunk] = registry.retriever.retrieve(
        query            = caption,
        query_lat        = latitude,
        query_lon        = longitude,
        query_date       = obs_datetime,
        crop_type_filter = crop_filter,
    )
    timings["retrieval_ms"] = round((time.perf_counter() - t) * 1000)
    print(f"[Pipeline] Retrieved {len(retrieved)} chunks ({timings['retrieval_ms']}ms)")

    if not retrieved:
        raise HTTPException(
            status_code=404,
            detail="No relevant documents found. Check that the knowledge base is populated."
        )

    # ── Stage 6: Claude synthesis ──────────────────────────────────────────────
    t = time.perf_counter()
    report: AdvisoryReport = synthesize_advisory(
        caption          = caption,
        retrieved_chunks = retrieved,
        query_lat        = latitude,
        query_lon        = longitude,
        query_date       = observation_date,
        crop_type        = crop_filter,
    )
    timings["synthesis_ms"] = round((time.perf_counter() - t) * 1000)
    print(f"[Pipeline] Synthesis done ({timings['synthesis_ms']}ms). "
          f"Confidence: {report.confidence_level}")

    # ── Build response ─────────────────────────────────────────────────────────
    sources_out = [
        SourceDocOut(
            index            = i,
            title            = chunk.title,
            source           = chunk.source,
            region           = chunk.region,
            publication_date = chunk.publication_date,
            geo_weight       = round(chunk.geo_weight,      3),
            temporal_weight  = round(chunk.temporal_weight, 3),
            ce_normalized    = round(chunk.ce_normalized,   4),
            final_score      = round(chunk.final_score,     4),
            content_snippet  = chunk.content[:350] + "..." if len(chunk.content) > 350 else chunk.content,
        )
        for i, chunk in enumerate(retrieved)
    ]

    timings["total_ms"] = sum(timings.values())

    return AnalysisResponse(
        caption                   = caption,
        summary                   = report.summary,
        diagnosis                 = report.diagnosis,
        confidence_level          = report.confidence_level,
        treatment_recommendations = report.treatment_recommendations,
        follow_up_actions         = report.follow_up_actions,
        cited_claims              = [
            CitedClaimOut(claim=c.claim, source_indices=c.source_indices)
            for c in report.cited_claims
        ],
        sources               = sources_out,
        processing_metadata   = timings,
    )


@app.post("/evaluate")
async def trigger_evaluation():
    """Trigger RAGAS evaluation on the golden dataset. Used by CI/CD pipeline."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "evaluation.py"],
        capture_output=True, text=True,
    )
    return {
        "exit_code": result.returncode,
        "stdout":    result.stdout[-3000:],  # Tail to avoid large responses
        "stderr":    result.stderr[-1000:],
        "passed":    result.returncode == 0,
    }
