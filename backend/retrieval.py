"""
Hybrid Retrieval Pipeline with Geo-Temporal Reranking
=====================================================
This module implements the core novel contribution of the AgriRAG system:
geo-temporal score modulation on top of hybrid dense+sparse retrieval.

Full pipeline per query:
  1. Dense search  → Qdrant (BGE-large embeddings, top-50)
  2. Sparse search → bm25s keyword index (top-50)
  3. RRF fusion    → Reciprocal Rank Fusion (top-20 candidates)
  4. CE reranking  → Fine-tuned CrossEncoder (score all 20 pairs)
  5. Geo-temporal  → Haversine decay × temporal decay → final_score

The geo-temporal step (5) is the paper's novel contribution:
  final_score = sigmoid(CE_score) × geo_weight(d_km) × temporal_weight(Δ_days)

References:
  - Karpukhin et al. 2020: Dense Passage Retrieval (DPR)
  - Cormack et al. 2009: Reciprocal Rank Fusion (RRF)
  - Nogueira & Cho 2019: Passage Re-ranking with BERT (CrossEncoder)
"""

import json
import math
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import bm25s
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from sentence_transformers import CrossEncoder, SentenceTransformer


COLLECTION_NAME  = "agri_knowledge"
EMBEDDING_MODEL  = "BAAI/bge-large-en-v1.5"
FINETUNED_CE     = "./models/agri_crossencoder"
FALLBACK_CE      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BM25_INDEX_DIR   = "data/bm25_index"

# Geo-temporal decay constants (tunable hyperparameters — reported in paper)
GEO_DECAY_RADIUS_KM  = 500.0   # Distance at which geo weight = exp(-0.5) ≈ 0.607
TEMPORAL_DECAY_DAYS  = 730.0   # 2 years half-life for temporal decay
RRF_K                = 60       # Standard RRF k constant (Cormack et al. 2009)


# ── Data Model ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single retrieved document chunk with all intermediate scores for
    transparency, debugging, and RAGAS evaluation.
    """
    doc_id:           str
    content:          str
    title:            str
    source:           str
    crop_type:        Optional[str]
    disease_name:     Optional[str]
    region:           Optional[str]
    latitude:         Optional[float]
    longitude:        Optional[float]
    publication_date: Optional[str]
    # ── Pipeline scores ──────────────────────────────────────────
    dense_rank:      Optional[int] = None
    bm25_rank:       Optional[int] = None
    rrf_score:       float = 0.0    # After RRF fusion
    ce_score:        float = 0.0    # Raw CrossEncoder logit
    ce_normalized:   float = 0.0    # sigmoid(ce_score) → [0, 1]
    geo_weight:      float = 1.0    # Gaussian distance decay
    temporal_weight: float = 1.0    # Exponential time decay
    final_score:     float = 0.0    # ce_normalized × geo × temporal


# ── Geo-Temporal Scoring Functions ─────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance in kilometres between two GPS coordinates.
    Uses the Haversine formula (accurate to ~0.5% for distances < 1000 km).
    """
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi       = math.radians(lat2 - lat1)
    dlambda    = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def compute_geo_weight(
    doc_lat:          Optional[float],
    doc_lon:          Optional[float],
    query_lat:        float,
    query_lon:        float,
    decay_radius_km:  float = GEO_DECAY_RADIUS_KM,
) -> float:
    """
    Gaussian distance decay geo-weight.

    w_geo = exp(-0.5 × (d_km / r)²)

    where d_km is haversine distance and r is the decay radius.
    - At d=0 km:   w=1.000 (same location)
    - At d=500 km: w=0.607 (half-life radius)
    - At d=1000 km: w=0.135
    - At d→∞:      w→0

    Documents without coordinates receive a penalty weight of 0.80
    (mild penalty — they may still be relevant, just not geo-verified).
    """
    if doc_lat is None or doc_lon is None:
        return 0.80
    dist_km = haversine_km(query_lat, query_lon, doc_lat, doc_lon)
    return math.exp(-0.5 * (dist_km / decay_radius_km) ** 2)


def compute_temporal_weight(
    doc_date_str:  Optional[str],
    query_date:    datetime,
    decay_days:    float = TEMPORAL_DECAY_DAYS,
) -> float:
    """
    Exponential temporal recency weight.

    w_temporal = exp(-Δ_days / τ)

    where Δ_days is the absolute day difference and τ is the decay constant.
    - At Δ=0 days:   w=1.000 (same day)
    - At Δ=730 days: w=0.368 (2-year half-life)
    - At Δ=1460 days: w=0.135

    Documents without dates receive 0.75 (penalised but not excluded).
    """
    if not doc_date_str:
        return 0.75
    try:
        doc_date  = datetime.fromisoformat(doc_date_str)
        delta     = abs((query_date - doc_date).days)
        return math.exp(-delta / decay_days)
    except (ValueError, TypeError):
        return 0.75


# ── RRF Fusion ─────────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    dense_results: list[ScoredPoint],
    bm25_results:  list[tuple[str, float]],
    k:             int = RRF_K,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion (Cormack et al. 2009).

    RRF(d) = Σ_r  1 / (k + rank_r(d))

    Combines rankings from dense and sparse retrievers without requiring
    score normalisation. k=60 is the standard from the original paper.
    """
    scores: dict[str, float] = {}
    for rank, point in enumerate(dense_results):
        doc_id = point.payload["doc_id"]
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(bm25_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


# ── Main Retriever ─────────────────────────────────────────────────────────────

class HybridAgriculturalRetriever:
    """
    Orchestrates the full 5-stage retrieval pipeline.

    Stage 1 — Dense:    BGE-large embedding → Qdrant cosine search (top-50)
    Stage 2 — Sparse:   bm25s keyword match → ranked list (top-50)
    Stage 3 — RRF:      Score fusion → unified ranking (top-20 candidates)
    Stage 4 — CE:       Fine-tuned CrossEncoder relevance scoring
    Stage 5 — Geo-Tmp:  Multiply CE score by geo + temporal decay weights
    """

    def __init__(
        self,
        qdrant_url:    str = "http://localhost:6333",
        bm25_index_dir: str = BM25_INDEX_DIR,
        dense_top_k:   int = 50,
        bm25_top_k:    int = 50,
        rerank_top_k:  int = 20,
        final_top_k:   int = 5,
    ):
        self.qdrant       = QdrantClient(url=qdrant_url)
        self.embedder     = SentenceTransformer(EMBEDDING_MODEL)
        self.dense_top_k  = dense_top_k
        self.bm25_top_k   = bm25_top_k
        self.rerank_top_k = rerank_top_k
        self.final_top_k  = final_top_k

        # Load fine-tuned CE (fall back to pretrained if not yet trained)
        ce_path = Path(FINETUNED_CE)
        if ce_path.exists() and any(ce_path.iterdir()):
            print(f"[Retrieval] Loading fine-tuned CE: {FINETUNED_CE}")
            self.cross_encoder = CrossEncoder(str(ce_path))
        else:
            print(f"[Retrieval] Fine-tuned model not found. Falling back to: {FALLBACK_CE}")
            self.cross_encoder = CrossEncoder(FALLBACK_CE)

        # Load BM25 index and payload maps
        bm25_path = Path(bm25_index_dir)
        self.bm25_retriever = bm25s.BM25.load(str(bm25_path / "bm25_index"))
        with open(bm25_path / "doc_ids.json") as f:
            self.bm25_doc_ids: list[str] = json.load(f)
        with open(bm25_path / "payloads.pkl", "rb") as f:
            payloads: list[dict] = pickle.load(f)
        self._bm25_payload_map: dict[str, dict] = {p["doc_id"]: p for p in payloads}
        print(f"[Retrieval] BM25 loaded. {len(self.bm25_doc_ids)} indexed documents.")

    # ── Stage 1: Dense Search ──────────────────────────────────────────────────

    def _dense_search(self, query: str) -> list[ScoredPoint]:
        emb = self.embedder.encode(
            f"Represent this sentence for retrieval: {query}",
            normalize_embeddings=True,
        )
        return self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query          =emb.tolist(),
            limit          =self.dense_top_k,
            with_payload   =True,
        ).points

    # ── Stage 2: BM25 Sparse Search ───────────────────────────────────────────

    def _bm25_search(self, query: str) -> list[tuple[str, float]]:
        q_tokens       = bm25s.tokenize(query, stopwords="en")
        results, scores = self.bm25_retriever.retrieve(
            q_tokens, k=min(self.bm25_top_k, len(self.bm25_doc_ids))
        )
        return [
            (self.bm25_doc_ids[idx], float(scores[0][i]))
            for i, idx in enumerate(results[0])
        ]

    # ── Stage 4: CrossEncoder Reranking ───────────────────────────────────────

    def _cross_encode(self, query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Batch-score all (query, passage) pairs with the fine-tuned CE."""
        pairs  = [[query, c.content] for c in chunks]
        scores = self.cross_encoder.predict(pairs, show_progress_bar=False)
        for chunk, score in zip(chunks, scores):
            chunk.ce_score      = float(score)
            chunk.ce_normalized = 1.0 / (1.0 + math.exp(-chunk.ce_score))  # sigmoid
        return sorted(chunks, key=lambda c: c.ce_score, reverse=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query:             str,
        query_lat:         float,
        query_lon:         float,
        query_date:        datetime,
        crop_type_filter:  Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Execute the full 5-stage retrieval pipeline.

        Args:
            query:            Text query derived from BLIP-2 caption
            query_lat:        GPS latitude from image EXIF / user input
            query_lon:        GPS longitude from image EXIF / user input
            query_date:       Observation datetime for temporal decay
            crop_type_filter: Optional hard filter to restrict to one crop type

        Returns:
            Top-N RetrievedChunk objects sorted by final geo-temporal score.
        """

        # ── Stage 1+2: Parallel dense + sparse search ─────────────────────────
        dense_results = self._dense_search(query)
        bm25_results  = self._bm25_search(query)

        # ── Stage 3: RRF fusion ───────────────────────────────────────────────
        rrf_scores = reciprocal_rank_fusion(dense_results, bm25_results)

        # Collect payload map for all unique candidates
        payload_map: dict[str, dict] = {}
        for point in dense_results:
            payload_map[point.payload["doc_id"]] = point.payload
        for doc_id, _ in bm25_results:
            if doc_id not in payload_map and doc_id in self._bm25_payload_map:
                payload_map[doc_id] = self._bm25_payload_map[doc_id]

        # Sort by RRF score, keep top-rerank_top_k for CE
        sorted_ids = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)[: self.rerank_top_k]

        # Build RetrievedChunk objects; apply optional crop_type hard filter
        candidates: list[RetrievedChunk] = []
        for rank, doc_id in enumerate(sorted_ids):
            if doc_id not in payload_map:
                continue
            p = payload_map[doc_id]
            if crop_type_filter and p.get("crop_type") and p["crop_type"] != crop_type_filter:
                continue
            candidates.append(RetrievedChunk(
                doc_id           = doc_id,
                content          = p.get("content", ""),
                title            = p.get("title",   ""),
                source           = p.get("source",  ""),
                crop_type        = p.get("crop_type"),
                disease_name     = p.get("disease_name"),
                region           = p.get("region"),
                latitude         = p.get("latitude"),
                longitude        = p.get("longitude"),
                publication_date = p.get("publication_date"),
                rrf_score        = rrf_scores[doc_id],
            ))

        if not candidates:
            return []

        # ── Stage 4: Fine-tuned CrossEncoder reranking ────────────────────────
        candidates = self._cross_encode(query, candidates)

        # ── Stage 5: Geo-Temporal Score Modulation (Novel Contribution) ───────
        for chunk in candidates:
            chunk.geo_weight      = compute_geo_weight(
                chunk.latitude, chunk.longitude, query_lat, query_lon
            )
            chunk.temporal_weight = compute_temporal_weight(
                chunk.publication_date, query_date
            )
            # Combined score: semantically relevant + geographically proximate + recent
            chunk.final_score     = chunk.ce_normalized * chunk.geo_weight * chunk.temporal_weight

        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates[: self.final_top_k]
