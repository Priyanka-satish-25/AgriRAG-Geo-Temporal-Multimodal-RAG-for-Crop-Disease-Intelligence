"""
Agricultural Knowledge Base Ingestion Pipeline
==============================================
Indexes agricultural documents into Qdrant (dense) + bm25s (sparse),
attaching geo-temporal metadata as Qdrant payload fields for retrieval reranking.

Usage:
    python ingestion.py --data-dir data/raw/plantvillage --qdrant-url http://localhost:6333
"""

import argparse
import json
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import bm25s
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

COLLECTION_NAME = "agri_knowledge"
EMBEDDING_DIM = 1024          # BAAI/bge-large-en-v1.5 output dimension
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 64


# ── Data Model ─────────────────────────────────────────────────────────────────

@dataclass
class AgriculturalDocument:
    """
    Represents a single knowledge-base document with full geo-temporal metadata.

    Geo fields (latitude, longitude, region, country) power the novel
    geo-temporal reranking contribution: documents geographically proximate
    to the query image receive a score boost during reranking.

    Temporal fields (publication_date, season) implement temporal decay:
    recent documents are weighted higher than stale ones.
    """
    doc_id: str
    content: str
    title: str
    source: str
    # ── Geo metadata ──────────────────────────────────────────
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    region: Optional[str] = None
    country: Optional[str] = None
    # ── Temporal metadata ─────────────────────────────────────
    publication_date: Optional[datetime] = None
    season: Optional[str] = None          # 'kharif' | 'rabi' | 'zaid' | 'summer' | 'winter'
    # ── Domain metadata ───────────────────────────────────────
    crop_type: Optional[str] = None
    disease_name: Optional[str] = None
    keywords: list[str] = field(default_factory=list)


# ── Ingester ───────────────────────────────────────────────────────────────────

class AgriculturalKnowledgeBaseIngester:
    """
    Orchestrates dual-index ingestion:
      1. Qdrant — dense vector index with full payload for semantic search
      2. bm25s  — sparse keyword index for lexical matching
    Both indexes are required for the downstream RRF hybrid retrieval step.
    """

    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        bm25_index_dir: str = "data/bm25_index",
    ):
        print(f"[Ingestion] Connecting to Qdrant at {qdrant_url}...")
        self.qdrant = QdrantClient(url=qdrant_url, timeout=60.0)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.bm25_index_dir = Path(bm25_index_dir)
        self.bm25_index_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_collection()

    # ── Qdrant setup ──────────────────────────────────────────────────────────

    def _ensure_collection(self):
        existing = [c.name for c in self.qdrant.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            # Create indexed payload fields for fast metadata filtering during search
            self.qdrant.create_payload_index(COLLECTION_NAME, "crop_type",        PayloadSchemaType.KEYWORD)
            self.qdrant.create_payload_index(COLLECTION_NAME, "region",           PayloadSchemaType.KEYWORD)
            self.qdrant.create_payload_index(COLLECTION_NAME, "country",          PayloadSchemaType.KEYWORD)
            self.qdrant.create_payload_index(COLLECTION_NAME, "season",           PayloadSchemaType.KEYWORD)
            self.qdrant.create_payload_index(COLLECTION_NAME, "disease_name",     PayloadSchemaType.KEYWORD)
            self.qdrant.create_payload_index(COLLECTION_NAME, "publication_year", PayloadSchemaType.INTEGER)
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' created with payload indexes.")
        else:
            print(f"[Qdrant] Collection '{COLLECTION_NAME}' already exists — upserting.")

    # ── Embedding ─────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """
        Embeds text with BGE retrieval prefix for maximum retrieval performance.
        BGE models expect this prefix during indexing to align with query embeddings.
        """
        prefixed = [f"Represent this sentence for retrieval: {t}" for t in texts]
        all_embs: list[np.ndarray] = []
        for i in range(0, len(prefixed), BATCH_SIZE):
            batch = prefixed[i : i + BATCH_SIZE]
            embs = self.embedder.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_embs.append(embs)
        return np.vstack(all_embs)

    # ── Payload serialisation ─────────────────────────────────────────────────

    @staticmethod
    def _doc_to_payload(doc: AgriculturalDocument) -> dict:
        payload = {
            "doc_id":       doc.doc_id,
            "content":      doc.content,
            "title":        doc.title,
            "source":       doc.source,
            "crop_type":    doc.crop_type,
            "disease_name": doc.disease_name,
            "region":       doc.region,
            "country":      doc.country,
            "season":       doc.season,
            "keywords":     doc.keywords,
            # Raw coordinates stored as floats for haversine computation at query time
            "latitude":     doc.latitude,
            "longitude":    doc.longitude,
        }
        if doc.publication_date:
            payload["publication_date"]  = doc.publication_date.isoformat()
            payload["publication_year"]  = doc.publication_date.year
            payload["publication_month"] = doc.publication_date.month
        return payload

    # ── Public API ────────────────────────────────────────────────────────────

    def ingest_documents(self, documents: list[AgriculturalDocument]):
        """
        Main entry point. Performs:
          (a) Batch dense embedding → Qdrant upsert
          (b) Full BM25 index rebuild over entire corpus
        """
        if not documents:
            print("[Ingestion] No documents provided.")
            return

        print(f"[Ingestion] Embedding {len(documents)} documents with {EMBEDDING_MODEL}...")
        texts = [f"{doc.title}. {doc.content}" for doc in documents]
        embeddings = self._embed_batch(texts)

        # ── (a) Qdrant dense upsert ───────────────────────────────────────────
        points: list[PointStruct] = []
        for doc, emb in zip(documents, embeddings):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=self._doc_to_payload(doc),
            ))

        for i in tqdm(range(0, len(points), BATCH_SIZE), desc="Upserting to Qdrant"):
            self.qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i : i + BATCH_SIZE])
        print(f"[Qdrant] Upserted {len(points)} vectors.")

        # ── (b) BM25 index rebuild ────────────────────────────────────────────
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """
        Scrolls all Qdrant payloads and rebuilds the BM25 index from scratch.
        Called after every ingest to keep sparse and dense indexes in sync.
        """
        print("[BM25] Scrolling Qdrant for full corpus rebuild...")
        all_payloads = self._scroll_all_payloads()

        corpus_texts = [f"{p['title']}. {p['content']}" for p in all_payloads]
        print(f"[BM25] Indexing {len(corpus_texts)} documents...")

        corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)

        retriever.save(str(self.bm25_index_dir / "bm25_index"))
        doc_ids = [p["doc_id"] for p in all_payloads]
        with open(self.bm25_index_dir / "doc_ids.json", "w") as f:
            json.dump(doc_ids, f)
        with open(self.bm25_index_dir / "payloads.pkl", "wb") as f:
            pickle.dump(all_payloads, f)

        print(f"[BM25] Index persisted to {self.bm25_index_dir} ({len(corpus_texts)} docs).")

    def _scroll_all_payloads(self) -> list[dict]:
        all_payloads: list[dict] = []
        offset = None
        while True:
            results, next_offset = self.qdrant.scroll(
                collection_name=COLLECTION_NAME,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            all_payloads.extend([r.payload for r in results])
            if next_offset is None:
                break
            offset = next_offset
        return all_payloads


# ── Data Loaders ───────────────────────────────────────────────────────────────

def load_from_json_dir(data_dir: str) -> list[AgriculturalDocument]:
    """
    Loads documents from a directory of structured JSON files.

    Expected JSON record schema:
    {
      "id": "optional-string",
      "title": "Early Blight of Tomato",
      "content": "Early blight (Alternaria solani) causes...",
      "source": "PlantVillage / CABI / ICAR",
      "latitude": 28.6139,
      "longitude": 77.2090,
      "region": "North India",
      "country": "India",
      "publication_date": "2023-06-15",
      "season": "kharif",
      "crop_type": "tomato",
      "disease_name": "early_blight",
      "keywords": ["Alternaria solani", "fungicide", "brown lesions"]
    }
    """
    docs: list[AgriculturalDocument] = []
    data_path = Path(data_dir)
    json_files = list(data_path.glob("**/*.json"))
    print(f"[Loader] Found {len(json_files)} JSON files in {data_dir}")

    for json_file in json_files:
        with open(json_file) as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = [records]
        for rec in records:
            pub_date: Optional[datetime] = None
            if rec.get("publication_date"):
                try:
                    pub_date = datetime.fromisoformat(rec["publication_date"])
                except ValueError:
                    pass
            docs.append(AgriculturalDocument(
                doc_id          = rec.get("id", str(uuid.uuid4())),
                content         = rec["content"],
                title           = rec.get("title", ""),
                source          = rec.get("source", json_file.name),
                latitude        = rec.get("latitude"),
                longitude       = rec.get("longitude"),
                region          = rec.get("region"),
                country         = rec.get("country"),
                publication_date= pub_date,
                season          = rec.get("season"),
                crop_type       = rec.get("crop_type"),
                disease_name    = rec.get("disease_name"),
                keywords        = rec.get("keywords", []),
            ))

    print(f"[Loader] Loaded {len(docs)} documents total.")
    return docs


def create_sample_dataset(output_dir: str = "data/raw/plantvillage"):
    """
    Creates a minimal sample dataset for immediate testing.
    Replace with real PlantVillage / CABI / ICAR data for production.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "id": "doc_001", "title": "Tomato Early Blight (Alternaria solani)",
            "content": (
                "Early blight of tomato is caused by the fungus Alternaria solani. "
                "Symptoms appear as small, circular to angular brown spots on older leaves, "
                "often surrounded by a yellow halo. Lesions expand with concentric rings "
                "giving a target-board appearance. The disease thrives in warm (24-29°C), "
                "humid conditions. Management includes copper-based fungicide application "
                "every 7-10 days, removal of infected plant debris, and crop rotation. "
                "Resistant varieties such as Pusa Ruby show reduced susceptibility."
            ),
            "source": "ICAR Plant Pathology Bulletin 2023",
            "latitude": 28.6139, "longitude": 77.2090, "region": "Delhi NCR", "country": "India",
            "publication_date": "2023-04-10", "season": "kharif",
            "crop_type": "tomato", "disease_name": "early_blight",
            "keywords": ["Alternaria solani", "fungicide", "copper", "brown spots", "target lesion"],
        },
        {
            "id": "doc_002", "title": "Tomato Late Blight (Phytophthora infestans)",
            "content": (
                "Late blight, caused by Phytophthora infestans, is the most devastating "
                "tomato disease. Dark, water-soaked lesions appear on leaves and stems, "
                "rapidly turning brown-black. White fungal growth is visible on the underside "
                "of leaves in humid conditions. The pathogen spreads rapidly at 15-20°C with "
                "high humidity. Control requires preventive metalaxyl-M or mancozeb sprays, "
                "destruction of infected plants, and avoidance of overhead irrigation."
            ),
            "source": "CABI Crop Protection Compendium 2024",
            "latitude": 13.0827, "longitude": 80.2707, "region": "Tamil Nadu", "country": "India",
            "publication_date": "2024-02-15", "season": "rabi",
            "crop_type": "tomato", "disease_name": "late_blight",
            "keywords": ["Phytophthora infestans", "metalaxyl", "water-soaked", "mancozeb"],
        },
        {
            "id": "doc_003", "title": "Wheat Rust (Puccinia species) — Punjab Region",
            "content": (
                "Wheat rust encompasses three diseases — stem rust (P. graminis), "
                "leaf rust (P. triticina), and yellow/stripe rust (P. striiformis). "
                "Yellow rust is the most damaging in the Indo-Gangetic plains. "
                "Pustules of yellow-orange spores appear in parallel rows on leaves. "
                "Disease pressure peaks at 10-15°C with dew periods. Propiconazole "
                "and tebuconazole fungicides at flag leaf stage provide excellent control. "
                "Resistant varieties HD-2967 and DBW-187 are recommended for the region."
            ),
            "source": "Punjab Agricultural University Extension Report 2023",
            "latitude": 30.9010, "longitude": 75.8573, "region": "Punjab", "country": "India",
            "publication_date": "2023-12-01", "season": "rabi",
            "crop_type": "wheat", "disease_name": "yellow_rust",
            "keywords": ["Puccinia striiformis", "propiconazole", "tebuconazole", "flag leaf", "rust"],
        },
        {
            "id": "doc_004", "title": "Rice Blast (Magnaporthe oryzae) Management",
            "content": (
                "Rice blast caused by Magnaporthe oryzae is the most important rice disease "
                "worldwide. Diamond-shaped lesions with grey centres and brown borders appear "
                "on leaves (leaf blast) or at the neck (neck blast). Neck blast at heading "
                "causes complete yield loss. High nitrogen, dense planting, and intermittent "
                "drizzle favour epidemics. Tricyclazole at 0.6 g/litre applied at tillering "
                "and panicle initiation is the standard treatment. Silicon application improves "
                "resistance in silicon-deficient soils."
            ),
            "source": "International Rice Research Institute (IRRI) Fact Sheet 2024",
            "latitude": 14.1768, "longitude": 121.2448, "region": "Luzon", "country": "Philippines",
            "publication_date": "2024-06-20", "season": "kharif",
            "crop_type": "rice", "disease_name": "rice_blast",
            "keywords": ["Magnaporthe oryzae", "tricyclazole", "neck blast", "silicon", "IRRI"],
        },
        {
            "id": "doc_005", "title": "Potato Late Blight — Highland Regions Advisory",
            "content": (
                "Potato late blight (Phytophthora infestans) is especially severe in "
                "highland regions with cool, wet weather. Lesions start at leaf margins "
                "as dark water-soaked patches. Tuber infection causes brown rot rendering "
                "them unsaleable. The 7-10 day fungicide schedule should use alternating "
                "chemistry: cymoxanil+mancozeb, then dimethomorph, to prevent resistance. "
                "Certified seed potatoes, haulm destruction 2 weeks before harvest, and "
                "avoiding irrigation during susceptible periods are critical IPM measures."
            ),
            "source": "CIP (International Potato Center) Advisory 2024",
            "latitude": -13.5319, "longitude": -71.9675, "region": "Andes", "country": "Peru",
            "publication_date": "2024-03-01", "season": "summer",
            "crop_type": "potato", "disease_name": "late_blight",
            "keywords": ["Phytophthora infestans", "cymoxanil", "dimethomorph", "tuber rot", "CIP"],
        },
    ]
    with open(f"{output_dir}/sample_agri_docs.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[Sample] Written {len(samples)} sample documents to {output_dir}/sample_agri_docs.json")


# ── CLI Entry Point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriRAG Knowledge Base Ingestion")
    parser.add_argument("--data-dir",    default="data/raw/plantvillage", help="Path to JSON document directory")
    parser.add_argument("--qdrant-url",  default="http://localhost:6333",  help="Qdrant server URL")
    parser.add_argument("--bm25-dir",    default="data/bm25_index",        help="BM25 index output directory")
    parser.add_argument("--create-sample", action="store_true",            help="Generate sample dataset and exit")
    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset(args.data_dir)
        print("Sample dataset created. Run without --create-sample to ingest.")
    else:
        docs = load_from_json_dir(args.data_dir)
        ingester = AgriculturalKnowledgeBaseIngester(
            qdrant_url=args.qdrant_url,
            bm25_index_dir=args.bm25_dir,
        )
        ingester.ingest_documents(docs)
        print("[Ingestion] Complete.")
