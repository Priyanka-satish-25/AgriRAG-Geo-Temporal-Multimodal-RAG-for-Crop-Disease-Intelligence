"""
Fine-Tuning the Cross-Encoder on Agricultural Domain QA Pairs
=============================================================
Adapts `cross-encoder/ms-marco-MiniLM-L-6-v2` to the agricultural retrieval
domain using (query, positive, hard_negative) triplets.

Key design decisions:
  - Hard negatives are BM25-mined (top-K non-relevant passages), which are
    more challenging than random negatives and produce a better-trained model.
  - Training uses sentence-transformers CrossEncoder.fit() with BCE loss.
  - Evaluation uses CERerankingEvaluator (MRR@10), run every 500 steps.
  - Best model is checkpointed automatically; used as drop-in in retrieval.py.

Usage:
    # Step 1 — generate synthetic pairs (or provide real AgriQA data)
    python finetune_crossencoder.py --generate-data --n-samples 500

    # Step 2 — mine hard negatives from a corpus
    python finetune_crossencoder.py --mine-negatives --corpus-path data/raw/plantvillage

    # Step 3 — train
    python finetune_crossencoder.py --train
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import bm25s
import torch
from sentence_transformers import CrossEncoder, InputExample
from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
from torch.utils.data import DataLoader


# ── Configuration ──────────────────────────────────────────────────────────────

BASE_MODEL      = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OUTPUT_PATH     = "./models/agri_crossencoder"
TRAIN_DATA_PATH = "data/finetune/agri_qa_train.json"
EVAL_DATA_PATH  = "data/finetune/agri_qa_eval.json"

TRAIN_CONFIG = dict(
    num_epochs        = 3,
    batch_size        = 16,
    warmup_steps      = 100,
    learning_rate     = 2e-5,
    max_length        = 512,
    evaluation_steps  = 500,
)


# ── Data Schema ────────────────────────────────────────────────────────────────

@dataclass
class TrainingPair:
    query:         str
    positive:      str
    hard_negative: str


# ── Synthetic Data Generator ───────────────────────────────────────────────────

CROPS    = ["wheat", "rice", "tomato", "potato", "maize", "cotton", "soybean", "chickpea", "groundnut"]
DISEASES = ["early blight", "late blight", "powdery mildew", "yellow rust", "stem rust",
            "bacterial leaf blight", "downy mildew", "leaf curl virus", "rice blast", "smut"]
SEASONS  = ["kharif", "rabi", "summer"]

TEMPLATES = [
    {
        "query":    "What causes {symptom} on {crop} leaves?",
        "positive": (
            "{disease} on {crop} causes {symptom}. The pathogen thrives under warm and humid conditions. "
            "Management includes removal of infected debris, application of {fungicide} fungicide "
            "at 7-10 day intervals, and selection of resistant varieties. Early detection is critical "
            "to prevent crop losses exceeding 30-40%."
        ),
        "hard_neg": (
            "{crop} cultivation requires well-drained soil with pH 6.0-7.0. "
            "Optimal planting density is 45×30 cm with irrigation at 5-7 day intervals. "
            "The {season} season crop typically matures in 90-120 days after sowing."
        ),
        "vars": {
            "symptom":   ["circular brown lesions", "yellow chlorotic spots", "dark water-soaked patches",
                          "orange pustules", "white powdery growth", "necrotic angular spots"],
            "fungicide": ["copper-based", "mancozeb", "propiconazole", "metalaxyl-M", "tebuconazole"],
        }
    },
    {
        "query":    "How to treat {disease} in {crop} during {season} season?",
        "positive": (
            "{disease} in {crop} during {season} is managed through integrated pest management. "
            "Apply {fungicide} fungicide preventively before disease onset. "
            "Ensure 40-50 cm plant spacing for air circulation to reduce humidity. "
            "Destroy infected plant material and avoid overhead irrigation. "
            "Crop rotation with non-host species for 2-3 years is recommended."
        ),
        "hard_neg": (
            "{crop} requires a base fertiliser of 80:40:40 NPK kg/ha applied at sowing. "
            "Top dressing with 40 kg urea/ha at 30 days after sowing improves yield. "
            "Irrigation at critical stages — flowering and grain filling — is essential."
        ),
        "vars": {
            "fungicide": ["copper oxychloride", "chlorothalonil", "azoxystrobin", "cymoxanil"],
        }
    },
    {
        "query":    "What are the symptoms of {disease} on {crop}?",
        "positive": (
            "{disease} on {crop} presents as distinct visual symptoms: {symptom}. "
            "The progression starts from lower older leaves moving upward. "
            "Under high disease pressure, defoliation occurs within 2-3 weeks. "
            "Diagnosis can be confirmed by laboratory isolation of the causal organism."
        ),
        "hard_neg": (
            "{crop} is an important {season} season crop cultivated across 12 million hectares. "
            "Average national yield is 2.5 tonnes/ha with improved varieties yielding up to 4.5 t/ha. "
            "Major producing states include Punjab, Haryana, Uttar Pradesh, and Madhya Pradesh."
        ),
        "vars": {
            "symptom": [
                "diamond-shaped grey lesions with brown borders",
                "concentric ring spots resembling a target board",
                "angular water-soaked lesions turning brown-black",
                "yellowing with parallel rows of orange spore pustules",
                "whitish fungal growth on upper leaf surface",
            ]
        }
    },
    {
        "query":    "Which fungicide is effective against {disease} in {crop}?",
        "positive": (
            "For {disease} in {crop}, {fungicide} applied at 1-2 ml/litre has shown 75-85% "
            "efficacy in field trials. Alternating between two fungicide groups every spray "
            "cycle prevents resistance development. The first spray at disease appearance "
            "followed by repeat applications at 10-14 day intervals is the recommended protocol. "
            "Systemic fungicides like {fungicide} penetrate plant tissue for curative action."
        ),
        "hard_neg": (
            "{crop} seeds should be treated with Thiram 3g/kg or Carbendazim 2g/kg before sowing. "
            "Seed priming in 1% KH₂PO₄ solution for 8 hours improves germination under drought. "
            "Plant growth regulators at vegetative stage improve biomass accumulation."
        ),
        "vars": {
            "fungicide": ["propiconazole", "tebuconazole", "tricyclazole", "mancozeb", "metalaxyl"],
        }
    },
    {
        "query":    "How does weather affect {disease} spread in {crop}?",
        "positive": (
            "{disease} in {crop} is strongly influenced by weather conditions. "
            "High humidity (>85%), frequent rainfall, and temperatures of 18-25°C create "
            "ideal conditions for infection and rapid spread. Leaf wetness duration exceeding "
            "6 hours after rain events triggers mass sporulation. Disease forecasting models "
            "use temperature-humidity indices to predict outbreak risk and guide spray timing."
        ),
        "hard_neg": (
            "Soil health management for {crop} includes deep ploughing to 30 cm before {season} sowing. "
            "Application of 10 tonnes/ha FYM or compost improves soil organic carbon. "
            "Subsoil compaction should be broken using chisel plough at 2-year intervals."
        ),
        "vars": {}
    }
]


def _fill_template(template: dict, crop: str, disease: str, season: str) -> TrainingPair:
    """Fills a template with random crop/disease/season values."""
    def pick(key): return random.choice(template["vars"].get(key, [""]))
    symptom   = pick("symptom")
    fungicide = pick("fungicide")
    fmt = dict(crop=crop, disease=disease, season=season, symptom=symptom, fungicide=fungicide)
    return TrainingPair(
        query         = template["query"].format(**fmt),
        positive      = template["positive"].format(**fmt),
        hard_negative = template["hard_neg"].format(**fmt),
    )


def generate_synthetic_pairs(n_samples: int = 500) -> tuple[list[dict], list[dict]]:
    """
    Generates domain-specific synthetic training pairs via template expansion.

    In a real project, replace/augment with:
      - AgriQA dataset (Harvested from extension bulletins)
      - CABI Crop Protection Compendium Q&A
      - PlantVillage community expert answers
    """
    pairs: list[dict] = []
    for _ in range(n_samples):
        tmpl    = random.choice(TEMPLATES)
        crop    = random.choice(CROPS)
        disease = random.choice(DISEASES)
        season  = random.choice(SEASONS)
        pair    = _fill_template(tmpl, crop, disease, season)
        pairs.append({
            "query":         pair.query,
            "positive":      pair.positive,
            "hard_negative": pair.hard_negative,
        })

    random.shuffle(pairs)
    split       = int(0.8 * len(pairs))
    train_pairs = pairs[:split]
    eval_pairs  = pairs[split:]

    out = Path("data/finetune")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "agri_qa_train.json", "w") as f: json.dump(train_pairs, f, indent=2)
    with open(out / "agri_qa_eval.json",  "w") as f: json.dump(eval_pairs,  f, indent=2)

    print(f"[DataGen] {len(train_pairs)} train pairs, {len(eval_pairs)} eval pairs written to data/finetune/")
    return train_pairs, eval_pairs


# ── Hard Negative Mining ───────────────────────────────────────────────────────

def mine_bm25_hard_negatives(
    qa_pairs_path: str,
    corpus_texts:  list[str],
    top_k:         int = 20,
    n_negatives:   int = 1,
) -> list[dict]:
    """
    BM25 Hard Negative Mining Strategy
    ───────────────────────────────────
    For each (query, positive) pair:
      1. Retrieve top-K documents using BM25.
      2. Remove the known positive from the results.
      3. Sample n_negatives from the top of the BM25 list.

    These BM25-top results are semantically plausible but not relevant —
    exactly the challenging cases that most improve cross-encoder training.

    This strategy is analogous to the hard negative mining used in:
    "Dense Passage Retrieval for Open-Domain QA" (Karpukhin et al., 2020)
    """
    with open(qa_pairs_path) as f:
        qa_pairs: list[dict] = json.load(f)

    print(f"[HardNeg] Building BM25 index over {len(corpus_texts)} corpus texts...")
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords="en")
    retriever     = bm25s.BM25()
    retriever.index(corpus_tokens)

    augmented: list[dict] = []
    for item in qa_pairs:
        q_tokens = bm25s.tokenize(item["query"], stopwords="en")
        results, _scores = retriever.retrieve(q_tokens, k=min(top_k, len(corpus_texts)))

        # Exclude the true positive
        candidates = [
            corpus_texts[idx] for idx in results[0]
            if corpus_texts[idx].strip() != item["positive"].strip()
        ]

        # Sample hard negatives from top of BM25 list (most confusable)
        sampled_negs = candidates[:n_negatives]
        for neg in sampled_negs:
            augmented.append({
                "query":         item["query"],
                "positive":      item["positive"],
                "hard_negative": neg,
            })

    # Overwrite with BM25-mined negatives
    out_path = Path(qa_pairs_path)
    with open(out_path, "w") as f:
        json.dump(augmented, f, indent=2)

    print(f"[HardNeg] Wrote {len(augmented)} BM25-mined pairs to {qa_pairs_path}")
    return augmented


# ── Trainer ────────────────────────────────────────────────────────────────────

class AgriculturalCrossEncoderTrainer:
    """
    Fine-tunes CrossEncoder on agricultural (query, passage, label) samples.

    Architecture note: ms-marco-MiniLM-L-6-v2 uses a BERT-style dual-input
    sequence [CLS] query [SEP] passage [SEP] → sigmoid → relevance score.
    Fine-tuning updates all transformer layers on the agricultural domain,
    improving both domain vocabulary and retrieval ranking quality.
    """

    def __init__(self, cfg: dict = TRAIN_CONFIG):
        self.cfg = cfg
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Trainer] Device: {device}")

        self.model = CrossEncoder(
            BASE_MODEL,
            num_labels  = 1,
            max_length  = cfg["max_length"],
            device      = device,
        )

    def load_train_samples(self, path: str) -> list[InputExample]:
        with open(path) as f:
            raw: list[dict] = json.load(f)

        samples: list[InputExample] = []
        for item in raw:
            # Positive pair: label = 1.0
            samples.append(InputExample(texts=[item["query"], item["positive"]],      label=1.0))
            # Hard negative pair: label = 0.0
            samples.append(InputExample(texts=[item["query"], item["hard_negative"]], label=0.0))

        random.shuffle(samples)
        print(f"[Trainer] Loaded {len(samples)} training examples ({len(raw)} QA pairs × 2).")
        return samples

    def build_evaluator(self, path: str) -> CERerankingEvaluator:
        """
        CERerankingEvaluator computes MRR@10 on the held-out set.
        Each query has one positive and one or more negatives.
        """
        with open(path) as f:
            raw: list[dict] = json.load(f)

        samples_map: dict[str, dict] = {}
        for item in raw:
            qid = item["query"]
            if qid not in samples_map:
                samples_map[qid] = {"query": qid, "positive": set(), "negative": []}
            samples_map[qid]["positive"].add(item["positive"])
            samples_map[qid]["negative"].append(item["hard_negative"])

        for qid in samples_map:
            samples_map[qid]["positive"] = list(samples_map[qid]["positive"])  # Convert set to list
        
        samples_list = list(samples_map.values())
        print(f"[Trainer] Evaluator: {len(samples_list)} unique queries.")
        return CERerankingEvaluator(samples_list, name="agri-reranking-eval")

    def train(self):
        Path(OUTPUT_PATH).mkdir(parents=True, exist_ok=True)

        train_samples = self.load_train_samples(TRAIN_DATA_PATH)
        train_loader  = DataLoader(
            train_samples, shuffle=True, batch_size=self.cfg["batch_size"]
        )
        evaluator = self.build_evaluator(EVAL_DATA_PATH)

        total_steps = len(train_loader) * self.cfg["num_epochs"]
        print(f"[Trainer] Training: {self.cfg['num_epochs']} epochs, {total_steps} steps, "
              f"LR={self.cfg['learning_rate']}, warmup={self.cfg['warmup_steps']}")

        self.model.fit(
            train_dataloader   = train_loader,
            evaluator          = evaluator,
            epochs             = self.cfg["num_epochs"],
            warmup_steps       = self.cfg["warmup_steps"],
            optimizer_params   = {"lr": self.cfg["learning_rate"]},
            output_path        = OUTPUT_PATH,
            evaluation_steps   = self.cfg["evaluation_steps"],
            save_best_model    = True,
            show_progress_bar  = True,
            use_amp            = torch.cuda.is_available(),  # FP16 mixed precision on GPU
        )
        print(f"[Trainer] Best model saved to {OUTPUT_PATH}")

    def evaluate_saved_model(self):
        print(f"[Trainer] Loading saved model from {OUTPUT_PATH} for evaluation...")
        model     = CrossEncoder(OUTPUT_PATH)
        evaluator = self.build_evaluator(EVAL_DATA_PATH)
        score     = evaluator(model)
        print(f"[Trainer] Final MRR@10 on eval set: {score:.4f}")
        return score


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriRAG Cross-Encoder Fine-Tuning")
    parser.add_argument("--generate-data",   action="store_true", help="Generate synthetic training pairs")
    parser.add_argument("--n-samples",       type=int, default=500)
    parser.add_argument("--mine-negatives",  action="store_true", help="Mine BM25 hard negatives")
    parser.add_argument("--corpus-path",     default="data/raw/plantvillage")
    parser.add_argument("--train",           action="store_true", help="Run fine-tuning")
    parser.add_argument("--evaluate",        action="store_true", help="Evaluate saved checkpoint")
    args = parser.parse_args()

    if args.generate_data:
        generate_synthetic_pairs(n_samples=args.n_samples)

    if args.mine_negatives:
        # Load corpus texts from the raw data directory
        import glob
        corpus: list[str] = []
        for jf in glob.glob(f"{args.corpus_path}/**/*.json", recursive=True):
            with open(jf) as f:
                records = json.load(f)
                if isinstance(records, dict): records = [records]
                corpus.extend([f"{r.get('title','')}. {r.get('content','')}" for r in records])
        mine_bm25_hard_negatives(TRAIN_DATA_PATH, corpus)

    if args.train:
        trainer = AgriculturalCrossEncoderTrainer()
        trainer.train()

    if args.evaluate:
        trainer = AgriculturalCrossEncoderTrainer()
        trainer.evaluate_saved_model()

    if not any([args.generate_data, args.mine_negatives, args.train, args.evaluate]):
        print("No action specified. Use --generate-data, --mine-negatives, --train, or --evaluate.")
        parser.print_help()
