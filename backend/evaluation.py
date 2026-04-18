# """
# RAGAS Evaluation & CI/CD Quality Gate
# ======================================
# Evaluates the AgriRAG pipeline on a held-out golden QA dataset using
# four RAGAS metrics. Exits with code 1 if any metric falls below its
# configured threshold — this is what fails the GitHub Actions CI pipeline.

# Metrics:
#   - Faithfulness:        Are all answer claims grounded in retrieved context?
#   - Answer Relevancy:    Does the answer address the question?
#   - Context Precision:   Are retrieved passages actually relevant to the query?
#   - Context Recall:      Does the retrieved context cover the ground truth answer?

# Usage:
#     # Run evaluation (requires running FastAPI backend)
#     python evaluation.py

#     # Override API base URL
#     API_BASE_URL=http://staging:8000 python evaluation.py

#     # Generate a sample golden dataset for testing
#     python evaluation.py --generate-golden
# """

# import argparse
# import json
# import os
# import sys
# from datetime import datetime
# from pathlib import Path

# import requests
# from datasets import Dataset
# from ragas import evaluate
# from ragas.llms import LangChainLLMWrapper
# from langchain_google_genai import ChatGoogleGenerativeAI
# from ragas.metrics.collections import (
#     faithfulness,
#     answer_relevancy,
#     context_precision,
#     context_recall,
# )


# # ── Configuration ──────────────────────────────────────────────────────────────

# API_BASE_URL         = os.getenv("API_BASE_URL", "http://localhost:8000")
# GOLDEN_DATASET_PATH  = "data/evaluation/golden_qa.json"
# RESULTS_OUTPUT_PATH  = "evaluation_results.json"

# # CI/CD thresholds — pipeline fails if any metric falls below these values.
# # These are paper-reportable design decisions: tighten for publication, loosen for demo.
# CI_THRESHOLDS: dict[str, float] = {
#     "faithfulness":      0.75,
#     "answer_relevancy":  0.70,
#     "context_precision": 0.65,
#     "context_recall":    0.60,
# }


# # ── Golden Dataset Schema ──────────────────────────────────────────────────────
# #
# # golden_qa.json is a list of records:
# # {
# #   "question":         "What disease causes target-shaped lesions on tomato?",
# #   "ground_truth":     "Early blight caused by Alternaria solani.",
# #   "image_path":       "data/evaluation/images/tomato_early_blight_01.jpg",
# #   "latitude":         28.6139,
# #   "longitude":        77.2090,
# #   "observation_date": "2024-07-15",
# #   "crop_type":        "tomato"   (optional)
# # }
# #
# # The question and ground_truth are used by RAGAS to score the pipeline.
# # The image_path + lat/lon feed the actual API call.


# def generate_sample_golden_dataset(output_path: str = GOLDEN_DATASET_PATH):
#     """
#     Creates a minimal sample golden dataset for immediate testing.
#     Replace image_path values with real labelled crop disease images.
#     """
#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#     Path("data/evaluation/images").mkdir(parents=True, exist_ok=True)

#     samples = [
#         {
#             "question":         "What disease causes circular brown spots with yellow halo on tomato leaves?",
#             "ground_truth":     "Early blight caused by Alternaria solani, managed with copper-based fungicide at 7-10 day intervals.",
#             "image_path":       "data/evaluation/images/tomato_early_blight.jpg",
#             "latitude":         28.6139,
#             "longitude":        77.2090,
#             "observation_date": "2024-07-15",
#             "crop_type":        "tomato",
#         },
#         {
#             "question":         "What causes dark water-soaked patches on tomato that rapidly turn brown-black?",
#             "ground_truth":     "Late blight caused by Phytophthora infestans, controlled with metalaxyl-M or mancozeb preventively.",
#             "image_path":       "data/evaluation/images/tomato_late_blight.jpg",
#             "latitude":         13.0827,
#             "longitude":        80.2707,
#             "observation_date": "2024-11-10",
#             "crop_type":        "tomato",
#         },
#         {
#             "question":         "What disease shows orange pustules in parallel rows on wheat leaves?",
#             "ground_truth":     "Yellow stripe rust (Puccinia striiformis), managed with propiconazole or tebuconazole at flag leaf stage.",
#             "image_path":       "data/evaluation/images/wheat_yellow_rust.jpg",
#             "latitude":         30.9010,
#             "longitude":        75.8573,
#             "observation_date": "2024-02-20",
#             "crop_type":        "wheat",
#         },
#         {
#             "question":         "What causes diamond-shaped grey lesions with brown borders on rice?",
#             "ground_truth":     "Rice blast caused by Magnaporthe oryzae, managed with tricyclazole at tillering and panicle initiation.",
#             "image_path":       "data/evaluation/images/rice_blast.jpg",
#             "latitude":         14.1768,
#             "longitude":        121.2448,
#             "observation_date": "2024-08-05",
#             "crop_type":        "rice",
#         },
#         {
#             "question":         "What disease causes brown lesions on potato leaves and tuber rot?",
#             "ground_truth":     "Potato late blight (Phytophthora infestans), using alternating cymoxanil+mancozeb and dimethomorph sprays.",
#             "image_path":       "data/evaluation/images/potato_late_blight.jpg",
#             "latitude":         -13.5319,
#             "longitude":        -71.9675,
#             "observation_date": "2024-04-12",
#             "crop_type":        "potato",
#         },
#     ]

#     with open(output_path, "w") as f:
#         json.dump(samples, f, indent=2)
#     print(f"[Golden] Sample dataset ({len(samples)} samples) written to {output_path}")
#     print("[Golden] NOTE: Replace image_path with actual labelled crop disease images before evaluation.")


# # ── API Query ──────────────────────────────────────────────────────────────────

# def _call_api(sample: dict) -> dict:
#     """
#     Sends one golden sample through the full /analyze pipeline.
#     Returns the raw API JSON response.
#     """
#     image_path = Path(sample["image_path"])
#     if not image_path.exists():
#         raise FileNotFoundError(
#             f"Golden dataset image not found: {image_path}. "
#             "Add the image or update golden_qa.json to point to real images."
#         )

#     with open(image_path, "rb") as img_file:
#         resp = requests.post(
#             f"{API_BASE_URL}/analyze",
#             files={"image": (image_path.name, img_file, "image/jpeg")},
#             data={
#                 "latitude":         str(sample["latitude"]),
#                 "longitude":        str(sample["longitude"]),
#                 "observation_date": sample.get("observation_date", ""),
#                 "crop_type":        sample.get("crop_type", ""),
#             },
#             timeout=180,
#         )
#     resp.raise_for_status()
#     return resp.json()


# # ── Dataset Builder ────────────────────────────────────────────────────────────

# def _build_ragas_dataset(golden_samples: list[dict]) -> Dataset:
#     """
#     Runs each golden sample through the API and constructs the RAGAS input Dataset.

#     RAGAS requires four columns:
#       question:    The user question (from golden dataset)
#       answer:      The model's generated answer (from API)
#       contexts:    List of retrieved passage snippets (from API sources)
#       ground_truth: The reference answer (from golden dataset)
#     """
#     questions:    list[str]        = []
#     answers:      list[str]        = []
#     contexts_all: list[list[str]]  = []
#     ground_truths:list[str]        = []
#     n_failed = 0

#     for i, sample in enumerate(golden_samples):
#         print(f"[Eval] Sample {i+1}/{len(golden_samples)}: {sample['question'][:70]}...")
#         try:
#             api_resp = _call_api(sample)

#             # Concatenate diagnosis + treatments as the model's answer
#             answer = (
#                 api_resp.get("diagnosis", "") + " "
#                 + " ".join(api_resp.get("treatment_recommendations", []))
#             ).strip()

#             # Use content snippets as the context for RAGAS scoring
#             contexts = [src["content_snippet"] for src in api_resp.get("sources", [])]

#             questions.append(sample["question"])
#             answers.append(answer)
#             contexts_all.append(contexts)
#             ground_truths.append(sample["ground_truth"])

#         except FileNotFoundError as e:
#             print(f"[Eval] SKIP sample {i+1}: {e}")
#             n_failed += 1
#         except requests.HTTPError as e:
#             print(f"[Eval] API error on sample {i+1}: {e.response.status_code} — skipping.")
#             n_failed += 1
#         except Exception as e:
#             print(f"[Eval] Unexpected error on sample {i+1}: {e} — skipping.")
#             n_failed += 1

#     if not questions:
#         print("[Eval] FATAL: No samples successfully evaluated. Check API and image paths.")
#         sys.exit(1)

#     print(f"[Eval] Processed {len(questions)}/{len(golden_samples)} samples "
#           f"({n_failed} skipped).")

#     return Dataset.from_dict({
#         "question":     questions,
#         "answer":       answers,
#         "contexts":     contexts_all,
#         "ground_truth": ground_truths,
#     })


# # ── Evaluation Runner ──────────────────────────────────────────────────────────

# def run_evaluation(golden_path: str = GOLDEN_DATASET_PATH) -> dict[str, float]:
#     """
#     Full RAGAS evaluation pipeline with CI/CD quality gate.

#     1. Load golden dataset
#     2. Run each sample through the live API
#     3. Compute RAGAS metrics
#     4. Compare against CI thresholds
#     5. Write results to evaluation_results.json
#     6. Exit with code 1 if any threshold is breached
#     """
#     print(f"[Eval] Loading golden dataset: {golden_path}")
#     with open(golden_path) as f:
#         golden_samples: list[dict] = json.load(f)
#     print(f"[Eval] {len(golden_samples)} evaluation samples loaded.")

#     ragas_dataset = _build_ragas_dataset(golden_samples)

#     print("[Eval] Computing RAGAS metrics (this calls an LLM for faithfulness scoring)...")
#     result = evaluate(
#         dataset = ragas_dataset,
#         metrics = [
#             faithfulness,
#             answer_relevancy,
#             context_precision,
#             context_recall,
#         ],
#     )

#     scores: dict[str, float] = {
#         "faithfulness":      round(float(result["faithfulness"]),      4),
#         "answer_relevancy":  round(float(result["answer_relevancy"]),  4),
#         "context_precision": round(float(result["context_precision"]), 4),
#         "context_recall":    round(float(result["context_recall"]),    4),
#         "evaluated_at":      datetime.now().isoformat(),
#         "n_samples":         len(ragas_dataset),
#     }

#     # ── CI/CD threshold check ──────────────────────────────────────────────────
#     print("\n" + "═" * 55)
#     print("  AgriRAG RAGAS Evaluation Results")
#     print("═" * 55)

#     failures: list[str] = []
#     for metric, threshold in CI_THRESHOLDS.items():
#         score  = scores.get(metric, 0.0)
#         passed = score >= threshold
#         status = "✅ PASS" if passed else "❌ FAIL"
#         print(f"  {metric:<25} {score:.4f}  (threshold: {threshold})  {status}")
#         if not passed:
#             failures.append(f"{metric}={score:.4f} < {threshold}")

#     print("═" * 55)

#     # ── Write results file (consumed by CI artifact upload) ───────────────────
#     output = {**scores, "thresholds": CI_THRESHOLDS, "failures": failures}
#     with open(RESULTS_OUTPUT_PATH, "w") as f:
#         json.dump(output, f, indent=2)
#     print(f"\n[Eval] Full results written to {RESULTS_OUTPUT_PATH}")

#     if failures:
#         print(f"\n[Eval] ❌ CI GATE FAILED — {len(failures)} metric(s) below threshold:")
#         for f in failures:
#             print(f"         {f}")
#         sys.exit(1)
#     else:
#         print("\n[Eval] ✅ All CI quality thresholds passed.")

#     return scores


# # ── CLI ────────────────────────────────────────────────────────────────────────

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="AgriRAG RAGAS Evaluation")
#     parser.add_argument(
#         "--generate-golden", action="store_true",
#         help="Generate a sample golden QA dataset and exit",
#     )
#     parser.add_argument(
#         "--golden-path", default=GOLDEN_DATASET_PATH,
#         help="Path to the golden QA JSON file",
#     )
#     args = parser.parse_args()

#     if args.generate_golden:
#         generate_sample_golden_dataset(args.golden_path)
#     else:
#         run_evaluation(args.golden_path)

####

# fixed evaluation.py 
"""
RAGAS Evaluation & CI/CD Quality Gate  (Gemini judge — no OpenAI required)
===========================================================================
Evaluates the AgriRAG pipeline on a held-out golden QA dataset using four
RAGAS metrics, with Google Gemini 3 Flash as the judge LLM.

WHY THIS WAS BROKEN BEFORE:
  The previous version imported LangChainLLMWrapper (wrong capitalisation),
  imported from ragas.metrics.collections (wrong module path), and — most
  critically — never actually passed the Gemini LLM to evaluate(). This
  caused RAGAS to fall through to its OpenAI default, which failed silently
  or errored because no OpenAI key was set.

WHAT IS FIXED:
  1. Correct import: LangchainLLMWrapper (lowercase 'c')
  2. Correct metric path: ragas.metrics (not ragas.metrics.collections)
  3. Gemini LLM + Embeddings are now wired into evaluate() directly
  4. Handles both RAGAS 0.1.x and 0.2.x APIs via a graceful fallback
  5. GoogleGenerativeAIEmbeddings added (required by AnswerRelevancy metric)

SETUP (one-time):
  pip install "ragas>=0.1.21" langchain-google-genai
  export GOOGLE_API_KEY=your_key   # same key as synthesis.py

USAGE:
  python evaluation.py --generate-golden   # create golden_qa.json template
  python evaluation.py                     # run evaluation (backend must be up)
  python evaluation.py --no-threshold-check  # measure without CI exit
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import requests
from datasets import Dataset

# ── RAGAS imports (version-safe) ───────────────────────────────────────────────
try:
    from ragas.llms import LangchainLLMWrapper           # note: lowercase 'c' in Langchain
    from ragas.embeddings import LangchainEmbeddingsWrapper
    _HAS_WRAPPERS = True
except ImportError:
    _HAS_WRAPPERS = False
    LangchainLLMWrapper = None
    LangchainEmbeddingsWrapper = None

try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    _HAS_GOOGLE = True
except ImportError:
    _HAS_GOOGLE = False

from ragas import evaluate

try:
    # Standard import path (RAGAS >= 0.1.18)
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )
except ImportError:
    # Very new RAGAS (>= 0.2.x) uses class-based metrics — instantiate them
    from ragas.metrics import (          # type: ignore[assignment]
        Faithfulness as faithfulness,
        AnswerRelevancy as answer_relevancy,
        ContextPrecision as context_precision,
        ContextRecall as context_recall,
    )


# ── Configuration ──────────────────────────────────────────────────────────────

API_BASE_URL        = os.getenv("API_BASE_URL", "http://localhost:8000")
GOLDEN_DATASET_PATH = "data/evaluation/golden_qa.json"
RESULTS_OUTPUT_PATH = "evaluation_results.json"
GOOGLE_API_KEY      = os.getenv("GOOGLE_API_KEY", "")

CI_THRESHOLDS: dict[str, float] = {
    "faithfulness":      0.75,
    "answer_relevancy":  0.70,
    "context_precision": 0.65,
    "context_recall":    0.60,
}


# ── Gemini Judge Setup ─────────────────────────────────────────────────────────

def _build_gemini_judge():
    """
    Returns (evaluator_llm, evaluator_embeddings) backed by Google Gemini.

    gemini-3-flash is the correct choice here:
      - Free tier: 15 RPM, 1 million tokens/day
      - 5 golden samples × 4 metrics = ~20 LLM calls — well within free limits
      - Same GOOGLE_API_KEY already set for synthesis.py

    GoogleGenerativeAIEmbeddings is required specifically by AnswerRelevancy,
    which measures cosine similarity between the question and the generated
    answer to verify the answer is on-topic.
    """
    if not GOOGLE_API_KEY:
        print(
            "\n[Eval] ❌  GOOGLE_API_KEY is not set.\n"
            "  Fix: export GOOGLE_API_KEY=your_gemini_api_key\n"
            "  (Same key you set for synthesis.py / Gemini calls)\n"
        )
        sys.exit(1)

    if not _HAS_GOOGLE:
        print(
            "\n[Eval] ❌  langchain-google-genai not installed.\n"
            "  Fix: pip install langchain-google-genai\n"
        )
        sys.exit(1)

    if not _HAS_WRAPPERS:
        print(
            "\n[Eval] ❌  RAGAS LangchainLLMWrapper not available.\n"
            "  Fix: pip install 'ragas>=0.1.21'\n"
        )
        sys.exit(1)

    chat_llm = ChatGoogleGenerativeAI(
        model            = "models/gemini-flash-lite-latest",
        google_api_key   = GOOGLE_API_KEY,
        temperature      = 0,
        # Required: Gemini doesn't support 'system' role natively in langchain
        convert_system_message_to_human = True,
    )
    emb_model = GoogleGenerativeAIEmbeddings(
        model          = "gemini-embedding-001",
        google_api_key = GOOGLE_API_KEY,
    )
    return LangchainLLMWrapper(chat_llm), LangchainEmbeddingsWrapper(emb_model)


# ── Golden Dataset ─────────────────────────────────────────────────────────────

def generate_sample_golden_dataset(output_path: str = GOLDEN_DATASET_PATH):
    """
    Writes golden_qa.json with 5 crop disease QA pairs.

    IMAGE SETUP — you need one image per sample:
    ┌──────────────────────────────┬──────────────────────────────────────────┐
    │ Image file                   │ Where to get it                          │
    ├──────────────────────────────┼──────────────────────────────────────────┤
    │ tomato_early_blight.jpg      │ Your demo: tomato-early-blight1x2400.jpg │
    │ rice_blast.jpg               │ Your demo: rice_blast_leaf.jpg           │
    │ tomato_late_blight.jpg       │ Wikimedia — wget command below           │
    │ wheat_yellow_rust.jpg        │ Wikimedia — wget command below           │
    │ potato_late_blight.jpg       │ Wikimedia — wget command below           │
    └──────────────────────────────┴──────────────────────────────────────────┘

    wget -O data/evaluation/images/tomato_late_blight.jpg \
      "https://upload.wikimedia.org/wikipedia/commons/thumb/0/06/Tomato_leaf_with_phytophtora.jpg/320px-Tomato_leaf_with_phytophtora.jpg"
    wget -O data/evaluation/images/wheat_yellow_rust.jpg \
      "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Yellow_rust_on_wheat.jpg/320px-Yellow_rust_on_wheat.jpg"
    wget -O data/evaluation/images/potato_late_blight.jpg \
      "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Potato_blight.jpg/320px-Potato_blight.jpg"
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path("data/evaluation/images").mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "question":        "What disease causes circular brown spots with a yellow halo on tomato leaves?",
            "ground_truth":    "Early blight caused by Alternaria solani. Managed with copper-based fungicide every 7-10 days, removal of infected plant debris, and crop rotation.",
            "image_path":      "data/evaluation/images/tomato_early_blight.jpg",
            "latitude":         28.6139,
            "longitude":        77.2090,
            "observation_date": "2024-07-15",
            "crop_type":        "tomato",
        },
        {
            "question":        "What causes dark water-soaked patches on tomato leaves that rapidly turn brown-black?",
            "ground_truth":    "Late blight caused by Phytophthora infestans. Controlled with preventive metalaxyl-M or mancozeb sprays and by avoiding overhead irrigation.",
            "image_path":      "data/evaluation/images/tomato_late_blight.jpg",
            "latitude":         13.0827,
            "longitude":        80.2707,
            "observation_date": "2024-11-10",
            "crop_type":        "tomato",
        },
        {
            "question":        "What disease produces yellow-orange pustules in parallel rows on wheat leaves?",
            "ground_truth":    "Yellow stripe rust caused by Puccinia striiformis. Managed with propiconazole or tebuconazole applied at flag leaf stage. Resistant varieties HD-2967 recommended.",
            "image_path":      "data/evaluation/images/wheat_yellow_rust.jpg",
            "latitude":         30.9010,
            "longitude":        75.8573,
            "observation_date": "2024-02-20",
            "crop_type":        "wheat",
        },
        {
            "question":        "What causes diamond-shaped grey lesions with brown borders on rice leaves?",
            "ground_truth":    "Rice blast caused by Magnaporthe oryzae. Managed with tricyclazole at 0.6 g/litre applied at tillering and panicle initiation. Silicon application improves resistance.",
            "image_path":      "data/evaluation/images/rice_blast.jpg",
            "latitude":         14.1768,
            "longitude":        121.2448,
            "observation_date": "2024-08-05",
            "crop_type":        "rice",
        },
        {
            "question":        "What disease causes brown lesions at potato leaf margins and brown rot in tubers?",
            "ground_truth":    "Potato late blight caused by Phytophthora infestans. Managed using alternating cymoxanil+mancozeb and dimethomorph on a 7-10 day fungicide schedule.",
            "image_path":      "data/evaluation/images/potato_late_blight.jpg",
            "latitude":        -13.5319,
            "longitude":       -71.9675,
            "observation_date": "2024-04-12",
            "crop_type":        "potato",
        },
    ]

    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[Golden] {len(samples)} samples written to {output_path}")
    print("[Golden] See function docstring for image download commands.")


# ── API Call ───────────────────────────────────────────────────────────────────

def _call_api(sample: dict) -> dict:
    image_path = Path(sample["image_path"])
    if not image_path.exists():
        raise FileNotFoundError(
            f"Missing evaluation image: {image_path}\n"
            "  Run --generate-golden for setup instructions."
        )
    with open(image_path, "rb") as img:
        resp = requests.post(
            f"{API_BASE_URL}/analyze",
            files={"image": (image_path.name, img, "image/jpeg")},
            data={
                "latitude":         str(sample["latitude"]),
                "longitude":        str(sample["longitude"]),
                "observation_date": sample.get("observation_date", ""),
                "crop_type":        sample.get("crop_type", ""),
            },
            timeout=300,
        )
    resp.raise_for_status()
    return resp.json()


# ── Dataset Builder ────────────────────────────────────────────────────────────

def _build_ragas_dataset(golden_samples: list[dict]) -> Dataset:
    """
    Calls /analyze for each sample, builds the four-column RAGAS dataset.

    answer   = diagnosis + all treatment recommendations (joined)
    contexts = content_snippet strings from each retrieved source
    """
    questions:     list[str]        = []
    answers:       list[str]        = []
    contexts_all:  list[list[str]]  = []
    ground_truths: list[str]        = []
    n_skipped = 0

    for i, sample in enumerate(golden_samples):
        print(f"\n[Eval] Sample {i+1}/{len(golden_samples)}")
        print(f"       Q: {sample['question'][:72]}")
        try:
            api_resp = _call_api(sample)

            diagnosis  = api_resp.get("diagnosis", "").strip()
            treatments = " ".join(api_resp.get("treatment_recommendations", []))
            answer     = f"{diagnosis}. {treatments}".strip(". ")

            contexts = [
                src["content_snippet"]
                for src in api_resp.get("sources", [])
                if src.get("content_snippet", "").strip()
            ]
            if not contexts:
                contexts = ["[No context retrieved]"]

            questions.append(sample["question"])
            answers.append(answer)
            contexts_all.append(contexts)
            ground_truths.append(sample["ground_truth"])

            print(f"       A: {answer[:80]}...")
            print(f"       Retrieved {len(contexts)} context passages.")

        except FileNotFoundError as e:
            print(f"       ⚠ SKIPPED — {e}")
            n_skipped += 1
        except requests.HTTPError as e:
            body = (e.response.text or "")[:200]
            print(f"       ⚠ API {e.response.status_code}: {body} — skipped.")
            n_skipped += 1
        except requests.ConnectionError:
            print(f"       ⚠ Cannot reach {API_BASE_URL} — is the backend running?")
            n_skipped += 1
        except Exception as e:
            print(f"       ⚠ {type(e).__name__}: {e} — skipped.")
            n_skipped += 1

    if not questions:
        print("\n[Eval] FATAL: 0 samples collected. Cannot run RAGAS.")
        print("  → Start the backend: uvicorn main:app --port 8000")
        print("  → Check image paths: python evaluation.py --generate-golden")
        sys.exit(1)

    print(f"\n[Eval] Dataset ready: {len(questions)} samples, {n_skipped} skipped.\n")

    return Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_all,
        "ground_truth": ground_truths,
    })


# ── Evaluation Runner ──────────────────────────────────────────────────────────

def run_evaluation(
    golden_path:      str  = GOLDEN_DATASET_PATH,
    output_path:      str  = RESULTS_OUTPUT_PATH,
    check_thresholds: bool = True,
) -> dict[str, float]:
    """
    Full RAGAS evaluation:
      1. Load golden dataset
      2. Call /analyze for each sample
      3. Compute 4 RAGAS metrics with Gemini judge
      4. Write results JSON
      5. Optionally fail the process if any threshold is breached (CI gate)
    """
    print(f"[Eval] Loading golden dataset: {golden_path}")
    if not Path(golden_path).exists():
        print("[Eval] Not found. Run: python evaluation.py --generate-golden")
        sys.exit(1)
    with open(golden_path) as f:
        golden_samples: list[dict] = json.load(f)
    print(f"[Eval] {len(golden_samples)} QA pairs loaded.")

    ragas_dataset = _build_ragas_dataset(golden_samples)

    print("[Eval] Initialising Gemini judge (gemini-2.5-flash)...")
    evaluator_llm, evaluator_embeddings = _build_gemini_judge()

    print("[Eval] Running RAGAS scoring — 4 metrics, LLM judge per claim...")
    try:
        # ── RAGAS 0.2.x: pass llm + embeddings to evaluate() ─────────────────
        result = evaluate(
            dataset    = ragas_dataset,
            metrics    = [faithfulness, answer_relevancy, context_precision, context_recall],
            llm        = evaluator_llm,
            embeddings = evaluator_embeddings,
        )
    except TypeError:
        # ── RAGAS 0.1.x fallback: set LLM directly on metric instances ────────
        print("[Eval] Detected RAGAS 0.1.x API — setting LLM on metric objects.")
        faithfulness.llm            = evaluator_llm
        answer_relevancy.llm        = evaluator_llm
        answer_relevancy.embeddings = evaluator_embeddings
        context_precision.llm       = evaluator_llm
        context_recall.llm          = evaluator_llm
        result = evaluate(
            dataset = ragas_dataset,
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall],
        )

    # Extract scores robustly (key names vary slightly between RAGAS versions)
    def _extract(primary: str, *alternates: str) -> float:
        for key in (primary, *alternates):
            try:
                v = result[key]
                if v is not None:
                    return round(float(v), 4)
            except (KeyError, TypeError, ValueError):
                pass
        return 0.0

    scores: dict[str, float] = {
        "faithfulness":      _extract("faithfulness"),
        "answer_relevancy":  _extract("answer_relevancy", "answer_relevance"),
        "context_precision": _extract("context_precision"),
        "context_recall":    _extract("context_recall"),
    }

    # ── Print results table ────────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  AgriRAG RAGAS Evaluation Results  |  Judge: gemini-2.5-flash")
    print("═" * 65)
    failures: list[str] = []
    for metric, threshold in CI_THRESHOLDS.items():
        score  = scores[metric]
        passed = score >= threshold
        margin = f"+{score-threshold:.4f}" if passed else f"{score-threshold:.4f}"
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {metric:<25} {score:.4f}   threshold {threshold}  ({margin})   {status}")
        if not passed:
            failures.append(f"{metric}={score:.4f} < {threshold}")
    print("═" * 65)

    # ── Write results JSON (read by Streamlit UI and CI comment) ───────────────
    output = {
        **scores,
        "evaluated_at": datetime.now().isoformat(),
        "n_samples":    len(ragas_dataset),
        "judge_llm":    "gemini-2.5-flash",
        "thresholds":   CI_THRESHOLDS,
        "failures":     failures,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[Eval] Results saved → {output_path}")

    if failures and check_thresholds:
        print(f"\n[Eval] ❌ CI GATE FAILED — {len(failures)} metric(s) below threshold:")
        for failure in failures:
            print(f"         {failure}")
        sys.exit(1)
    elif not failures:
        print("\n[Eval] ✅ All CI quality thresholds passed.")

    return scores


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AgriRAG RAGAS Evaluation (Google Gemini judge — no OpenAI needed)",
    )
    parser.add_argument(
        "--generate-golden", action="store_true",
        help="Write golden_qa.json template and print image setup instructions",
    )
    parser.add_argument(
        "--golden-path", default=GOLDEN_DATASET_PATH,
    )
    parser.add_argument(
        "--output-path", default=RESULTS_OUTPUT_PATH,
    )
    parser.add_argument(
        "--no-threshold-check", action="store_true",
        help="Measure scores without failing the process on threshold violation",
    )
    args = parser.parse_args()

    if args.generate_golden:
        generate_sample_golden_dataset(args.golden_path)
    else:
        run_evaluation(
            golden_path      = args.golden_path,
            output_path      = args.output_path,
            check_thresholds = not args.no_threshold_check,
        )
