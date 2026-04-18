"""
AgriRAG — Streamlit Frontend
============================
Professional demo UI for the Precision Agriculture & Crop Disease
Intelligence system. Interacts with the FastAPI backend at API_BASE_URL.

Run with:
    streamlit run app.py
"""

import json
from datetime import date, datetime
from typing import Optional

import requests
import streamlit as st
from PIL import Image
import io

API_BASE = "http://localhost:8000"

# ── Page configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title       = "AgriRAG — Crop Disease Intelligence",
    page_icon        = "🌾",
    layout           = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* === Layout === */
section[data-testid="stSidebar"] { background: #f0f5f0; border-right: 1px solid #d8e8d8; }
.main .block-container { padding-top: 1.5rem; }

/* === Confidence badges === */
.badge-high   { background:#d4edda; color:#155724; padding:5px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
.badge-medium { background:#fff3cd; color:#7d5a00; padding:5px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }
.badge-low    { background:#f8d7da; color:#721c24; padding:5px 14px; border-radius:20px; font-weight:700; font-size:13px; display:inline-block; }

/* === Citation pill === */
.cite-pill {
    background: #e3f0fb; color: #1565c0;
    border-radius: 6px; padding: 1px 7px;
    font-size: 11px; font-weight: 700;
    margin-left: 4px; vertical-align: middle;
    border: 1px solid #b3d1f0;
}

/* === Report sections === */
.report-box {
    background: white;
    border: 1px solid #e0ece0;
    border-left: 4px solid #2e7d32;
    border-radius: 8px;
    padding: 16px 20px;
    margin: 10px 0;
}
.report-box-title {
    font-size: 13px; font-weight: 700;
    color: #1b5e20; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 10px;
}

/* === Claim row === */
.claim-row {
    background: #f8fdf8;
    border: 1px solid #e0ece0;
    border-radius: 6px;
    padding: 9px 13px;
    margin: 6px 0;
    font-size: 14px;
    line-height: 1.5;
}

/* === Treatment step === */
.step-row {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin: 8px 0;
}
.step-num {
    background: #2e7d32; color: white;
    border-radius: 50%; width: 24px; height: 24px; min-width: 24px;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; margin-top: 1px;
}

/* === Metric mini-card === */
.mini-card {
    background: white; border: 1px solid #e5ece5;
    border-radius: 8px; padding: 12px 8px;
    text-align: center;
}
.mini-val   { font-size: 20px; font-weight: 700; color: #1b5e20; }
.mini-label { font-size: 11px; color: #888; margin-top: 3px; }

/* === Source card === */
.source-card {
    background: #f9fbf9;
    border: 1px solid #ddeedd;
    border-left: 3px solid #66bb6a;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0;
}

/* === Score bar === */
.bar-wrap { background: #e8f5e9; border-radius: 4px; height: 7px; width: 100%; margin: 4px 0; }
.bar-fill { background: #43a047; border-radius: 4px; height: 7px; }

/* === Empty state === */
.empty-state { text-align: center; padding: 50px 20px; color: #888; }
.empty-icon  { font-size: 64px; margin-bottom: 12px; }

/* === Pipeline table === */
.pipeline-table { border-collapse: collapse; margin: auto; font-size: 13px; }
.pipeline-table td { padding: 6px 16px; }
.pipeline-table .lbl { text-align: right; color: #aaa; }
.pipeline-table .val { color: #333; }
.pipeline-table .novel { color: #2e7d32; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _badge(level: str) -> str:
    cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(level, "badge-low")
    icon = {"High": "●", "Medium": "◐", "Low": "○"}.get(level, "○")
    return f'<span class="{cls}">{icon} {level} Confidence</span>'


def _bar(value: float) -> str:
    pct = min(100, max(0, int(value * 100)))
    return f'<div class="bar-wrap"><div class="bar-fill" style="width:{pct}%;"></div></div>'


def _check_health() -> tuple[bool, dict]:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        data = r.json()
        return r.status_code == 200 and data.get("status") == "healthy", data
    except Exception:
        return False, {}


def _call_analyze_api(
    image_bytes:      bytes,
    filename:         str,
    latitude:         float,
    longitude:        float,
    observation_date: str,
    crop_type:        Optional[str],
) -> dict:
    resp = requests.post(
        f"{API_BASE}/analyze",
        files={"image": (filename, image_bytes, "image/jpeg")},
        data={
            "latitude":         str(latitude),
            "longitude":        str(longitude),
            "observation_date": observation_date,
            "crop_type":        crop_type or "",
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌾 AgriRAG")
    st.caption("Precision Agriculture & Crop Disease Intelligence  \nHybrid RAG · Geo-Temporal Reranking")
    st.divider()

    # Health indicator
    api_healthy, health_data = _check_health()
    if api_healthy:
        device = health_data.get("device", "cpu").upper()
        st.success(f"API Online  |  {device}", icon="✅")
    else:
        st.error("API Offline — start the FastAPI server first.", icon="🔴")
        st.code("uvicorn main:app --port 8000", language="bash")

    st.divider()

    # ── Geo-Temporal inputs (the novel contribution) ──────────────────────────
    st.markdown("### 📍 Geo-Temporal Context")
    st.caption(
        "These parameters activate the system's **novel contribution**: "
        "documents geographically and temporally closer to the observation "
        "receive boosted retrieval scores."
    )

    col_lat, col_lon = st.columns(2)
    with col_lat:
        latitude  = st.number_input("Latitude",  value=28.6139, min_value=-90.0,  max_value=90.0,  format="%.4f", step=0.0001)
    with col_lon:
        longitude = st.number_input("Longitude", value=77.2090, min_value=-180.0, max_value=180.0, format="%.4f", step=0.0001)

    # Quick location presets
    preset = st.selectbox(
        "Location preset",
        ["Custom", "Delhi NCR, India", "Punjab, India", "Tamil Nadu, India",
         "Luzon, Philippines", "Andes, Peru", "Iowa, USA", "Midwest, Kenya"],
        label_visibility="collapsed",
    )
    PRESETS = {
        "Delhi NCR, India":    (28.6139,   77.2090),
        "Punjab, India":       (30.9010,   75.8573),
        "Tamil Nadu, India":   (13.0827,   80.2707),
        "Luzon, Philippines":  (14.1768,  121.2448),
        "Andes, Peru":         (-13.5319, -71.9675),
        "Iowa, USA":           (41.8780,  -93.0977),
        "Midwest, Kenya":      (-1.2921,   36.8219),
    }
    if preset != "Custom" and preset in PRESETS:
        latitude, longitude = PRESETS[preset]

    observation_date = st.date_input("Observation Date", value=date.today())

    st.divider()

    # ── Crop / Analysis settings ──────────────────────────────────────────────
    st.markdown("### 🌱 Analysis Settings")
    crop_type = st.selectbox(
        "Crop Type (optional)",
        ["", "tomato", "wheat", "rice", "potato", "maize", "cotton", "soybean", "chickpea", "other"],
        help="Restricts retrieval to documents for this crop. Leave blank for auto-detection.",
    )

    st.divider()
    st.markdown("### 🛠 Debug Options")
    show_scores   = st.checkbox("Show retrieval scores",    value=True)
    show_raw_json = st.checkbox("Show raw API response",    value=False)
    expand_first  = st.checkbox("Auto-expand first source", value=True)


# ── Main Content ───────────────────────────────────────────────────────────────

st.title("🌾 Crop Disease Intelligence Report")
st.markdown(
    "Upload a crop photograph. The system captions the image with BLIP-2, "
    "retrieves geo-temporally relevant agricultural knowledge via Hybrid RAG, "
    "and synthesises a grounded advisory report with source citations."
)
st.divider()

# ── Image upload ──────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload Crop Image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Photograph of diseased crop leaves, stems, or field — JPG/PNG/WebP.",
)

if uploaded_file is None:
    # ── Empty state ───────────────────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🌿</div>
        <h3 style="color:#2e7d32; margin-bottom:6px;">Upload a crop image to begin</h3>
        <p style="color:#888; margin-bottom:24px;">
            Set your GPS location and date in the sidebar, then upload a crop photograph.
        </p>
        <table class="pipeline-table">
            <tr><td class="lbl">Visual understanding</td><td class="val">BLIP-2 image captioning</td></tr>
            <tr><td class="lbl">Sparse retrieval</td><td class="val">BM25s keyword search</td></tr>
            <tr><td class="lbl">Dense retrieval</td><td class="val">Qdrant + BGE-large embeddings</td></tr>
            <tr><td class="lbl">Score fusion</td><td class="val">Reciprocal Rank Fusion (RRF)</td></tr>
            <tr><td class="lbl">Reranking</td><td class="val">Fine-tuned CrossEncoder (domain-adapted)</td></tr>
            <tr><td class="lbl" style="font-weight:700;">Novel contribution</td><td class="val novel">Geo-Temporal Score Modulation</td></tr>
            <tr><td class="lbl">Synthesis</td><td class="val">Claude Sonnet + grounded citations</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Image + parameter preview ─────────────────────────────────────────────────
col_img, col_params = st.columns([1, 1], gap="large")
with col_img:
    st.image(uploaded_file, caption="Uploaded Crop Image", use_column_width=True)

with col_params:
    st.markdown("#### Analysis Parameters")
    st.write(f"📍 **Location:** `{latitude:.4f}°N, {longitude:.4f}°E`")
    st.write(f"📅 **Date:** `{observation_date.isoformat()}`")
    st.write(f"🌱 **Crop:** `{crop_type or 'Auto-detect from image'}`")
    st.markdown("---")
    st.markdown("**Pipeline:**")
    st.markdown("BLIP-2 → BM25s + Qdrant BGE → RRF → CrossEncoder → **Geo-Temporal** → Gemini 3 flash")
    st.caption(
        "ℹ️ Geo-weight uses Gaussian decay (r=500 km). "
        "Temporal-weight uses exponential decay (τ=730 days)."
    )

st.divider()

# ── Analyze button ────────────────────────────────────────────────────────────
btn_col, _ = st.columns([1, 3])
with btn_col:
    analyze_clicked = st.button(
        "🔬 Analyze Crop Image",
        type="primary",
        disabled=not api_healthy,
        use_container_width=True,
    )

if not analyze_clicked:
    st.stop()

# ── API call ──────────────────────────────────────────────────────────────────
image_bytes = uploaded_file.read()

with st.spinner("Running pipeline… BLIP-2 captioning → Hybrid RAG → Geo-Temporal Reranking → Gemini 3 flash Synthesis"):
    try:
        result = _call_analyze_api(
            image_bytes      = image_bytes,
            filename         = uploaded_file.name,
            latitude         = latitude,
            longitude        = longitude,
            observation_date = observation_date.isoformat(),
            crop_type        = crop_type if crop_type else None,
        )
    except requests.HTTPError as e:
        st.error(f"API Error {e.response.status_code}: {e.response.text[:500]}")
        st.stop()
    except requests.ConnectionError:
        st.error("Cannot reach the FastAPI backend. Ensure it is running on port 8000.")
        st.stop()
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.stop()

st.success("Analysis complete!", icon="✅")
st.divider()

# ── Performance metrics ────────────────────────────────────────────────────────
meta = result.get("processing_metadata", {})
m1, m2, m3, m4, m5 = st.columns(5)
for col, label, key in [
    (m1, "Caption",   "caption_ms"),
    (m2, "Retrieval", "retrieval_ms"),
    (m3, "Synthesis", "synthesis_ms"),
    (m4, "Total",     "total_ms"),
    (m5, "Sources",   None),
]:
    value = f"{meta.get(key, '—')}ms" if key else str(len(result.get("sources", [])))
    col.markdown(
        f'<div class="mini-card"><div class="mini-val">{value}</div>'
        f'<div class="mini-label">{label}</div></div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── BLIP-2 Caption ────────────────────────────────────────────────────────────
st.info(f"**🖼️ Visual Description (BLIP-2):** {result.get('caption', '')}", icon="🤖")

st.divider()

# ── Advisory Report ────────────────────────────────────────────────────────────
st.subheader("📋 Advisory Report")
confidence = result.get("confidence_level", "Low")
st.markdown(_badge(confidence), unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    # Summary
    st.markdown(
        f'<div class="report-box"><div class="report-box-title">📌 Summary</div>'
        f'{result.get("summary", "")}</div>',
        unsafe_allow_html=True
    )

    # Diagnosis
    st.markdown(
        f'<div class="report-box"><div class="report-box-title">🔬 Diagnosis</div>'
        f'{result.get("diagnosis", "")}</div>',
        unsafe_allow_html=True
    )

    # ── Grounded Claims (core demo of citation novelty) ──────────────────────
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown('<div class="report-box-title">🔗 Grounded Claims — Cited Evidence</div>', unsafe_allow_html=True)
    st.caption("Every claim is linked to a retrieved source. Hover a citation index to see the source.")

    cited_claims = result.get("cited_claims", [])
    if cited_claims:
        for claim_obj in cited_claims:
            pills = "".join([
                f'<span class="cite-pill">[{i}]</span>'
                for i in claim_obj.get("source_indices", [])
            ])
            st.markdown(
                f'<div class="claim-row">{claim_obj["claim"]}{pills}</div>',
                unsafe_allow_html=True
            )
    else:
        st.caption("No cited claims returned — try a clearer image or check the knowledge base.")
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    # Treatment Recommendations
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown('<div class="report-box-title">💊 Treatment Plan</div>', unsafe_allow_html=True)
    for i, step in enumerate(result.get("treatment_recommendations", []), 1):
        st.markdown(
            f'<div class="step-row"><div class="step-num">{i}</div><div style="font-size:14px;">{step}</div></div>',
            unsafe_allow_html=True
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # Follow-up Actions
    st.markdown('<div class="report-box">', unsafe_allow_html=True)
    st.markdown('<div class="report-box-title">📅 Follow-up Actions</div>', unsafe_allow_html=True)
    for action in result.get("follow_up_actions", []):
        st.markdown(f"- {action}")
    st.markdown("</div>", unsafe_allow_html=True)


# ── Retrieved Sources ──────────────────────────────────────────────────────────
st.divider()
st.subheader("📚 Retrieved Knowledge Sources")
st.caption(
    "Documents are ranked by: **sigmoid(CE score) × geo-weight × temporal-weight**. "
    "Geo-weight uses Gaussian decay over Haversine distance (r=500 km). "
    "Temporal-weight uses exponential decay (τ=730 days). "
    "This tri-factor score is the system's novel academic contribution."
)

sources = result.get("sources", [])
for src in sources:
    idx       = src["index"]
    is_top    = idx == 0 and expand_first
    score_pct = int(src["final_score"] * 100)

    with st.expander(
        f"[{idx}]  {src['title']}   —   Final score: {src['final_score']:.4f}",
        expanded=is_top,
    ):
        score_cols = st.columns(3)
        score_cols[0].metric("Geo-Weight",      f"{src['geo_weight']:.3f}",      help="1.0 = same location, decays with distance")
        score_cols[1].metric("Temporal-Weight", f"{src['temporal_weight']:.3f}", help="1.0 = today, decays with document age")
        score_cols[2].metric("CE Score (norm)", f"{src['ce_normalized']:.3f}",   help="CrossEncoder relevance, sigmoid normalised")

        if show_scores:
            st.markdown(f"**Geo proximity:** {_bar(src['geo_weight'])}", unsafe_allow_html=True)
            st.markdown(f"**Temporal recency:** {_bar(src['temporal_weight'])}", unsafe_allow_html=True)
            st.markdown(f"**CE relevance:** {_bar(src['ce_normalized'])}", unsafe_allow_html=True)

        meta_cols = st.columns(3)
        meta_cols[0].write(f"**Source:** {src['source']}")
        meta_cols[1].write(f"**Region:** {src.get('region', '—')}")
        meta_cols[2].write(f"**Date:** {src.get('publication_date', '—')}")

        st.markdown("**Content preview:**")
        st.markdown(
            f'<div class="source-card" style="font-size:13px; color:#444;">{src["content_snippet"]}</div>',
            unsafe_allow_html=True
        )


# ── Raw JSON debug ─────────────────────────────────────────────────────────────
if show_raw_json:
    with st.expander("🛠️ Raw API Response (JSON)", expanded=False):
        st.json(result)


# ── Evaluation link ────────────────────────────────────────────────────────────
st.divider()
with st.expander("📊 RAGAS Evaluation Results (last run)", expanded=False):
    results_path = "evaluation_results.json"
    try:
        import json as _json, os
        if os.path.exists(results_path):
            with open(results_path) as f:
                eval_results = _json.load(f)
            e1, e2, e3, e4 = st.columns(4)
            for col, key in zip([e1,e2,e3,e4],
                ["faithfulness","answer_relevancy","context_precision","context_recall"]):
                val = eval_results.get(key, 0.0)
                col.metric(key.replace("_", " ").title(), f"{val:.3f}")
            st.caption(f"Evaluated at: {eval_results.get('evaluated_at','—')} | "
                       f"n={eval_results.get('n_samples','—')}")
            failures = eval_results.get("failures", [])
            if failures:
                st.error(f"CI failures: {', '.join(failures)}")
            else:
                st.success("All CI thresholds passed")
        else:
            st.info("No evaluation results found. Run `python evaluation.py` to generate.")
    except Exception as e:
        st.warning(f"Could not load evaluation results: {e}")
