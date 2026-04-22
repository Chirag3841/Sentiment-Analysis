import streamlit as st
import torch
import numpy as np
import re
import os
import joblib
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
from collections import Counter

# ── Page config ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ · Flipkart",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500;600&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg:       #09090b;
    --surface:  #111114;
    --border:   #1e1e24;
    --border2:  #2a2a34;
    --text:     #e8e8f0;
    --muted:    #52525e;
    --nav-text: #a0a0b0;
    --accent:   #6ee7b7;
    --pos:      #4ade80;
    --neu:      #fbbf24;
    --neg:      #f87171;
    --blue:     #60a5fa;
    --purple:   #a78bfa;
}

html, body, [class*="css"] {
    font-family: 'Geist Mono', monospace;
    background: var(--bg) !important;
    color: var(--text);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { font-family: 'Geist Mono', monospace !important; }

.sidebar-logo {
    font-size: 1.1rem; font-weight: 600; color: var(--text);
    padding: 1.5rem 0 0.25rem;
    letter-spacing: -0.5px;
}
.sidebar-logo span { color: var(--accent); }
.sidebar-tagline { font-size: 0.65rem; color: var(--muted); margin-bottom: 2rem; letter-spacing: 0.1em; }

/* Nav: all text elements inside radio labels — fully visible grey */
div[data-testid="stRadio"] label,
div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label div,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] p {
    color: #b8b8c8 !important;
    font-size: 0.82rem !important;
    font-family: 'Geist Mono', monospace !important;
}
div[data-testid="stRadio"] label:hover,
div[data-testid="stRadio"] label:hover p,
div[data-testid="stRadio"] label:hover span {
    color: #e8e8f0 !important;
}
/* Selected state — full brightness */
div[data-testid="stRadio"] [aria-checked="true"] ~ div,
div[data-testid="stRadio"] [aria-checked="true"] ~ div p,
div[data-testid="stRadio"] [aria-checked="true"] ~ div span {
    color: #e8e8f0 !important;
}

/* ── Main layout ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1100px !important;
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Page headers ── */
.page-eyebrow {
    font-size: 0.65rem; letter-spacing: 0.15em; color: var(--accent);
    text-transform: uppercase; margin-bottom: 6px;
}
.page-title {
    font-family: 'Instrument Serif', serif;
    font-size: 2.4rem; color: var(--text);
    line-height: 1.15; margin: 0 0 8px;
    letter-spacing: -0.5px;
}
.page-title em { font-style: italic; color: var(--accent); }
.page-desc { font-size: 0.82rem; color: var(--muted); max-width: 520px; line-height: 1.7; margin-bottom: 2.5rem; }

/* ── Stat cards ── */
.stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin-bottom: 2rem; }
.stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 1.1rem 1.25rem;
}
.stat-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 6px; }
.stat-value { font-size: 1.6rem; font-weight: 600; color: var(--text); line-height: 1; margin-bottom: 4px; }
.stat-sub   { font-size: 0.7rem; color: var(--muted); }
.stat-pos   { border-top: 2px solid var(--pos); }
.stat-neu   { border-top: 2px solid var(--neu); }
.stat-neg   { border-top: 2px solid var(--neg); }
.stat-blue  { border-top: 2px solid var(--blue); }

/* ── Cards ── */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 1.4rem 1.5rem;
    margin-bottom: 12px;
}
.card-title { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; color: var(--muted); margin-bottom: 1rem; }

/* ── Input analyze card ── */
.analyze-wrap {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 1.5rem;
    margin-bottom: 1rem;
}

/* ── Result display ── */
.result-hero {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px; padding: 1.75rem 2rem;
    margin-bottom: 12px;
    display: flex; align-items: center; justify-content: space-between;
    gap: 2rem;
}
.rh-label-sm { font-size: 0.65rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.rh-sentiment { font-family: 'Instrument Serif', serif; font-size: 3rem; line-height: 1; margin-bottom: 6px; }
.s-pos { color: var(--pos); } .s-neu { color: var(--neu); } .s-neg { color: var(--neg); }
.rh-conf { font-size: 0.78rem; color: var(--muted); }
.rh-badge {
    width: 64px; height: 64px; border-radius: 12px;
    display: flex; align-items: center; justify-content: center; font-size: 2rem; flex-shrink: 0;
}
.rb-pos { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.2); }
.rb-neu { background: rgba(251,191,36,0.1);  border: 1px solid rgba(251,191,36,0.2); }
.rb-neg { background: rgba(248,113,113,0.1); border: 1px solid rgba(248,113,113,0.2); }

/* ── Bar chart manual ── */
.bar-section { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.25rem 1.5rem; margin-bottom: 12px; }
.bar-row { display: flex; align-items: center; gap: 12px; margin-bottom: 11px; }
.bar-row:last-child { margin-bottom: 0; }
.bar-lbl { font-size: 0.75rem; color: var(--muted); width: 68px; flex-shrink: 0; }
.bar-track { flex: 1; height: 4px; background: var(--border); border-radius: 99px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 99px; }
.bf-pos { background: var(--pos); } .bf-neu { background: var(--neu); } .bf-neg { background: var(--neg); }
.bar-pct { font-size: 0.72rem; color: var(--muted); width: 36px; text-align: right; flex-shrink: 0; }

/* ── History ── */
.hist-row {
    display: flex; align-items: center; gap: 10px;
    padding: 9px 0; border-bottom: 1px solid var(--border);
    font-size: 0.78rem;
}
.hist-row:last-child { border-bottom: none; }
.hist-txt { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--text); }
.hbadge { padding: 2px 9px; border-radius: 99px; font-size: 0.68rem; font-weight: 500; flex-shrink: 0; }
.hb-pos { background: rgba(74,222,128,0.1);  color: var(--pos); }
.hb-neu { background: rgba(251,191,36,0.1);  color: var(--neu); }
.hb-neg { background: rgba(248,113,113,0.1); color: var(--neg); }
.hist-conf { width: 36px; text-align: right; color: var(--muted); font-size: 0.7rem; flex-shrink: 0; }

/* ── Dataset analysis ── */
.review-item {
    background: var(--border);
    border-radius: 8px; padding: 0.9rem 1.1rem; margin-bottom: 8px;
    font-size: 0.78rem; line-height: 1.6; color: var(--text);
    border-left: 3px solid transparent;
}
.ri-pos { border-left-color: var(--pos); }
.ri-neu { border-left-color: var(--neu); }
.ri-neg { border-left-color: var(--neg); }
.ri-meta { font-size: 0.65rem; color: var(--muted); margin-bottom: 5px; }

/* ── About / Info page ── */
.info-section { margin-bottom: 2rem; }
.info-h2 {
    font-family: 'Instrument Serif', serif;
    font-size: 1.4rem; color: var(--text); margin-bottom: 8px;
}
.info-p {
    font-size: 0.82rem; color: var(--nav-text);
    line-height: 1.8; margin-bottom: 8px;
}
.info-list {
    list-style: none; padding: 0; margin: 0 0 10px;
}
.info-list li {
    font-size: 0.8rem; color: var(--nav-text);
    line-height: 1.8; padding-left: 1rem; position: relative;
}
.info-list li::before {
    content: '·'; position: absolute; left: 0; color: var(--accent);
}
.metric-table { width: 100%; border-collapse: collapse; font-size: 0.78rem; }
.metric-table th {
    color: var(--muted); font-weight: 500; text-align: left;
    padding: 8px 12px; border-bottom: 1px solid var(--border);
    font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
}
.metric-table td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
.metric-table tr:last-child td { border-bottom: none; }
.tech-pill {
    display: inline-block; background: var(--border);
    border: 1px solid var(--border2); border-radius: 6px;
    padding: 4px 10px; font-size: 0.72rem; color: var(--nav-text); margin: 3px;
}
.code-block {
    background: var(--border); border: 1px solid var(--border2);
    border-radius: 8px; padding: 1rem 1.25rem;
    font-size: 0.75rem; color: var(--accent); line-height: 1.7; margin: 10px 0;
    font-family: 'Geist Mono', monospace;
    white-space: pre;
}
.highlight-link { color: var(--accent); text-decoration: none; border-bottom: 1px solid rgba(110,231,183,0.3); }

/* ── Streamlit overrides ── */
.stTextArea > div > div > textarea {
    background: var(--bg) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.85rem !important;
    color: var(--text) !important;
    line-height: 1.65 !important;
}
.stTextArea > div > div > textarea::placeholder { color: var(--muted) !important; }
.stTextArea > div > div > textarea:focus {
    border-color: rgba(110,231,183,0.4) !important;
    box-shadow: 0 0 0 3px rgba(110,231,183,0.06) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: var(--muted) !important; font-size: 0.72rem !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}

.stButton > button {
    background: var(--accent) !important;
    color: #09090b !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    width: 100% !important;
    padding: 0.65rem 0 !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* FILE UPLOADER — fix overlapping button text & overall styling */
div[data-testid="stFileUploader"] {
    background: var(--surface) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] {
    padding: 1rem 1.25rem !important;
    display: flex !important;
    align-items: center !important;
    gap: 1rem !important;
    flex-wrap: nowrap !important;
    overflow: visible !important;
}
/* The upload button itself */
div[data-testid="stFileUploaderDropzoneInstructions"] {
    display: flex !important;
    align-items: center !important;
    gap: 1rem !important;
    flex-direction: row !important;
}
div[data-testid="stFileUploader"] button {
    background: var(--border2) !important;
    color: var(--text) !important;
    border: 1px solid #3a3a4a !important;
    border-radius: 6px !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    padding: 6px 14px !important;
    white-space: nowrap !important;
    min-width: 80px !important;
    width: auto !important;
    position: relative !important;
    overflow: hidden !important;
    flex-shrink: 0 !important;
}
/* Hide the ghost/duplicate span that causes the overlap */
div[data-testid="stFileUploader"] button span:not(:first-child) {
    display: none !important;
}
div[data-testid="stFileUploader"] button span {
    position: static !important;
    display: block !important;
    white-space: nowrap !important;
}
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p {
    color: var(--muted) !important;
    font-family: 'Geist Mono', monospace !important;
    font-size: 0.75rem !important;
    margin: 0 !important;
}

.stAlert { background: var(--surface) !important; border: 1px solid var(--border2) !important; border-radius: 8px !important; }
.stAlert * { color: var(--muted) !important; font-family: 'Geist Mono', monospace !important; }

div[data-testid="stMarkdownContainer"] p { color: var(--muted); font-size: 0.82rem; }
.stSelectbox > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Geist Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 128
MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "artifacts/best_bert_sentiment.pt"
ENC_PATH   = "artifacts/label_encoder.pkl"
TOK_PATH   = "artifacts/distilbert_tokenizer"


def clean_text(text):
    if not text: return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9!?.,'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource(show_spinner=False)
def load_artifacts():
    tokenizer = AutoTokenizer.from_pretrained(
        TOK_PATH if os.path.isdir(TOK_PATH) else MODEL_NAME
    )
    if os.path.exists(ENC_PATH):
        label_encoder = joblib.load(ENC_PATH)
        num_classes   = len(label_encoder.classes_)
    else:
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(["Negative", "Neutral", "Positive"])
        num_classes = 3
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_classes)
    trained = os.path.exists(MODEL_PATH)
    if trained:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return tokenizer, model, label_encoder, trained


@torch.no_grad()
def predict_one(text, tokenizer, model, label_encoder):
    enc   = tokenizer(clean_text(text), max_length=MAX_LEN,
                      padding="max_length", truncation=True, return_tensors="pt")
    out   = model(input_ids=enc["input_ids"].to(DEVICE),
                  attention_mask=enc["attention_mask"].to(DEVICE))
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx   = int(np.argmax(probs))
    label = label_encoder.inverse_transform([idx])[0]
    conf  = float(probs[idx]) * 100
    rating = 1 if label == "Negative" else 3 if label == "Neutral" else 5
    return label, rating, conf, probs, label_encoder.classes_


@torch.no_grad()
def predict_batch(texts, tokenizer, model, label_encoder, batch_size=64):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            [clean_text(t) for t in batch],
            max_length=MAX_LEN, padding="max_length",
            truncation=True, return_tensors="pt"
        )
        out   = model(input_ids=enc["input_ids"].to(DEVICE),
                      attention_mask=enc["attention_mask"].to(DEVICE))
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()
        for p in probs:
            idx   = int(np.argmax(p))
            label = label_encoder.inverse_transform([idx])[0]
            conf  = float(p[idx]) * 100
            results.append({
                "sentiment": label,
                "confidence": conf,
                "prob_pos": float(p[list(label_encoder.classes_).index("Positive")]) if "Positive" in label_encoder.classes_ else 0,
                "prob_neu": float(p[list(label_encoder.classes_).index("Neutral")])  if "Neutral"  in label_encoder.classes_ else 0,
                "prob_neg": float(p[list(label_encoder.classes_).index("Negative")]) if "Negative" in label_encoder.classes_ else 0,
            })
    return results


# ── Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">◈ Sentiment<span>IQ</span></div>
    <div class="sidebar-tagline">FLIPKART · DISTILBERT · NLP</div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["⚡  Live Inference", "📂  Dataset Analysis", "ℹ️  About"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.65rem; color:#3a3a4a; line-height:1.8;">
    MODEL<br>distilbert-base-uncased<br><br>
    DATASET<br>~205k Flipkart reviews<br><br>
    CLASSES<br>Positive · Neutral · Negative
    </div>
    """, unsafe_allow_html=True)

with st.spinner(""):
    tokenizer, model, label_encoder, trained = load_artifacts()


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE INFERENCE
# ════════════════════════════════════════════════════════════════════════════════
if "⚡" in page:
    st.markdown('<div class="page-eyebrow">Live Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analyze any<br><em>review instantly</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Paste a Flipkart product review. The fine-tuned DistilBERT model classifies it in milliseconds with a confidence score.</div>', unsafe_allow_html=True)

    if not trained:
        st.warning("⚠ Model weights not found at artifacts/. Run sentiment_analysis.py first.")

    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        review = st.text_area(
            "Review text",
            placeholder="e.g. The phone feels premium but the battery life is terrible after 2 months of use...",
            height=160,
        )
        analyze_btn = st.button("Analyze →")

        if st.session_state.get("history"):
            st.markdown('<br><div class="card"><div class="card-title">Recent</div>', unsafe_allow_html=True)
            bmap = {"Positive": "hb-pos", "Neutral": "hb-neu", "Negative": "hb-neg"}
            rows = "".join([
                f'<div class="hist-row">'
                f'<span class="hist-txt">{h["text"]}</span>'
                f'<span class="hbadge {bmap[h["sentiment"]]}">{h["sentiment"]}</span>'
                f'<span class="hist-conf">{round(h["conf"])}%</span>'
                f'</div>'
                for h in st.session_state.history
            ])
            st.markdown(rows + "</div>", unsafe_allow_html=True)

    with col_out:
        if analyze_btn:
            if not review.strip():
                st.warning("Please enter a review.")
            else:
                with st.spinner("Running inference…"):
                    sentiment, rating, conf, probs, classes = predict_one(
                        review, tokenizer, model, label_encoder
                    )

                c   = sentiment.lower()[:3]
                emo = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}[sentiment]
                stars = "★" * rating + "☆" * (5 - rating)

                def gp(cls):
                    return round(float(probs[list(classes).index(cls)]) * 100) if cls in classes else 0

                pos_p, neu_p, neg_p = gp("Positive"), gp("Neutral"), gp("Negative")

                st.markdown(f"""
                <div class="result-hero">
                  <div>
                    <div class="rh-label-sm">Detected sentiment</div>
                    <div class="rh-sentiment s-{c}">{sentiment}</div>
                    <div class="rh-conf">{conf:.1f}% confidence &nbsp;·&nbsp; {stars}</div>
                  </div>
                  <div class="rh-badge rb-{c}">{emo}</div>
                </div>

                <div class="bar-section">
                  <div class="card-title">Probability breakdown</div>
                  <div class="bar-row">
                    <span class="bar-lbl">Positive</span>
                    <div class="bar-track"><div class="bar-fill bf-pos" style="width:{pos_p}%"></div></div>
                    <span class="bar-pct">{pos_p}%</span>
                  </div>
                  <div class="bar-row">
                    <span class="bar-lbl">Neutral</span>
                    <div class="bar-track"><div class="bar-fill bf-neu" style="width:{neu_p}%"></div></div>
                    <span class="bar-pct">{neu_p}%</span>
                  </div>
                  <div class="bar-row">
                    <span class="bar-lbl">Negative</span>
                    <div class="bar-track"><div class="bar-fill bf-neg" style="width:{neg_p}%"></div></div>
                    <span class="bar-pct">{neg_p}%</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                if "history" not in st.session_state:
                    st.session_state.history = []
                st.session_state.history.insert(0, {
                    "text": review[:65] + ("…" if len(review) > 65 else ""),
                    "sentiment": sentiment, "conf": conf,
                })
                st.session_state.history = st.session_state.history[:6]
        else:
            st.markdown("""
            <div style="height:280px; display:flex; flex-direction:column; align-items:center; justify-content:center; gap:10px; opacity:0.3;">
              <div style="font-size:2.5rem;">◈</div>
              <div style="font-size:0.75rem; letter-spacing:0.15em; color:#52525e;">AWAITING INPUT</div>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
elif "📂" in page:
    st.markdown('<div class="page-eyebrow">Dataset Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Upload CSV,<br><em>get full analysis</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Upload any CSV with a text column. The model will classify every row and produce charts, metrics, and sample results — no code required.</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded, encoding="latin1")
            st.markdown(f'<div style="font-size:0.72rem;color:#52525e;margin-bottom:1rem;">Loaded {len(df_raw):,} rows · {len(df_raw.columns)} columns</div>', unsafe_allow_html=True)

            text_col_options = [c for c in df_raw.columns if df_raw[c].dtype == object]
            if not text_col_options:
                st.error("No text columns found in CSV.")
                st.stop()

            col_a, col_b = st.columns([1, 2])
            with col_a:
                text_col = st.selectbox("Text column", text_col_options)
            with col_b:
                sample_n = st.selectbox("Rows to analyze", [100, 500, 1000, 5000, len(df_raw)], index=1)

            run_btn = st.button(f"Analyze {sample_n} rows →")

            if run_btn:
                df_sub  = df_raw[text_col].dropna().astype(str).head(sample_n).tolist()
                progress = st.progress(0, text="Classifying…")
                results  = []
                bs = 64
                for i in range(0, len(df_sub), bs):
                    batch_res = predict_batch(df_sub[i:i+bs], tokenizer, model, label_encoder)
                    results.extend(batch_res)
                    progress.progress(min((i + bs) / len(df_sub), 1.0), text=f"Batch {i//bs+1}/{(len(df_sub)-1)//bs+1}")
                progress.empty()

                df_res = pd.DataFrame(results)
                df_res["text"] = df_sub[:len(df_res)]

                counts = df_res["sentiment"].value_counts()
                total  = len(df_res)
                pos_n  = counts.get("Positive", 0)
                neu_n  = counts.get("Neutral", 0)
                neg_n  = counts.get("Negative", 0)

                # ── Stat row
                st.markdown(f"""
                <div class="stat-grid">
                  <div class="stat-card stat-blue">
                    <div class="stat-label">Total analyzed</div>
                    <div class="stat-value">{total:,}</div>
                    <div class="stat-sub">reviews</div>
                  </div>
                  <div class="stat-card stat-pos">
                    <div class="stat-label">Positive</div>
                    <div class="stat-value">{pos_n:,}</div>
                    <div class="stat-sub">{pos_n/total*100:.1f}% of total</div>
                  </div>
                  <div class="stat-card stat-neu">
                    <div class="stat-label">Neutral</div>
                    <div class="stat-value">{neu_n:,}</div>
                    <div class="stat-sub">{neu_n/total*100:.1f}% of total</div>
                  </div>
                  <div class="stat-card stat-neg">
                    <div class="stat-label">Negative</div>
                    <div class="stat-value">{neg_n:,}</div>
                    <div class="stat-sub">{neg_n/total*100:.1f}% of total</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Pie chart only (histogram removed)
                fig_pie = go.Figure(go.Pie(
                    labels=["Positive", "Neutral", "Negative"],
                    values=[pos_n, neu_n, neg_n],
                    hole=0.6,
                    marker_colors=["#4ade80", "#fbbf24", "#f87171"],
                    textinfo="percent",
                    textfont_size=12,
                    textfont_color="#09090b",
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Geist Mono, monospace", color="#52525e", size=11),
                    legend=dict(orientation="h", y=-0.1, font_color="#a0a0b0"),
                    margin=dict(t=10, b=10, l=10, r=10),
                    showlegend=True,
                    annotations=[dict(
                        text=f"<b>{total}</b>", x=0.5, y=0.5,
                        font_size=18, showarrow=False, font_color="#e8e8f0"
                    )]
                )
                st.markdown('<div class="card"><div class="card-title">Sentiment distribution</div>', unsafe_allow_html=True)
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar": False})
                st.markdown("</div>", unsafe_allow_html=True)

                # ── Sample reviews
                st.markdown('<div class="card"><div class="card-title">Sample results</div>', unsafe_allow_html=True)
                tab_pos, tab_neu, tab_neg = st.tabs(["Positive", "Neutral", "Negative"])
                for tab, sent, css in [(tab_pos,"Positive","pos"),(tab_neu,"Neutral","neu"),(tab_neg,"Negative","neg")]:
                    with tab:
                        sub = df_res[df_res["sentiment"]==sent].head(8)
                        if sub.empty:
                            st.markdown(f'<div style="color:#52525e;font-size:0.78rem;padding:1rem;">No {sent.lower()} reviews found.</div>', unsafe_allow_html=True)
                        for _, row in sub.iterrows():
                            st.markdown(f"""
                            <div class="review-item ri-{css}">
                              <div class="ri-meta">{sent} · {row['confidence']:.1f}% confidence</div>
                              {row['text'][:180]}{'…' if len(row['text'])>180 else ''}
                            </div>
                            """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                # ── Download
                st.download_button(
                    "Download results as CSV",
                    df_res[["text","sentiment","confidence"]].to_csv(index=False),
                    file_name="sentiment_results.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.markdown("""
        <div style="background:#111114; border:1px dashed #2a2a34; border-radius:12px; padding:3rem; text-align:center; margin-top:1rem;">
          <div style="font-size:2rem; margin-bottom:12px; opacity:0.4;">📂</div>
          <div style="font-size:0.8rem; color:#52525e; line-height:1.8;">
            Upload any CSV with a review/text column.<br>
            Works with Flipkart, Amazon, or custom datasets.<br><br>
            <span style="font-size:0.7rem;">Expected column: Summary, Review, text, or any text column</span>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ════════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="page-eyebrow">About this project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Flipkart Review<br><em>Sentiment Analyzer</em></div>', unsafe_allow_html=True)

    col_info, col_metrics = st.columns([1.3, 0.7], gap="large")

    with col_info:
        # FIX #1: Replaced raw HTML div blocks with proper st.markdown text + structured HTML
        st.markdown('<div class="info-section"><div class="info-h2">Overview</div></div>', unsafe_allow_html=True)
        st.markdown(
            "This project implements a sentiment analysis system using a fine-tuned DistilBERT model "
            "trained on more than 205,000 Flipkart product reviews. The model classifies reviews into "
            "three categories: **Positive**, **Neutral**, and **Negative**."
        )
        st.markdown(
            "The goal is a practical NLP application that understands the context of customer reviews "
            "and predicts sentiment accurately using transfer learning with a transformer-based "
            "architecture — achieving strong performance while keeping inference efficient."
        )

        st.markdown('<div class="info-section"><div class="info-h2">Key Features</div></div>', unsafe_allow_html=True)
        st.markdown("""
- Fine-tuned DistilBERT for multi-class sentiment classification
- Handles class imbalance using random oversampling
- Mixed precision (fp16) training for speed
- Real-time predictions with confidence scores
- Star rating output derived from sentiment
- Batch CSV analysis with charts and export
        """)

        st.markdown('<div class="info-section"><div class="info-h2">Tech Stack</div></div>', unsafe_allow_html=True)
        st.markdown("""
<div>
  <span class="tech-pill">DistilBERT</span>
  <span class="tech-pill">HuggingFace Transformers</span>
  <span class="tech-pill">PyTorch</span>
  <span class="tech-pill">Streamlit</span>
  <span class="tech-pill">Plotly</span>
  <span class="tech-pill">scikit-learn</span>
  <span class="tech-pill">imbalanced-learn</span>
</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="info-section"><div class="info-h2">Usage</div></div>', unsafe_allow_html=True)
        st.markdown("""
<div class="code-block">analyzer = SentimentAnalyzer(model, tokenizer, label_encoder)
sentiment, rating, confidence = analyzer.predict_sentiment("Great product!")
# Returns: ('Positive', 5, 94.2)</div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="info-section"><div class="info-h2">Running the Project</div></div>', unsafe_allow_html=True)
        st.markdown("""
<div class="code-block">pip install -r requirements.txt
python sentiment_analysis.py   # train once, saves artifacts/
streamlit run app.py           # launch the app</div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="info-section"><div class="info-h2">Project Structure</div></div>', unsafe_allow_html=True)
        st.markdown("""
<div class="code-block">sentiment/
├── app.py
├── sentiment_analysis.py
├── requirements.txt
├── flipkart_product_copy.csv
└── artifacts/
    ├── best_bert_sentiment.pt
    ├── label_encoder.pkl
    └── distilbert_tokenizer/</div>
        """, unsafe_allow_html=True)

    with col_metrics:
        st.markdown("""
        <div class="card" style="margin-bottom:12px;">
          <div class="card-title">Model performance</div>
          <table class="metric-table">
            <tr><th>Metric</th><th>Score</th></tr>
            <tr><td>Train Accuracy</td><td style="color:#4ade80;">94.72%</td></tr>
            <tr><td>Val Accuracy</td><td style="color:#4ade80;">96.27%</td></tr>
            <tr><td>Epochs</td><td>2</td></tr>
            <tr><td>Batch size</td><td>64</td></tr>
            <tr><td>Max length</td><td>128 tokens</td></tr>
            <tr><td>Learning rate</td><td>2e-5</td></tr>
          </table>
        </div>

        <div class="card">
          <div class="card-title">Dataset info</div>
          <table class="metric-table">
            <tr><th>Split</th><th>Size</th></tr>
            <tr><td>Total reviews</td><td>~205,000</td></tr>
            <tr><td>Train</td><td>~148k</td></tr>
            <tr><td>Validation</td><td>~20k</td></tr>
            <tr><td>Test</td><td>~37k</td></tr>
            <tr><td>Classes</td><td>3</td></tr>
            <tr><td>Source</td><td>Flipkart</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
