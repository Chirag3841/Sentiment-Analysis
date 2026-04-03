import re
import numpy as np
import torch
import streamlit as st
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification
 
# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Flipkart Sentiment Analyzer",
    page_icon="🛒",
    layout="centered",
)
 
# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: white;
    }
 
    /* Header */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #aaa;
        font-size: 1rem;
    }
 
    /* Input card */
    .input-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
 
    /* Result card */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin-top: 1.5rem;
        text-align: center;
        animation: fadeIn 0.5s ease;
    }
    .result-positive {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
    }
    .result-neutral {
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: white;
    }
    .result-negative {
        background: linear-gradient(135deg, #eb3349, #f45c43);
        color: white;
    }
    .result-card h2 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .result-card p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.3rem 0;
    }
 
    /* Stars */
    .stars {
        font-size: 2rem;
        letter-spacing: 4px;
        margin: 0.5rem 0;
    }
 
    /* Confidence bar */
    .conf-label {
        font-size: 0.9rem;
        color: #ccc;
        margin-bottom: 0.2rem;
    }
 
    /* History */
    .history-item {
        background: rgba(255,255,255,0.05);
        border-left: 4px solid;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .history-positive { border-color: #38ef7d; }
    .history-neutral  { border-color: #ffd200; }
    .history-negative { border-color: #f45c43; }
 
    /* Button */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #f7971e, #ffd200);
        color: #1a1a2e;
        font-weight: 800;
        font-size: 1.1rem;
        border: none;
        border-radius: 12px;
        padding: 0.8rem;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
    }
 
    /* Text area */
    .stTextArea textarea {
        background: rgba(255,255,255,0.08) !important;
        color: white !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 10px !important;
        font-size: 1rem !important;
    }
 
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to   { opacity: 1; transform: translateY(0); }
    }
 
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)
 
# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 64
MODEL_NAME = "distilbert-base-uncased"
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9!?.,' \s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
@st.cache_resource
def load_model():
    tokenizer     = AutoTokenizer.from_pretrained("distilbert_tokenizer/")
    model         = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3
    )
    model.load_state_dict(torch.load("best_bert_sentiment.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    label_encoder = joblib.load("label_encoder.pkl")
    return tokenizer, model, label_encoder
 
@torch.no_grad()
def predict(text, tokenizer, model, label_encoder):
    text     = clean_text(text)
    encoding = tokenizer(
        text, max_length=MAX_LEN,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids      = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)
    outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
    probs          = torch.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    pred_idx       = int(np.argmax(probs))
    sentiment      = label_encoder.inverse_transform([pred_idx])[0]
    confidence     = float(probs[pred_idx]) * 100
    rating         = 1 if sentiment == "Negative" else 3 if sentiment == "Neutral" else 5
    all_probs      = {
        label_encoder.inverse_transform([i])[0]: float(probs[i]) * 100
        for i in range(len(probs))
    }
    return sentiment, rating, confidence, all_probs
 
# ── Session state for history ─────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
 
# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛒 Flipkart Sentiment Analyzer</h1>
    <p>Powered by DistilBERT · Instantly classify product reviews</p>
</div>
""", unsafe_allow_html=True)
 
# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("⚙️ Loading AI model..."):
    tokenizer, model, label_encoder = load_model()
 
# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="input-card">', unsafe_allow_html=True)
st.markdown("#### ✍️ Enter a Product Review")
review = st.text_area(
    label="",
    placeholder="e.g. The product quality is amazing! Totally worth the price.",
    height=130,
    label_visibility="collapsed"
)
 
col1, col2 = st.columns([3, 1])
with col1:
    analyse_btn = st.button("🔍 Analyse Sentiment", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear History", use_container_width=True)
 
st.markdown('</div>', unsafe_allow_html=True)
 
if clear_btn:
    st.session_state.history = []
    st.rerun()
 
# ── Analysis ──────────────────────────────────────────────────────────────────
if analyse_btn:
    if not review.strip():
        st.warning("⚠️ Please enter a review before clicking Analyse.")
    else:
        with st.spinner("🤖 Analysing your review..."):
            sentiment, rating, confidence, all_probs = predict(
                review, tokenizer, model, label_encoder
            )
 
        card_class = {
            "Positive": "result-positive",
            "Neutral":  "result-neutral",
            "Negative": "result-negative",
        }[sentiment]
 
        emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
        emoji     = emoji_map[sentiment]
        stars     = "⭐" * rating + "☆" * (5 - rating)
 
        # Result card
        st.markdown(f"""
        <div class="result-card {card_class}">
            <h2>{emoji} {sentiment}</h2>
            <div class="stars">{stars}</div>
            <p>Rating: {rating} / 5 &nbsp;|&nbsp; Confidence: {confidence:.1f}%</p>
        </div>
        """, unsafe_allow_html=True)
 
        # Probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Confidence Breakdown")
 
        for label, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
            st.markdown(
                f'<p class="conf-label">{emoji_map[label]} {label}: {prob:.1f}%</p>',
                unsafe_allow_html=True
            )
            st.progress(prob / 100)
 
        # Save to history
        st.session_state.history.insert(0, {
            "review":     review[:80] + ("..." if len(review) > 80 else ""),
            "sentiment":  sentiment,
            "confidence": confidence,
            "rating":     rating,
        })
        st.session_state.history = st.session_state.history[:5]
 
# ── History ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🕘 Recent Analyses")
    for item in st.session_state.history:
        css_class = f"history-{item['sentiment'].lower()}"
        emoji     = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}[item["sentiment"]]
        stars     = "⭐" * item["rating"]
        st.markdown(f"""
        <div class="history-item {css_class}">
            {emoji} <strong>{item['sentiment']}</strong> &nbsp;
            {stars} &nbsp;
            <span style="color:#aaa">{item['confidence']:.1f}% confidence</span><br>
            <span style="color:#ddd; font-size:0.85rem">"{item['review']}"</span>
        </div>
        """, unsafe_allow_html=True)
 
# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#555; font-size:0.8rem;">
    Built with ❤️ using DistilBERT + Streamlit
</div>
""", unsafe_allow_html=True)
