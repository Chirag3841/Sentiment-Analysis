import streamlit as st
import torch
import numpy as np
import re
import os
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Flipkart Sentiment Analyzer", page_icon="🛍️", layout="centered")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.stApp { background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); }
.title-block { text-align: center; padding: 2rem 0 1rem; }
.title-block h1 { font-family: 'Space Mono', monospace; font-size: 2.2rem; font-weight: 700; color: #fff; letter-spacing: -1px; }
.title-block p  { color: #a89fd8; font-size: 1rem; margin-top: 0.5rem; }
.result-card { background: rgba(255,255,255,0.07); border: 1px solid rgba(255,255,255,0.12); border-radius: 16px; padding: 1.8rem 2rem; margin-top: 1.5rem; }
.sentiment-label { font-family: 'Space Mono', monospace; font-size: 2rem; font-weight: 700; margin: 0; }
.positive { color: #4ade80; } .neutral { color: #facc15; } .negative { color: #f87171; }
.meta-row { display: flex; gap: 2rem; margin-top: 1rem; flex-wrap: wrap; }
.meta-item label { font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase; color: #9489c4; }
.meta-item span  { display: block; font-size: 1.2rem; font-weight: 600; color: #e2deff; margin-top: 2px; }
.history-item { background: rgba(255,255,255,0.04); border-left: 3px solid #7c6fe0; border-radius: 8px; padding: 0.8rem 1.1rem; margin-bottom: 0.7rem; font-size: 0.92rem; color: #ccc8f5; }
.badge { display: inline-block; border-radius: 20px; padding: 2px 10px; font-size: 0.78rem; font-weight: 600; margin-left: 8px; }
.badge-pos { background: rgba(74,222,128,0.15); color: #4ade80; }
.badge-neu { background: rgba(250,204,21,0.15);  color: #facc15; }
.badge-neg { background: rgba(248,113,113,0.15); color: #f87171; }
.warn-box  { background: rgba(250,204,21,0.1); border: 1px solid rgba(250,204,21,0.3); border-radius: 10px; padding: 0.9rem 1.2rem; color: #fde68a; font-size: 0.88rem; margin-bottom: 1.2rem; }
.ok-box    { background: rgba(74,222,128,0.1);  border: 1px solid rgba(74,222,128,0.3);  border-radius: 10px; padding: 0.9rem 1.2rem; color: #bbf7d0; font-size: 0.88rem; margin-bottom: 1.2rem; }
.stButton > button { background: linear-gradient(135deg, #7c6fe0, #5b4fcf); color: white; border: none; border-radius: 10px; font-family: 'Space Mono', monospace; font-weight: 700; letter-spacing: 1px; padding: 0.6rem 2rem; width: 100%; }
.stTextArea textarea { background: rgba(255,255,255,0.06) !important; border: 1px solid rgba(255,255,255,0.15) !important; border-radius: 12px !important; color: #f0edff !important; font-family: 'Sora', sans-serif !important; }
div[data-testid="stMarkdownContainer"] p { color: #ccc8f5; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 64
MODEL_NAME = "distilbert-base-uncased"

# Artifact paths — produced by running sentiment_analysis.py once
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
def predict(text, tokenizer, model, label_encoder):
    enc = tokenizer(clean_text(text), max_length=MAX_LEN,
                    padding="max_length", truncation=True, return_tensors="pt")
    out   = model(input_ids=enc["input_ids"].to(DEVICE),
                  attention_mask=enc["attention_mask"].to(DEVICE))
    probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    idx   = int(np.argmax(probs))
    label = label_encoder.inverse_transform([idx])[0]
    conf  = float(probs[idx]) * 100
    rating = 1 if label == "Negative" else 3 if label == "Neutral" else 5
    return label, rating, conf, probs, label_encoder.classes_

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-block"><h1>🛍️ Flipkart Sentiment<br>Analyzer</h1><p>DistilBERT · Positive · Neutral · Negative</p></div>', unsafe_allow_html=True)

with st.spinner("Loading model…"):
    tokenizer, model, label_encoder, trained = load_artifacts()

if not trained:
    st.markdown('<div class="warn-box">⚠️ <b>Trained weights not found.</b><br>Run <code>python sentiment_analysis.py</code> once first, then restart the app.</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="ok-box">✅ Model loaded — ready for instant inference.</div>', unsafe_allow_html=True)

# ── Input ──────────────────────────────────────────────────────────────────────
review = st.text_area("Paste a Flipkart product review",
                      placeholder="e.g. Excellent quality, fast delivery!",
                      height=130)
if st.button("⚡ Analyze Sentiment"):
    if not review.strip():
        st.warning("Please enter a review.")
    else:
        with st.spinner("Analyzing…"):
            sentiment, rating, conf, probs, classes = predict(review, tokenizer, model, label_encoder)

        emoji_map = {"Positive": "😊", "Neutral": "😐", "Negative": "😞"}
        css_map   = {"Positive": "positive", "Neutral": "neutral", "Negative": "negative"}
        star_map  = {1: "★☆☆☆☆", 3: "★★★☆☆", 5: "★★★★★"}
        badge_map = {"Positive": "badge-pos", "Neutral": "badge-neu", "Negative": "badge-neg"}

        st.markdown(f"""
        <div class="result-card">
          <p class="sentiment-label {css_map[sentiment]}">{emoji_map[sentiment]} {sentiment}</p>
          <div class="meta-row">
            <div class="meta-item"><label>Rating</label><span>{star_map[rating]}</span></div>
            <div class="meta-item"><label>Confidence</label><span>{conf:.1f}%</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("#### Probability Breakdown")
        for cls in ["Negative", "Neutral", "Positive"]:
            if cls in classes:
                p = float(probs[list(classes).index(cls)]) * 100
                st.markdown(f"**{cls}**")
                st.progress(p / 100, text=f"{p:.1f}%")

        if "history" not in st.session_state:
            st.session_state.history = []
        st.session_state.history.insert(0, {"text": review[:80] + ("…" if len(review) > 80 else ""), "sentiment": sentiment, "conf": conf})
        st.session_state.history = st.session_state.history[:10]

# ── History ────────────────────────────────────────────────────────────────────
if st.session_state.get("history"):
    st.markdown("---")
    st.markdown("#### 🕘 Recent Analyses")
    badge_map = {"Positive": "badge-pos", "Neutral": "badge-neu", "Negative": "badge-neg"}
    for item in st.session_state.history:
        st.markdown(f'<div class="history-item">"{item["text"]}" <span class="badge {badge_map[item["sentiment"]]}">{item["sentiment"]} · {item["conf"]:.0f}%</span></div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown("<p style='text-align:center;color:#5a5280;font-size:0.8rem;'>DistilBERT fine-tuned on Flipkart reviews · 3-class sentiment</p>", unsafe_allow_html=True)
