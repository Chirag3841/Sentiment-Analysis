import re
import os
import joblib
import numpy as np
import pandas as pd
import torch
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ · Flipkart",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR LOCK
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<script>
(function keepSidebarOpen() {
    var timer = setInterval(function() {
        var sels = [
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapsedControl"]',
            'button[data-testid="baseButton-headerNoPadding"]'
        ];
        sels.forEach(function(sel) {
            window.parent.document.querySelectorAll(sel).forEach(function(b) {
                b.style.display = 'none';
            });
        });
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.transform  = 'translateX(0)';
            sidebar.style.minWidth   = '280px';
            sidebar.style.width      = '280px';
            sidebar.style.marginLeft = '0';
            if (sidebar.getAttribute('aria-expanded') === 'false') {
                sidebar.setAttribute('aria-expanded', 'true');
            }
        }
    }, 400);
    setTimeout(function() { clearInterval(timer); }, 15000);
})();
</script>
""", unsafe_allow_html=True)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Geist+Mono:wght@300;400;500;600&family=Instrument+Serif:ital@0;1&display=swap');
:root {
    --bg:#09090b; --surface:#111114; --border:#1e1e24; --border2:#2a2a34;
    --text:#e8e8f0; --muted:#8892a4; --nav-text:#a0a0b0; --accent:#6ee7b7;
    --pos:#4ade80; --neu:#fbbf24; --neg:#f87171; --blue:#60a5fa; --purple:#a78bfa;
    --sidebar-w: 280px;
}
html, body, [class*="css"] {
    font-family: 'Geist Mono', monospace;
    background: var(--bg) !important;
    color: var(--text);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ══════════════════════════════════════════════════════════
   FIX: Hide ALL sidebar collapse controls +
   prevent "keyboard_double_arrow_left" text from leaking
   ══════════════════════════════════════════════════════════ */
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"],
button[data-testid="baseButton-headerNoPadding"],
.st-emotion-cache-1dp5vir,
.st-emotion-cache-czk5ss,
[class*="collapsedControl"],
button[data-testid="baseButton-headerNoPadding"] span,
button[data-testid="baseButton-headerNoPadding"] svg {
    display:        none      !important;
    visibility:     hidden    !important;
    opacity:        0         !important;
    pointer-events: none      !important;
    width:          0         !important;
    height:         0         !important;
    overflow:       hidden    !important;
    position:       absolute  !important;
    top:            -9999px   !important;
    left:           -9999px   !important;
}

/* ══ SIDEBAR — hard-pinned open ══ */
section[data-testid="stSidebar"] {
    background:   var(--surface)   !important;
    border-right: 1px solid var(--border) !important;
    min-width:    var(--sidebar-w) !important;
    max-width:    var(--sidebar-w) !important;
    width:        var(--sidebar-w) !important;
    transform:    translateX(0)    !important;
    margin-left:  0                !important;
    display:      flex             !important;
    overflow:     hidden           !important;
}
section[data-testid="stSidebar"] > div:first-child {
    min-width:   var(--sidebar-w) !important;
    padding-top: 0                !important;
    overflow:    hidden           !important;
}
section[data-testid="stSidebar"] section {
    min-width: var(--sidebar-w) !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    margin-left: 0               !important;
    transform:   translateX(0)   !important;
    display:     flex            !important;
    min-width:   var(--sidebar-w) !important;
}
section[data-testid="stSidebar"] > div:first-child > div:first-child > div:first-child > button,
section[data-testid="stSidebar"] > div > div > div > button:first-child {
    display:    none !important;
    visibility: hidden !important;
}
section[data-testid="stSidebar"] * { font-family: 'Geist Mono', monospace !important; }

/* ── Sidebar brand strip ── */
.brand-strip {
    background: var(--accent);
    padding: 20px 22px 16px 22px;
    width: 100%;
    box-sizing: border-box;
    margin-bottom: 0;
}
.brand-strip .company {
    font-family: 'Geist Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #09090b;
    letter-spacing: -0.5px;
    margin: 0;
    line-height: 1.1;
}
.brand-strip .company span { opacity: 0.65; }
.brand-strip .tagline {
    font-size: 0.6rem;
    color: rgba(9,9,11,0.6);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 5px;
}

/* ── Sidebar nav label ── */
.nav-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    padding: 18px 18px 8px 18px;
    display: block;
}

div.nav-wrap       { margin: 2px 10px; border-radius: 7px; overflow: hidden; }
div.nav-active-wrap{ margin: 2px 10px; border-radius: 7px; overflow: hidden;
                     background: rgba(110,231,183,0.12);
                     border-left: 3px solid var(--accent); }

section[data-testid="stSidebar"] .stButton > button {
    background:    transparent  !important;
    color:         #b8c4d4      !important;
    border:        none         !important;
    border-radius: 6px          !important;
    font-family:   'Geist Mono', monospace !important;
    font-size:     0.82rem      !important;
    font-weight:   400          !important;
    width:         100%         !important;
    text-align:    left         !important;
    padding:       9px 12px     !important;
    letter-spacing: 0.01em      !important;
    transition:    background 0.15s, color 0.15s !important;
    box-shadow:    none         !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(110,231,183,0.08) !important;
    color:      var(--accent)          !important;
}
section[data-testid="stSidebar"] div.nav-active-wrap .stButton > button {
    color:       var(--accent) !important;
    font-weight: 600           !important;
}

.status-block {
    margin: 0 10px;
    padding: 9px 13px;
    border-radius: 6px;
    font-size: 0.78rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.status-ok   { background: rgba(74,222,128,0.07);  border: 1px solid rgba(74,222,128,0.25); color: var(--pos); }
.status-warn { background: rgba(251,191,36,0.07);  border: 1px solid rgba(251,191,36,0.25); color: var(--neu); }

/* ── Main content ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1100px !important;
}

/* ── Typography ── */
.page-eyebrow  { font-size:0.65rem; letter-spacing:0.15em; color:var(--accent); text-transform:uppercase; margin-bottom:6px; }
.page-title    { font-family:'Instrument Serif',serif; font-size:2.4rem; color:var(--text); line-height:1.15; margin:0 0 8px; letter-spacing:-0.5px; }
.page-title em { font-style:italic; color:var(--accent); }
.page-desc     { font-size:0.82rem; color:var(--muted); max-width:520px; line-height:1.7; margin-bottom:2.5rem; }

/* ── Stats ── */
.stat-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:2rem; }
.stat-card { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.1rem 1.25rem; }
.stat-label { font-size:0.65rem; color:var(--muted); text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
.stat-value { font-size:1.6rem; font-weight:600; color:var(--text); line-height:1; margin-bottom:4px; }
.stat-sub   { font-size:0.7rem; color:var(--muted); }
.stat-pos { border-top:2px solid var(--pos); }
.stat-neu { border-top:2px solid var(--neu); }
.stat-neg { border-top:2px solid var(--neg); }
.stat-blue{ border-top:2px solid var(--blue); }

/* ── Cards ── */
.card        { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.4rem 1.5rem; margin-bottom:12px; }
.card-title  { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--muted); margin-bottom:1rem; }

/* ── Result hero ── */
.result-hero { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:1.75rem 2rem; margin-bottom:12px; display:flex; align-items:center; justify-content:space-between; gap:2rem; }
.rh-label-sm { font-size:0.65rem; letter-spacing:0.12em; text-transform:uppercase; color:var(--muted); margin-bottom:8px; }
.rh-sentiment{ font-family:'Instrument Serif',serif; font-size:3rem; line-height:1; margin-bottom:6px; }
.s-pos { color:var(--pos); } .s-neu { color:var(--neu); } .s-neg { color:var(--neg); }
.rh-conf { font-size:0.78rem; color:var(--muted); display:flex; align-items:center; gap:8px; flex-wrap:wrap; margin-top:6px; }
.rh-badge { width:64px; height:64px; border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:2rem; flex-shrink:0; }
.rb-pos { background:rgba(74,222,128,0.1); border:1px solid rgba(74,222,128,0.2); }
.rb-neu { background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.2); }
.rb-neg { background:rgba(248,113,113,0.1); border:1px solid rgba(248,113,113,0.2); }

/* ── Star rating — CSS dot-based, immune to font rendering bugs ── */
.star-row { display:inline-flex; align-items:center; gap:4px; vertical-align:middle; }
.star-dot {
    width: 9px; height: 9px; border-radius: 50%;
    display: inline-block; flex-shrink: 0;
}
.star-dot.on  { background: #fbbf24; box-shadow: 0 0 4px rgba(251,191,36,0.5); }
.star-dot.off { background: transparent; border: 1px solid #3a3a44; }

/* ── Probability bars ── */
.bar-section { background:var(--surface); border:1px solid var(--border); border-radius:10px; padding:1.25rem 1.5rem; margin-bottom:12px; }
.bar-row     { display:flex; align-items:center; gap:12px; margin-bottom:11px; }
.bar-row:last-child { margin-bottom:0; }
.bar-lbl  { font-size:0.75rem; color:var(--muted); width:68px; flex-shrink:0; }
.bar-track{ flex:1; height:4px; background:var(--border); border-radius:99px; overflow:hidden; }
.bar-fill { height:100%; border-radius:99px; }
.bf-pos { background:var(--pos); } .bf-neu { background:var(--neu); } .bf-neg { background:var(--neg); }
.bar-pct  { font-size:0.72rem; color:var(--muted); width:36px; text-align:right; flex-shrink:0; }

/* ── History rows ── */
.hist-row { display:flex; align-items:center; gap:10px; padding:9px 0; border-bottom:1px solid var(--border); font-size:0.78rem; }
.hist-row:last-child { border-bottom:none; }
.hist-txt  { flex:1; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; color:var(--text); }
.hbadge    { padding:2px 9px; border-radius:99px; font-size:0.68rem; font-weight:500; flex-shrink:0; }
.hb-pos { background:rgba(74,222,128,0.1); color:var(--pos); }
.hb-neu { background:rgba(251,191,36,0.1); color:var(--neu); }
.hb-neg { background:rgba(248,113,113,0.1); color:var(--neg); }
.hist-conf { width:36px; text-align:right; color:var(--muted); font-size:0.7rem; flex-shrink:0; }

/* ── Review items ── */
.review-item { background:var(--border); border-radius:8px; padding:0.9rem 1.1rem; margin-bottom:8px; font-size:0.78rem; line-height:1.6; color:var(--text); border-left:3px solid transparent; }
.ri-pos { border-left-color:var(--pos); } .ri-neu { border-left-color:var(--neu); } .ri-neg { border-left-color:var(--neg); }
.ri-meta { font-size:0.65rem; color:var(--muted); margin-bottom:5px; }

/* ── About / info ── */
.info-section { margin-bottom:2rem; }
.info-h2 { font-family:'Instrument Serif',serif; font-size:1.4rem; color:var(--text); margin-bottom:8px; }
.metric-table { width:100%; border-collapse:collapse; font-size:0.78rem; }
.metric-table th { color:var(--muted); font-weight:500; text-align:left; padding:8px 12px; border-bottom:1px solid var(--border); font-size:0.68rem; letter-spacing:0.08em; text-transform:uppercase; }
.metric-table td { padding:10px 12px; border-bottom:1px solid var(--border); color:var(--text); }
.metric-table tr:last-child td { border-bottom:none; }
.tech-pill { display:inline-block; background:var(--border); border:1px solid var(--border2); border-radius:6px; padding:4px 10px; font-size:0.72rem; color:var(--nav-text); margin:3px; }
.code-block { background:var(--border); border:1px solid var(--border2); border-radius:8px; padding:1rem 1.25rem; font-size:0.75rem; color:var(--accent); line-height:1.7; margin:10px 0; font-family:'Geist Mono',monospace; white-space:pre; }

/* ── Form controls ── */
.stTextArea > div > div > textarea { background:var(--bg) !important; border:1px solid var(--border2) !important; border-radius:8px !important; font-family:'Geist Mono',monospace !important; font-size:0.85rem !important; color:var(--text) !important; line-height:1.65 !important; }
.stTextArea > div > div > textarea::placeholder { color:var(--muted) !important; }
.stTextArea > div > div > textarea:focus { border-color:rgba(110,231,183,0.4) !important; box-shadow:0 0 0 3px rgba(110,231,183,0.06) !important; }
label[data-testid="stWidgetLabel"] p { color:var(--muted) !important; font-size:0.72rem !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }
.stButton > button { background:var(--accent) !important; color:#09090b !important; border:none !important; border-radius:8px !important; font-family:'Geist Mono',monospace !important; font-size:0.82rem !important; font-weight:600 !important; width:100% !important; padding:0.65rem 0 !important; letter-spacing:0.02em !important; }
.stButton > button:hover { opacity:0.88 !important; }
div[data-testid="stFileUploader"] { background:var(--surface) !important; border:1px dashed var(--border2) !important; border-radius:10px !important; }
div[data-testid="stFileUploader"] button { background:var(--border2) !important; color:var(--text) !important; border:1px solid #3a3a4a !important; border-radius:6px !important; font-family:'Geist Mono',monospace !important; font-size:0.75rem !important; padding:6px 14px !important; white-space:nowrap !important; min-width:80px !important; width:auto !important; flex-shrink:0 !important; }
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p { color:var(--muted) !important; font-family:'Geist Mono',monospace !important; font-size:0.75rem !important; margin:0 !important; }
.stAlert { background:var(--surface) !important; border:1px solid var(--border2) !important; border-radius:8px !important; }
.stAlert * { color:var(--muted) !important; font-family:'Geist Mono',monospace !important; }
div[data-testid="stMarkdownContainer"] p { color:var(--muted); font-size:0.82rem; }
.stSelectbox > div > div { background:var(--surface) !important; border:1px solid var(--border2) !important; border-radius:8px !important; color:var(--text) !important; font-family:'Geist Mono',monospace !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 128
MODEL_NAME = "distilbert-base-uncased"
MODEL_PATH = "artifacts/best_bert_sentiment.pt"
ENC_PATH   = "artifacts/label_encoder.pkl"
TOK_PATH   = "artifacts/distilbert_tokenizer"

VALID_SHORT_WORDS = {
    'a','an','the','is','it','in','on','at','to','do','be','by','my','or','as',
    'so','up','of','if','no','ok','yes','not','but','and','for','are','was','has',
    'had','its','too','all','can','did','got','let','put','get','use','bad','lot',
    'way','now','how','who','why','one','two','out','off','old','new','top','bit',
    'low','big','try','buy','fit','fix','cut','run','per','yet','far','set','own',
    'due','any','may','few','add','act','age','ago','app','pro','con','fee','tip',
    'wow','see','say','she','him','her','his','our','war','win','hot','job','key',
}
CONCESSIVE = {
    "but","however","though","although","even though","yet","nevertheless",
    "nonetheless","despite","whereas","while","still","that said","having said that",
}
ADDITIVE = {
    "and","also","moreover","furthermore","in addition","besides","plus","as well",
}
CONCESSIVE_WEIGHTS = {"first": 0.35, "last": 0.65}
ADDITIVE_WEIGHT    = 1.0
MIN_CONFIDENCE     = 0.35

# ── Rating map: always use this, never derive from index arithmetic ────────────
RATING_MAP = {"Negative": 1, "Neutral": 3, "Positive": 5}


def make_star_dots(rating: int, total: int = 5) -> str:
    """
    Render rating as CSS dot-circles — completely immune to font/glyph rendering
    bugs that can make ☆ look filled or miscounted.
    rating is 1, 3, or 5.
    """
    dots = "".join(
        f'<span class="star-dot {"on" if i <= rating else "off"}"></span>'
        for i in range(1, total + 1)
    )
    return f'<span class="star-row">{dots}</span>'


@dataclass
class ClauseResult:
    text:        str
    sentiment:   str
    probs:       np.ndarray
    weight:      float
    conjunction: str


# ══════════════════════════════════════════════════════════════════════════════
# TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9!?.,'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    while tokens:
        last = tokens[-1].rstrip(".,!?'")
        if last.isalpha() and len(last) <= 3 and last not in VALID_SHORT_WORDS:
            tokens.pop()
        else:
            break
    return " ".join(tokens)


def validate_input(raw: str) -> Tuple[bool, str]:
    stripped = raw.strip()
    if not stripped:
        return False, "empty"
    cleaned     = clean_text(stripped)
    alpha_words = [w for w in cleaned.split()
                   if re.fullmatch(r'[a-z]{2,}', w.rstrip(".,!?'"))]
    if len(alpha_words) == 0:
        return False, "too_few_words"
    COMMON_WORDS = {
        "good","bad","nice","great","excellent","poor",
        "worst","amazing","average","ok","fine","awesome","product",
        "value","quality","item"
    }
    words = cleaned.split()
    if len(words) == 1 and words[0] not in COMMON_WORDS:
        return False, "too_few_words"
    non_space = re.sub(r'\s', '', stripped)
    if non_space and sum(c.isalpha() for c in non_space) / len(non_space) < 0.40:
        return False, "low_alpha_ratio"
    if re.fullmatch(r'(.)\1{4,}', stripped.replace(" ", "")):
        return False, "repeated_chars"
    return True, ""


# ══════════════════════════════════════════════════════════════════════════════
# CLAUSE SPLITTING
# ══════════════════════════════════════════════════════════════════════════════
def split_clauses(text: str) -> List[Tuple[str, str]]:
    all_conj = sorted(CONCESSIVE | ADDITIVE, key=len, reverse=True)
    pattern  = (r'(?<![a-z])(' + '|'.join(re.escape(c) for c in all_conj) + r')(?![a-z])')
    parts    = re.split(pattern, text, flags=re.IGNORECASE)
    clauses: List[Tuple[str, str]] = []
    current_conj = ""; buffer = ""
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if not chunk: i += 1; continue
        if chunk.lower() in {c.lower() for c in all_conj}:
            current_conj = chunk.lower(); i += 1; continue
        combined   = (buffer + " " + chunk).strip() if buffer else chunk
        word_count = len(combined.split())
        if word_count < 4 and i + 2 < len(parts):
            buffer = combined
        else:
            clauses.append((combined, current_conj))
            buffer = ""; current_conj = ""
        i += 1
    if buffer:
        if clauses: clauses[-1] = (clauses[-1][0] + " " + buffer, clauses[-1][1])
        else:       clauses.append((buffer, ""))
    return clauses if clauses else [(text, "")]


def assign_weights(clauses: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    if len(clauses) == 1:
        return [(clauses[0][0], clauses[0][1], 1.0)]
    has_conc = any(c.lower() in CONCESSIVE for _, c in clauses[1:])
    rw = []
    for idx, (_, conj) in enumerate(clauses):
        if idx == 0:
            rw.append(CONCESSIVE_WEIGHTS["first"] if has_conc else ADDITIVE_WEIGHT)
        else:
            rw.append(CONCESSIVE_WEIGHTS["last"] if conj.lower() in CONCESSIVE else ADDITIVE_WEIGHT)
    total = sum(rw)
    return [(clauses[i][0], clauses[i][1], rw[i] / total) for i in range(len(clauses))]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ── Strategy: Local files first, then Hugging Face Hub (Streamlit Cloud)
#    Set HF_REPO in Streamlit secrets: HF_REPO = "Chirag238/sentimentiq"
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_artifacts():
    from huggingface_hub import hf_hub_download
    from sklearn.preprocessing import LabelEncoder

    hf_repo = os.environ.get("HF_REPO", "")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if os.path.isdir(TOK_PATH):
        tokenizer = AutoTokenizer.from_pretrained(TOK_PATH)
    elif hf_repo:
        try:
            tok_file = hf_hub_download(repo_id=hf_repo, filename="tokenizer_config.json")
            tok_dir  = os.path.dirname(tok_file)
            tokenizer = AutoTokenizer.from_pretrained(tok_dir)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    else:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ── Label Encoder ─────────────────────────────────────────────────────────
    if os.path.exists(ENC_PATH):
        label_encoder = joblib.load(ENC_PATH)
        num_classes   = len(label_encoder.classes_)
    elif hf_repo:
        try:
            enc_path      = hf_hub_download(repo_id=hf_repo, filename="label_encoder.pkl")
            label_encoder = joblib.load(enc_path)
            num_classes   = len(label_encoder.classes_)
        except Exception:
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.array(["Negative", "Neutral", "Positive"])
            num_classes = 3
    else:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.array(["Negative", "Neutral", "Positive"])
        num_classes = 3

    # ── Model architecture ────────────────────────────────────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_classes
    )

    # ── Model weights (.pt) ───────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        pt_path = MODEL_PATH
    elif hf_repo:
        try:
            pt_path = hf_hub_download(
                repo_id=hf_repo,
                filename="best_bert_sentiment.pt",
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not download model weights from HF Hub "
                f"(repo='{hf_repo}'): {e}"
            )
    else:
        raise FileNotFoundError(
            "Model weights not found. Set HF_REPO in Streamlit secrets."
        )

    # ── Load weights — weights_only=False for PyTorch >= 2.6 compatibility ───
    state_dict = torch.load(pt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state_dict)

    model.to(DEVICE).eval()
    return tokenizer, model, label_encoder, True


# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _infer_single(text: str, tokenizer, model, label_encoder) -> np.ndarray:
    enc = tokenizer(
        clean_text(text),
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    out = model(
        input_ids=enc["input_ids"].to(DEVICE),
        attention_mask=enc["attention_mask"].to(DEVICE)
    )
    logits = out.logits.squeeze(0)
    text_lower = text.lower()
    # POSITIVE BOOST (logits)
    if any(w in text_lower for w in ["good","nice","great","excellent","amazing"]):
        pos_idx = label_encoder.transform(["Positive"])[0]
        logits[pos_idx] += 0.5
    # NEGATIVE BOOST (logits)
    if any(w in text_lower for w in ["bad","worst","terrible","poor","damaged"]):
        neg_idx = label_encoder.transform(["Negative"])[0]
        logits[neg_idx] += 0.5
    probs = torch.softmax(logits, dim=0)
    return probs.cpu().numpy()


@torch.no_grad()
def predict_detailed(text, tokenizer, model, label_encoder):
    classes = label_encoder.classes_

    is_valid, reason = validate_input(text)
    if not is_valid:
        dummy = np.zeros(len(classes))
        return "Invalid", 0, 0.0, dummy, [], False, reason

    text_clean = clean_text(text)

    # SHORT TEXT
    if len(text_clean.split()) <= 5:
        probs = _infer_single(text_clean, tokenizer, model, label_encoder)
        pred_idx = int(np.argmax(probs))
        sentiment = classes[pred_idx]
        confidence = float(probs[pred_idx]) * 100
        rating = RATING_MAP.get(sentiment, 3)
        return sentiment, rating, confidence, probs, [], True, ""

    # CLAUSE LOGIC
    clauses = assign_weights(split_clauses(text_clean))
    results = []
    blended = np.zeros(len(classes))

    for clause_text, conjunction, weight in clauses:
        probs = _infer_single(clause_text, tokenizer, model, label_encoder)
        pred_idx = int(np.argmax(probs))

        results.append(ClauseResult(
            text=clause_text,
            sentiment=classes[pred_idx],
            probs=probs,
            weight=weight,
            conjunction=conjunction,
        ))

        blended += weight * probs

    # NORMALIZE (important)
    if np.sum(blended) > 0:
        blended = blended / np.sum(blended)

    # FALLBACK
    if float(np.max(blended)) < MIN_CONFIDENCE and len(clauses) > 1:
        blended = _infer_single(text_clean, tokenizer, model, label_encoder)
        if np.sum(blended) > 0:
            blended = blended / np.sum(blended)

    final_idx = int(np.argmax(blended))
    sentiment = classes[final_idx]
    confidence = float(blended[final_idx]) * 100
    rating = RATING_MAP.get(sentiment, 3)

    return sentiment, rating, confidence, blended, results, True, ""


@torch.no_grad()
def predict_batch(texts, tokenizer, model, label_encoder, batch_size=64):
    results = []
    classes = list(label_encoder.classes_)

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        enc = tokenizer(
            [clean_text(t) for t in batch],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        out = model(
            input_ids=enc["input_ids"].to(DEVICE),
            attention_mask=enc["attention_mask"].to(DEVICE)
        )

        logits = out.logits

        # APPLY BOOST PER SAMPLE
        probs_list = []
        for j, text in enumerate(batch):
            logit = logits[j].clone()
            text_lower = text.lower()

            if any(w in text_lower for w in ["good","nice","great","excellent","amazing"]):
                pos_idx = label_encoder.transform(["Positive"])[0]
                logit[pos_idx] += 0.5

            if any(w in text_lower for w in ["bad","worst","terrible","poor","damaged"]):
                neg_idx = label_encoder.transform(["Negative"])[0]
                logit[neg_idx] += 0.5

            probs = torch.softmax(logit, dim=0).cpu().numpy()
            probs_list.append(probs)

        for p in probs_list:
            idx = int(np.argmax(p))
            label = label_encoder.inverse_transform([idx])[0]

            results.append({
                "sentiment": label,
                "confidence": float(p[idx]) * 100,
                "prob_pos": float(p[classes.index("Positive")]) if "Positive" in classes else 0,
                "prob_neu": float(p[classes.index("Neutral")]) if "Neutral" in classes else 0,
                "prob_neg": float(p[classes.index("Negative")]) if "Negative" in classes else 0,
            })

    return results


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
for key, default in {
    "history":      [],
    "last_result":  None,
    "last_invalid": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div class="brand-strip">
    <div class="company">◈ Sentiment<span>IQ</span></div>
    <div class="tagline">Flipkart · DistilBERT · NLP</div>
</div>
""", unsafe_allow_html=True)

if "page" not in st.session_state:
    st.session_state["page"] = "Live Inference"

st.sidebar.markdown('<div class="nav-label">Navigation</div>', unsafe_allow_html=True)

NAV_ITEMS = [
    ("Live Inference",   "⚡  Live Inference"),
    ("Dataset Analysis", "📂  Dataset Analysis"),
    ("About",            "ℹ️  About"),
]

for _key, _label in NAV_ITEMS:
    _active_cls = "nav-active-wrap" if st.session_state["page"] == _key else "nav-wrap"
    st.sidebar.markdown(f'<div class="{_active_cls}">', unsafe_allow_html=True)
    if st.sidebar.button(_label, key=f"_nb_{_key}", use_container_width=True):
        st.session_state["page"] = _key
        st.rerun()
    st.sidebar.markdown("</div>", unsafe_allow_html=True)

page = st.session_state["page"]

st.sidebar.markdown('<div class="nav-label">Model Status</div>', unsafe_allow_html=True)

try:
    with st.spinner("Loading model..."):
        tokenizer, model, label_encoder, trained = load_artifacts()
    if trained:
        st.sidebar.markdown('<div class="status-block status-ok">▶ Model weights loaded</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="status-block status-warn">⚠ Using base weights — run training</div>', unsafe_allow_html=True)
except Exception as _load_err:
    st.error(f"Failed to load model: {_load_err}. Run sentiment_analysis.py first.")
    st.stop()

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown("""
<p style="font-size:0.62rem;color:#6b7280;text-align:center;letter-spacing:0.12em;text-transform:uppercase;padding:8px 0;">
  SentimentIQ v1.0 · Industrial NLP
</p>
""", unsafe_allow_html=True)

# Second JS block — runs after sidebar is rendered
st.markdown("""
<script>
(function keepSidebarOpen2() {
    function expand() {
        var sels = [
            '[data-testid="collapsedControl"]',
            '[data-testid="stSidebarCollapsedControl"]',
            'button[data-testid="baseButton-headerNoPadding"]'
        ];
        sels.forEach(function(sel) {
            window.parent.document.querySelectorAll(sel).forEach(function(b) {
                b.style.display    = 'none';
                b.style.visibility = 'hidden';
                b.style.opacity    = '0';
                b.style.width      = '0';
                b.style.height     = '0';
                b.style.overflow   = 'hidden';
                b.style.position   = 'absolute';
                b.style.top        = '-9999px';
                b.style.left       = '-9999px';
            });
        });
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.style.overflow = 'hidden';
            if (sidebar.getAttribute('aria-expanded') === 'false') {
                var openBtns = window.parent.document.querySelectorAll('button[kind="header"]');
                openBtns.forEach(function(b) {
                    if (b.getAttribute('aria-expanded') === 'false') { b.click(); }
                });
            }
        }
    }
    expand();
    setInterval(expand, 500);
})();
</script>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — LIVE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
if page == "Live Inference":
    st.markdown('<div class="page-eyebrow">Live Inference</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Analyze any<br><em>review instantly</em></div>', unsafe_allow_html=True)
    st.markdown('<div class="page-desc">Paste a Flipkart product review. The fine-tuned DistilBERT model classifies it in milliseconds — with clause-level reasoning for complex sentences.</div>', unsafe_allow_html=True)

    if not trained:
        st.warning("⚠ Model weights not found at artifacts/. Run sentiment_analysis.py first.")

    col_in, col_out = st.columns([1.1, 0.9], gap="large")

    with col_in:
        review      = st.text_area("Review text",
            placeholder="e.g. The phone feels premium but the battery life is terrible after 2 months of use...",
            height=160)
        analyze_btn = st.button("Analyze →")

        if st.session_state.history:
            st.markdown('<br><div class="card"><div class="card-title">Recent</div>', unsafe_allow_html=True)
            bmap = {"Positive":"hb-pos","Neutral":"hb-neu","Negative":"hb-neg"}
            rows = "".join([
                f'<div class="hist-row">'
                f'<span class="hist-txt">{h["text"]}</span>'
                f'<span class="hbadge {bmap.get(h["sentiment"],"hb-neu")}">{h["sentiment"]}</span>'
                f'<span class="hist-conf">{round(h["conf"])}%</span>'
                f'</div>'
                for h in st.session_state.history
            ])
            st.markdown(rows + "</div>", unsafe_allow_html=True)

    if analyze_btn:
        if not review.strip():
            st.session_state.last_result  = None
            st.session_state.last_invalid = "empty"
        else:
            sentiment, rating, conf, blended_probs, clause_results, is_valid, reason = \
                predict_detailed(review, tokenizer, model, label_encoder)

            if not is_valid:
                st.session_state.last_result  = None
                st.session_state.last_invalid = reason
            else:
                st.session_state.last_invalid = None
                st.session_state.last_result  = {
                    "sentiment":      sentiment,
                    "rating":         rating,
                    "conf":           conf,
                    "blended_probs":  blended_probs.tolist(),
                    "clause_results": [
                        {"text": cr.text, "sentiment": cr.sentiment,
                         "probs": cr.probs.tolist(), "weight": cr.weight,
                         "conjunction": cr.conjunction}
                        for cr in clause_results
                    ],
                }
                st.session_state.history.insert(0, {
                    "text":      review[:65] + ("…" if len(review) > 65 else ""),
                    "sentiment": sentiment,
                    "conf":      conf,
                })
                st.session_state.history = st.session_state.history[:6]

    with col_out:
        res     = st.session_state.last_result
        inv_rsn = st.session_state.last_invalid

        if inv_rsn:
            reason_messages = {
                "too_few_words":   ("Invalid input",
                                    "Please enter a meaningful word or review."),
                "low_alpha_ratio": ("Unreadable input",
                                    "Too many symbols or numbers. Please type a review in plain English."),
                "repeated_chars":  ("Repeated characters",
                                    "Input looks like noise. Please type an actual product review."),
                "empty":           ("Empty input", "Please enter a review."),
            }
            title, detail = reason_messages.get(inv_rsn, ("Invalid input", "Please enter a valid review."))
            components.html(f"""
            <!DOCTYPE html><html><head><style>
            *{{margin:0;padding:0;box-sizing:border-box;}}
            body{{background:transparent;font-family:'Geist Mono',monospace;}}
            .err{{background:#111114;border:1px solid #3a1a1a;border-left:3px solid #f87171;border-radius:10px;padding:1.25rem 1.5rem;}}
            .ey{{font-size:0.62rem;letter-spacing:0.12em;text-transform:uppercase;color:#f87171;margin-bottom:8px;}}
            .et{{font-size:1rem;font-weight:600;color:#e8e8f0;margin-bottom:6px;}}
            .ed{{font-size:0.78rem;color:#a0a0b0;line-height:1.65;}}
            .eh{{margin-top:14px;padding:10px 14px;background:#1a1a1e;border-radius:8px;font-size:0.72rem;color:#52525e;line-height:1.7;}}
            .eh span{{color:#6ee7b7;}}
            </style></head><body>
            <div class="err">
              <div class="ey">Invalid input</div>
              <div class="et">{title}</div>
              <div class="ed">{detail}</div>
              <div class="eh">
                <span>Try something like:</span><br>
                "The phone feels premium but battery drains fast."<br>
                "Great product, fast delivery and good packaging."<br>
                "Camera is terrible and the screen broke in a week."
              </div>
            </div>
            </body></html>""", height=230, scrolling=False)

        elif res:
            classes   = list(label_encoder.classes_)
            sentiment = res["sentiment"]
            rating    = res["rating"]
            conf      = res["conf"]
            bp        = np.array(res["blended_probs"])
            cr_list   = res["clause_results"]
            c         = sentiment.lower()[:3]
            emo       = {"Positive":"😊","Neutral":"😐","Negative":"😞"}.get(sentiment,"")

            star_html = make_star_dots(rating, total=5)

            def gp(cls):
                return round(float(bp[classes.index(cls)]) * 100) if cls in classes else 0

            pos_p, neu_p, neg_p = gp("Positive"), gp("Neutral"), gp("Negative")

            st.markdown(f"""
            <div class="result-hero">
              <div>
                <div class="rh-label-sm">Detected sentiment</div>
                <div class="rh-sentiment s-{c}">{sentiment}</div>
                <div class="rh-conf">
                  <span>{conf:.1f}% confidence</span>
                  <span style="opacity:0.4;">·</span>
                  {star_html}
                  <span style="font-size:0.68rem;color:#52525e;">{rating}/5</span>
                </div>
              </div>
              <div class="rh-badge rb-{c}">{emo}</div>
            </div>""", unsafe_allow_html=True)

            if len(cr_list) > 1:
                def _tag(conj):
                    if not conj:           return "ct-start",   "START"
                    if conj in CONCESSIVE: return "ct-concess", conj.upper()
                    return "ct-add", conj.upper()
                def _sc(sent):
                    return {"Positive":"cs-pos","Neutral":"cs-neu","Negative":"cs-neg"}.get(sent,"cs-neu")

                rows_html = ""
                for cr in cr_list:
                    tc, tl  = _tag(cr["conjunction"])
                    snippet = cr["text"][:90] + ("…" if len(cr["text"]) > 90 else "")
                    rows_html += (
                        f'<div class="clause-row">'
                        f'<span class="clause-tag {tc}">{tl}</span>'
                        f'<span class="clause-body">{snippet}</span>'
                        f'<span class="clause-sent {_sc(cr["sentiment"])}">{cr["sentiment"]}</span>'
                        f'<span class="clause-weight">w={cr["weight"]:.2f}</span>'
                        f'</div>'
                    )

                components.html(f"""
                <!DOCTYPE html><html><head><style>
                *{{margin:0;padding:0;box-sizing:border-box;}}
                body{{background:transparent;font-family:'Geist Mono',monospace;}}
                .cs{{background:#111114;border:1px solid #1e1e24;border-radius:10px;padding:1.25rem 1.5rem;}}
                .ct{{font-size:0.7rem;text-transform:uppercase;letter-spacing:0.1em;color:#52525e;margin-bottom:1rem;}}
                .clause-row{{display:flex;align-items:flex-start;gap:10px;padding:9px 0;border-bottom:1px solid #1e1e24;font-size:0.74rem;}}
                .clause-row:last-child{{border-bottom:none;padding-bottom:0;}}
                .clause-tag{{flex-shrink:0;padding:2px 8px;border-radius:6px;font-size:0.64rem;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;margin-top:2px;}}
                .ct-start{{background:rgba(96,165,250,0.12);color:#60a5fa;border:1px solid rgba(96,165,250,0.2);}}
                .ct-concess{{background:rgba(248,113,113,0.12);color:#f87171;border:1px solid rgba(248,113,113,0.2);}}
                .ct-add{{background:rgba(167,139,250,0.12);color:#a78bfa;border:1px solid rgba(167,139,250,0.2);}}
                .clause-body{{flex:1;line-height:1.55;color:#a0a0b0;}}
                .clause-sent{{font-size:0.68rem;font-weight:600;padding:1px 7px;border-radius:99px;margin-left:6px;flex-shrink:0;margin-top:2px;}}
                .cs-pos{{background:rgba(74,222,128,0.1);color:#4ade80;}}
                .cs-neu{{background:rgba(251,191,36,0.1);color:#fbbf24;}}
                .cs-neg{{background:rgba(248,113,113,0.1);color:#f87171;}}
                .clause-weight{{font-size:0.68rem;color:#52525e;flex-shrink:0;min-width:36px;text-align:right;margin-top:3px;}}
                </style></head><body>
                <div class="cs">
                  <div class="ct">Clause breakdown &middot; {len(cr_list)} clauses detected</div>
                  {rows_html}
                </div>
                </body></html>""",
                height=56 + len(cr_list) * 52, scrolling=False)

            st.markdown(f"""
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
            </div>""", unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="height:280px;display:flex;flex-direction:column;align-items:center;
                        justify-content:center;gap:10px;opacity:0.3;">
              <div style="font-size:2.5rem;">◈</div>
              <div style="font-size:0.75rem;letter-spacing:0.15em;color:#52525e;">AWAITING INPUT</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset Analysis":
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
                sample_n = st.selectbox("Rows to analyze",
                    [100,200,300,400,500,1000,2000,5000,10000,len(df_raw)], index=1)

            if st.button(f"Analyze {sample_n} rows →"):
                df_sub   = df_raw[text_col].dropna().astype(str).head(sample_n).tolist()
                progress = st.progress(0, text="Classifying…")
                results  = []
                bs = 64
                for i in range(0, len(df_sub), bs):
                    results.extend(predict_batch(df_sub[i:i+bs], tokenizer, model, label_encoder))
                    progress.progress(min((i+bs)/len(df_sub), 1.0),
                                      text=f"Batch {i//bs+1}/{(len(df_sub)-1)//bs+1}")
                progress.empty()

                df_res         = pd.DataFrame(results)
                df_res["text"] = df_sub[:len(df_res)]

                counts = df_res["sentiment"].value_counts()
                total  = len(df_res)
                pos_n  = counts.get("Positive", 0)
                neu_n  = counts.get("Neutral",  0)
                neg_n  = counts.get("Negative", 0)

                st.markdown(f"""
                <div class="stat-grid">
                  <div class="stat-card stat-blue"><div class="stat-label">Total analyzed</div><div class="stat-value">{total:,}</div><div class="stat-sub">reviews</div></div>
                  <div class="stat-card stat-pos"><div class="stat-label">Positive</div><div class="stat-value">{pos_n:,}</div><div class="stat-sub">{pos_n/total*100:.1f}%</div></div>
                  <div class="stat-card stat-neu"><div class="stat-label">Neutral</div><div class="stat-value">{neu_n:,}</div><div class="stat-sub">{neu_n/total*100:.1f}%</div></div>
                  <div class="stat-card stat-neg"><div class="stat-label">Negative</div><div class="stat-value">{neg_n:,}</div><div class="stat-sub">{neg_n/total*100:.1f}%</div></div>
                </div>""", unsafe_allow_html=True)

                fig_pie = go.Figure(go.Pie(
                    labels=["Positive","Neutral","Negative"],
                    values=[pos_n,neu_n,neg_n], hole=0.6,
                    marker_colors=["#4ade80","#fbbf24","#f87171"],
                    textinfo="percent", textfont_size=12, textfont_color="#09090b",
                ))
                fig_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Geist Mono,monospace", color="#52525e", size=11),
                    legend=dict(orientation="h", y=-0.1, font_color="#a0a0b0"),
                    margin=dict(t=10,b=10,l=10,r=10),
                    annotations=[dict(text=f"<b>{total}</b>",x=0.5,y=0.5,
                                      font_size=18,showarrow=False,font_color="#e8e8f0")]
                )
                st.markdown('<div class="card"><div class="card-title">Sentiment distribution</div>', unsafe_allow_html=True)
                st.plotly_chart(fig_pie, use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="card"><div class="card-title">Sample results</div>', unsafe_allow_html=True)
                for tab, sent, css in zip(
                    st.tabs(["Positive","Neutral","Negative"]),
                    ["Positive","Neutral","Negative"], ["pos","neu","neg"]
                ):
                    with tab:
                        sub = df_res[df_res["sentiment"]==sent].head(8)
                        if sub.empty:
                            st.markdown(f'<div style="color:#52525e;font-size:0.78rem;padding:1rem;">No {sent.lower()} reviews found.</div>', unsafe_allow_html=True)
                        for _, row in sub.iterrows():
                            st.markdown(f"""
                            <div class="review-item ri-{css}">
                              <div class="ri-meta">{sent} · {row['confidence']:.1f}% confidence</div>
                              {row['text'][:180]}{'…' if len(row['text'])>180 else ''}
                            </div>""", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.download_button("Download results as CSV",
                    df_res[["text","sentiment","confidence"]].to_csv(index=False),
                    file_name="sentiment_results.csv", mime="text/csv")

        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        st.markdown("""
        <div style="background:#111114;border:1px dashed #2a2a34;border-radius:12px;
                    padding:3rem;text-align:center;margin-top:1rem;">
          <div style="font-size:2rem;margin-bottom:12px;opacity:0.4;">📂</div>
          <div style="font-size:0.8rem;color:#52525e;line-height:1.8;">
            Upload any CSV with a review/text column.<br>
            Works with Flipkart, Amazon, or custom datasets.<br><br>
            <span style="font-size:0.7rem;">Expected column: Summary, Review, text, or any text column</span>
          </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="page-eyebrow">About this project</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Flipkart Review<br><em>Sentiment Analyzer</em></div>', unsafe_allow_html=True)

    col_info, col_metrics = st.columns([1.3, 0.7], gap="large")

    with col_info:
        st.markdown('<div class="info-section"><div class="info-h2">Overview</div></div>', unsafe_allow_html=True)
        st.markdown(
            "This project implements an **end-to-end sentiment analysis system** powered by a "
            "**fine-tuned DistilBERT model** trained on **205,000+ Flipkart product reviews**. "
            "It classifies reviews into **Positive**, **Neutral**, and **Negative** sentiments."
        )
        st.markdown(
            "The system goes beyond basic classification by incorporating a **conjunction-aware "
            "clause analysis pipeline**. In sentences containing contrast (e.g., *but*, *however*), "
            "the latter clause is given higher importance (~65%), enabling more human-like interpretation."
        )
        st.markdown(
            "Robust **input validation** ensures reliability by filtering out meaningless or noisy inputs "
            "such as random characters (`aaaa`), symbols (`../`), or incomplete text."
        )
        st.markdown(
            "Built using **transformer-based transfer learning**, the model achieves strong performance "
            "while maintaining **low-latency inference**, making it suitable for real-time applications."
        )
        st.markdown('<div class="info-section"><div class="info-h2">Key Features</div></div>', unsafe_allow_html=True)
        st.markdown("""
- Fine-tuned DistilBERT — 3-class sentiment classification
- Input validation — blocks symbols, noise, short/garbage input
- Trailing noise-token stripping — fixes "fresh ans" → "fresh"
- Conjunction-aware chunking (but/however/though/yet/whereas…)
- Weighted clause aggregation — concessive 35/65, additive equal
- Low-confidence fallback re-runs full text if blended score < 0.35
- Per-clause breakdown in UI for complex reviews
- CSS dot-based star rating — immune to font glyph rendering bugs
- Batch CSV analysis with donut chart and export
        """)
        st.markdown('<div class="info-section"><div class="info-h2">Tech Stack</div></div>', unsafe_allow_html=True)
        st.markdown("""<div>
          <span class="tech-pill">DistilBERT</span>
          <span class="tech-pill">HuggingFace Transformers</span>
          <span class="tech-pill">PyTorch</span>
          <span class="tech-pill">Streamlit</span>
          <span class="tech-pill">Plotly</span>
          <span class="tech-pill">scikit-learn</span>
          <span class="tech-pill">imbalanced-learn</span>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div class="info-section"><div class="info-h2">Running the Project</div></div>', unsafe_allow_html=True)
        st.markdown("""<div class="code-block">pip install -r requirements.txt
python sentiment_analysis.py   # train once, saves artifacts/
streamlit run app.py           # launch the app</div>""", unsafe_allow_html=True)
        st.markdown('<div class="info-section"><div class="info-h2">Project Structure</div></div>', unsafe_allow_html=True)
        st.markdown("""<div class="code-block">sentiment/
├── app.py
├── sentiment_analysis.py
├── requirements.txt
├── flipkart_product_data.csv
└── artifacts/
    ├── best_bert_sentiment.pt
    ├── label_encoder.pkl
    └── distilbert_tokenizer/</div>""", unsafe_allow_html=True)

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
        <div class="card" style="margin-bottom:12px;">
          <div class="card-title">Chunking config</div>
          <table class="metric-table">
            <tr><th>Type</th><th>Weight</th></tr>
            <tr><td>Concessive (first)</td><td>0.35</td></tr>
            <tr><td>Concessive (last)</td><td style="color:#4ade80;">0.65</td></tr>
            <tr><td>Additive</td><td>Equal</td></tr>
            <tr><td>Min clause length</td><td>4 words</td></tr>
            <tr><td>Confidence fallback</td><td>&lt; 0.45</td></tr>
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
          </table>
        </div>""", unsafe_allow_html=True)
