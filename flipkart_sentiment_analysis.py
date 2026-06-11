# -*- coding: utf-8 -*-
"""
Sentiment_Analysis — DistilBERT on Flipkart product reviews.
"""
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
warnings.filterwarnings('ignore')

# ── Global config ─────────────────────────────────────────────────────────────
DEVICE      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN     = 128
BATCH_SIZE  = 64
EPOCHS      = 2
LR          = 2e-5
MODEL_NAME  = 'distilbert-base-uncased'
print(f"Using device: {DEVICE}")

# ── Short-word whitelist: these 3-or-fewer-char tokens are REAL words ─────────
# Any token ≤3 chars NOT in this set and appearing at the END of input
# is treated as a noise fragment (e.g. 'ans', 'qw', 'zx') and stripped.
VALID_SHORT_WORDS = {
    'a', 'an', 'the', 'is', 'it', 'in', 'on', 'at', 'to', 'do',
    'be', 'by', 'my', 'or', 'as', 'so', 'up', 'of', 'if', 'no',
    'ok', 'yes', 'not', 'but', 'and', 'for', 'are', 'was', 'has',
    'had', 'its', 'too', 'all', 'can', 'did', 'got', 'let', 'put',
    'get', 'use', 'bad', 'lot', 'way', 'now', 'how', 'who', 'why',
    'one', 'two', 'out', 'off', 'old', 'new', 'top', 'bit', 'low',
    'big', 'try', 'buy', 'fit', 'fix', 'cut', 'run', 'per', 'yet',
    'far', 'set', 'own', 'due', 'any', 'may', 'few', 'add', 'act',
    'age', 'ago', 'aid', 'air', 'arm', 'art', 'ask', 'bed', 'box',
    'car', 'day', 'die', 'eat', 'end', 'eye', 'fun', 'hit', 'hot',
    'job', 'key', 'kid', 'lie', 'man', 'mix', 'oil', 'pay', 'red',
    'see', 'sit', 'six', 'say', 'she', 'him', 'her', 'his', 'our',
    'war', 'win', 'bit', 'app', 'pro', 'con', 'fee', 'tip', 'wow',
}

# ══════════════════════════════════════════════════════════════════════════════
# 1. TEXT UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Lowercase, strip special chars, collapse whitespace, and remove trailing
    noise tokens (short non-dictionary fragments like 'ans', 'qw', 'zx')
    that corrupt sentence-final sentiment signal.
    """
    if pd.isna(text) or not text:
        return ""
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z0-9!?.,'\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Strip trailing noise tokens: ≤3 alpha chars not in the whitelist
    tokens = text.split()
    while tokens:
        last = tokens[-1].rstrip(".,!?'")   # strip trailing punctuation for check
        if last.isalpha() and len(last) <= 3 and last not in VALID_SHORT_WORDS:
            tokens.pop()
        else:
            break
    return ' '.join(tokens)


def validate_input(raw: str) -> Tuple[bool, str]:
    """
    Improved validation:
    - Allows short meaningful inputs (e.g., "good", "bad")
    - Blocks garbage (e.g., "abcd", "....", "123")
    """

    stripped = raw.strip()
    if not stripped:
        return False, "empty"

    cleaned = clean_text(stripped)

    # Extract meaningful alphabetic words (len >= 2)
    alpha_words = [
        w for w in cleaned.split()
        if re.fullmatch(r'[a-z]{2,}', w.rstrip(".,!?'"))
    ]

    #  No real words → invalid
    if len(alpha_words) == 0:
        return False, "too_few_words"

    #  Single word but not meaningful → block garbage like "abcd"
    COMMON_WORDS = {
        "good","bad","nice","great","excellent","poor",
        "worst","amazing","average","ok","fine","awesome"
    }

    words = cleaned.split()
    if len(words) == 1 and words[0] not in COMMON_WORDS:
        return False, "too_few_words"

    #  Too many symbols / numbers
    non_space = re.sub(r'\s', '', stripped)
    if non_space:
        alpha_ratio = sum(c.isalpha() for c in non_space) / len(non_space)
        if alpha_ratio < 0.40:
            return False, "low_alpha_ratio"

    #  Repeated characters (aaaa, ....)
    if re.fullmatch(r'(.)\1{4,}', stripped.replace(" ", "")):
        return False, "repeated_chars"

    return True, ""
# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA LOADING & PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH='C:\Github_Projects\sentiment\flipkart_product_copy.csv'
df = pd.read_csv(DATA_PATH, encoding='latin1')
df['Rate'] = pd.to_numeric(df['Rate'], errors='coerce')
df = df.dropna(subset=['Rate'])

def map_sentiment(rate):
    if rate > 3:   return 'Positive'
    elif rate < 3: return 'Negative'
    else:          return 'Neutral'

df['sentiment']    = df['Rate'].apply(map_sentiment)
df['text']         = df['Summary'].fillna('') + ' ' + df['Review'].fillna('')
df['cleaned_text'] = df['text'].apply(clean_text)

print("Class distribution:")
print(df['sentiment'].value_counts())

label_encoder = LabelEncoder()
df['label']   = label_encoder.fit_transform(df['sentiment'])
NUM_CLASSES   = len(label_encoder.classes_)
print(f"Classes: {label_encoder.classes_}")

# ── Train / val / test split ──────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned_text'].values, df['label'].values,
    test_size=0.15, random_state=42, stratify=df['label']
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train,
    test_size=0.12, random_state=42, stratify=y_train
)

# ── Oversample minority classes ───────────────────────────────────────────────
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train.reshape(-1, 1), y_train)
X_train_res = X_train_res.flatten()
print(f"After oversampling: {Counter(y_train_res)}")

# ══════════════════════════════════════════════════════════════════════════════
# 3. DATASET & DATALOADERS
# ══════════════════════════════════════════════════════════════════════════════

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label':          torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = SentimentDataset(X_train_res, y_train_res, tokenizer, MAX_LEN)
val_dataset   = SentimentDataset(X_val,       y_val,       tokenizer, MAX_LEN)
test_dataset  = SentimentDataset(X_test,      y_test,      tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL, LOSS, OPTIMISER
# ══════════════════════════════════════════════════════════════════════════════

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=NUM_CLASSES
).to(DEVICE)

class_counts  = np.bincount(y_train_res)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(DEVICE)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scaler    = GradScaler()

total_steps = len(train_loader) * EPOCHS
scheduler   = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,
    num_training_steps=total_steps
)

# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def run_epoch(model, loader, criterion, optimizer=None, scheduler=None,
              scaler=None, train=True):
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0, [], []

    for i, batch in enumerate(loader):
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels         = batch['label'].to(DEVICE)
        if train:
            optimizer.zero_grad()
            with autocast():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss    = criterion(outputs.logits, labels)

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if i % 100 == 0:
            print(f"  Batch {i}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(loader), accuracy_score(all_labels, all_preds), all_preds, all_labels


history      = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
best_val_acc = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'='*50}\nEPOCH {epoch}/{EPOCHS}\n{'='*50}")

    tr_loss, tr_acc, _, _ = run_epoch(
        model, train_loader, criterion, optimizer, scheduler, scaler, train=True
    )
    vl_loss, vl_acc, _, _ = run_epoch(model, val_loader, criterion, train=False)

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)

    print(f"\nEpoch {epoch}/{EPOCHS} | "
          f"Train Loss: {tr_loss:.4f}  Train Acc: {tr_acc:.4f} | "
          f"Val Loss: {vl_loss:.4f}  Val Acc: {vl_acc:.4f}")

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), 'best_bert_sentiment.pt')
        print("✓ Best model saved!")

# ── Evaluation ────────────────────────────────────────────────────────────────
model.load_state_dict(torch.load('best_bert_sentiment.pt', map_location=DEVICE))
_, test_acc, y_pred, y_true = run_epoch(model, test_loader, criterion, train=False)

print(f"\nTest Accuracy: {test_acc:.4f}")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

plt.figure(figsize=(8, 6))
sns.heatmap(
    confusion_matrix(y_true, y_pred), annot=True, fmt='d', cmap='Blues',
    xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_
)
plt.title('Confusion Matrix - DistilBERT')
plt.ylabel('True Label'); plt.xlabel('Predicted Label')
plt.tight_layout(); plt.savefig('confusion_matrix.png'); plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, (tr, vl), title in zip(
    axes, [('train_loss', 'val_loss'), ('train_acc', 'val_acc')], ['Loss', 'Accuracy']
):
    ax.plot(history[tr], label='Train', marker='o')
    ax.plot(history[vl], label='Val',   marker='s')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('training_curves.png'); plt.show()

joblib.dump(label_encoder, 'label_encoder.pkl')
tokenizer.save_pretrained('distilbert_tokenizer/')
print("All artifacts saved!")

# ══════════════════════════════════════════════════════════════════════════════
# 6. CONJUNCTION-AWARE CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

CONCESSIVE = {
    "but", "however", "though", "although", "even though",
    "yet", "nevertheless", "nonetheless", "despite", "whereas",
    "while", "still", "that said", "having said that",
}
ADDITIVE = {
    "and", "also", "moreover", "furthermore", "in addition",
    "besides", "plus", "as well",
}

CONCESSIVE_WEIGHTS = {"first": 0.35, "last": 0.65}
ADDITIVE_WEIGHT    = 1.0


@dataclass
class ClauseResult:
    text:        str
    sentiment:   str
    probs:       np.ndarray
    weight:      float
    conjunction: str   # '' for the first clause


def split_clauses(text: str) -> List[Tuple[str, str]]:
    all_conj = sorted(CONCESSIVE | ADDITIVE, key=len, reverse=True)
    pattern  = (r'(?<![a-z])('
                + '|'.join(re.escape(c) for c in all_conj)
                + r')(?![a-z])')
    parts = re.split(pattern, text, flags=re.IGNORECASE)

    clauses: List[Tuple[str, str]] = []
    current_conj = ""
    buffer       = ""
    i = 0
    while i < len(parts):
        chunk = parts[i].strip()
        if not chunk:
            i += 1
            continue
        if chunk.lower() in {c.lower() for c in all_conj}:
            current_conj = chunk.lower()
            i += 1
            continue
        combined   = (buffer + " " + chunk).strip() if buffer else chunk
        word_count = len(combined.split())
        if word_count < 4 and i + 2 < len(parts):
            buffer = combined
        else:
            clauses.append((combined, current_conj))
            buffer = ""; current_conj = ""
        i += 1

    if buffer:
        if clauses:
            last_text, last_conj = clauses[-1]
            clauses[-1] = (last_text + " " + buffer, last_conj)
        else:
            clauses.append((buffer, ""))
    return clauses if clauses else [(text, "")]


def assign_weights(clauses: List[Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    if len(clauses) == 1:
        return [(clauses[0][0], clauses[0][1], 1.0)]
    has_concessive = any(c.lower() in CONCESSIVE for _, c in clauses[1:])
    raw_weights = []
    for idx, (_, conj) in enumerate(clauses):
        if idx == 0:
            raw_weights.append(CONCESSIVE_WEIGHTS["first"] if has_concessive else ADDITIVE_WEIGHT)
        else:
            raw_weights.append(CONCESSIVE_WEIGHTS["last"] if conj.lower() in CONCESSIVE else ADDITIVE_WEIGHT)
    total = sum(raw_weights)
    return [(clauses[i][0], clauses[i][1], raw_weights[i] / total) for i in range(len(clauses))]


# ══════════════════════════════════════════════════════════════════════════════
# 7. SENTIMENT ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class SentimentAnalyzer:
    MIN_CONFIDENCE = 0.35

    def __init__(self, model, tokenizer, label_encoder):
        self.model         = model.eval().to(DEVICE)
        self.tokenizer     = tokenizer
        self.label_encoder = label_encoder

    @torch.no_grad()
    def _infer(self, text: str) -> np.ndarray:
        encoding = self.tokenizer(
            text, max_length=MAX_LEN, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        outputs = self.model(
            input_ids=encoding['input_ids'].to(DEVICE),
            attention_mask=encoding['attention_mask'].to(DEVICE)
        )
        logits = outputs.logits.squeeze(0)
        text_lower = text.lower()
        # POSITIVE BOOST (logits)
        if any(w in text_lower for w in ["good","nice","great","excellent","amazing"]):
            pos_idx = self.label_encoder.transform(["Positive"])[0]
            logits[pos_idx] += 0.5
        # NEGATIVE BOOST (logits)
        if any(w in text_lower for w in ["bad","worst","terrible","damaged","poor"]):
            neg_idx = self.label_encoder.transform(["Negative"])[0]
            logits[neg_idx] += 0.5
        probs = torch.softmax(logits, dim=0)
        return probs.cpu().numpy()

    @torch.no_grad()
    def predict_sentiment(self, text: str, verbose: bool = False) -> Tuple[str, int, float]:
            is_valid, reason = validate_input(text)
            if not is_valid:
                return "Invalid", 0, 0.0

            text = clean_text(text)

            # SHORT TEXT
            if len(text.split()) <= 5:
                probs = self._infer(text)

                pred_idx = int(np.argmax(probs))
                prediction = self.label_encoder.classes_[pred_idx]
                confidence = float(probs[pred_idx]) * 100
                rating = {"Negative": 1, "Neutral": 3, "Positive": 5}[prediction]
                return prediction, rating, confidence

            # LONG TEXT
            if len(text.split()) > 25:
                probs = self._infer(text)

                pred_idx = int(np.argmax(probs))
                prediction = self.label_encoder.classes_[pred_idx]
                confidence = float(probs[pred_idx]) * 100
                rating = {"Negative": 1, "Neutral": 3, "Positive": 5}[prediction]
                return prediction, rating, confidence

            # CLAUSE LOGIC
            clauses = assign_weights(split_clauses(text))
            clauses = clauses[:3]

            blended = np.zeros(NUM_CLASSES)

            for clause_text, conjunction, weight in clauses:
                probs = self._infer(clause_text)
                blended += weight * probs

            # FALLBACK
            if float(np.max(blended)) < self.MIN_CONFIDENCE and len(clauses) > 1:
                blended = self._infer(text)

            pred_idx = int(np.argmax(blended))
            prediction = self.label_encoder.classes_[pred_idx]
            confidence = float(blended[pred_idx]) * 100
            rating = {"Negative": 1, "Neutral": 3, "Positive": 5}[prediction]

            return prediction, rating, confidence
    
    @torch.no_grad()
    def predict_detailed(self, text: str) -> Tuple[str, float, List[ClauseResult]]:
        is_valid, reason = validate_input(text)
        if not is_valid:
            return "Invalid", 0.0, []

        text = clean_text(text)

        # SHORT TEXT
        if len(text.split()) <= 5:
            probs = self._infer(text)
            pred_idx = int(np.argmax(probs))
            return self.label_encoder.classes_[pred_idx], float(probs[pred_idx]) * 100, []

        clauses = assign_weights(split_clauses(text))
        results = []
        blended = np.zeros(NUM_CLASSES)

        for clause_text, conjunction, weight in clauses:
            probs = self._infer(clause_text)
            pred_idx = int(np.argmax(probs))

            results.append(ClauseResult(
                text=clause_text,
                sentiment=self.label_encoder.classes_[pred_idx],
                probs=probs,
                weight=weight,
                conjunction=conjunction,
            ))

            blended += weight * probs

        # NORMALIZE (IMPORTANT)
        blended = blended / np.sum(blended)

        # FALLBACK
        if float(np.max(blended)) < self.MIN_CONFIDENCE and len(clauses) > 1:
            blended = self._infer(text)
            blended = blended / np.sum(blended)

        final_idx = int(np.argmax(blended))
        return self.label_encoder.classes_[final_idx], float(blended[final_idx]) * 100, results

analyzer = SentimentAnalyzer(model, tokenizer, label_encoder)


def user_interface():
    print("\n" + "="*58)
    print("  Flipkart Review Sentiment Analyzer [DistilBERT]")
    print("="*58)
    print("Enter a product review (or 'quit' to exit)\n")
    emoji = {"Positive": "😊", "Neutral": "😐", "Negative": "😞", "Invalid": "⚠️"}

    while True:
        user_input = input("Review: ").strip()
        if user_input.lower() in ('quit', 'exit'):
            print("Goodbye!"); break
        if not user_input:
            print("Please enter a valid review.\n"); continue

        is_valid, reason = validate_input(user_input)
        if not is_valid:
            msgs = {
                "too_few_words":   "Please enter at least 3 meaningful words.",
                "low_alpha_ratio": "Too many symbols/numbers. Please type a review in plain English.",
                "repeated_chars":  "Input looks like noise. Please type an actual review.",
                "empty":           "Please enter a review.",
            }
            print(f"\n  ⚠ Invalid input — {msgs.get(reason, 'Please enter a valid review.')}\n")
            continue

        sentiment, confidence, clause_results = analyzer.predict_detailed(user_input)
        rating = {"Negative": 1, "Neutral": 3, "Positive": 5}.get(sentiment, 0)
        bar    = "■" * rating + "□" * (5 - rating)

        if len(clause_results) > 1:
            print("\n── Clause breakdown ────────────────────────────────────")
            for cr in clause_results:
                tag  = f"[{cr.conjunction or 'start':>14}]"
                conf = float(np.max(cr.probs)) * 100
                print(f"  {tag}  w={cr.weight:.2f}  {cr.sentiment:<8}  ({conf:.1f}%)  '{cr.text[:55]}'")

        print("── Result ──────────────────────────────────────────────")
        print(f"  Sentiment  : {sentiment} {emoji.get(sentiment, '')}")
        print(f"  Rating     : [{bar}] {rating}/5")
        print(f"  Confidence : {confidence:.1f}%")
        print("─" * 56 + "\n")


if __name__ == "__main__":
    user_interface()
