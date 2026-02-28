# Flipkart Review Sentiment Analyzer 

A deep learning-based sentiment analysis system fine-tuned on 200k+ Flipkart product reviews.

##  Overview
Classifies product reviews into **Positive**, **Neutral**, or **Negative** sentiment using a fine-tuned DistilBERT transformer model.

## Tech Stack
- **Model:** DistilBERT (HuggingFace Transformers)
- **Framework:** PyTorch
- **Dataset:** Flipkart Product Reviews (~205k rows)

##  Key Features
- Transfer learning with DistilBERT fine-tuning
- fp16 mixed precision training (2x faster)
- Random oversampling for class imbalance
- Real-time inference with confidence scoring

##  Results
| Metric | Score |
|--------|-------|
| Train Accuracy | 94.72% |
| Validation Accuracy | 96.27% |
| Epochs | 2 |

##  Usage
```python
analyzer = SentimentAnalyzer(model, tokenizer, label_encoder)
sentiment, rating, confidence = analyzer.predict_sentiment("Great product!")
# Output → Positive  | Rating: [■■■■■] 5/5 | Confidence: 97.2%
```

##  Project Structure
```
├── sentiment_analyzer.py   # Main training script
├── best_bert_sentiment.pt  # Saved model weights
├── label_encoder.pkl       # Label encoder
├── distilbert_tokenizer/   # Saved tokenizer
├── confusion_matrix.png    # Evaluation plot
└── training_curves.png     # Loss/Accuracy curves
```
