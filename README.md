# Flipkart Review Sentiment Analyzer

This project implements a sentiment analysis system using a fine-tuned DistilBERT model trained on more than 200,000 Flipkart product reviews. The model classifies reviews into three categories: Positive, Neutral, and Negative.

## Overview

The goal of this project is to build a practical NLP application that understands the context of customer reviews and predicts sentiment accurately. It uses transfer learning with a transformer-based architecture to achieve strong performance while keeping inference efficient.

## Tech Stack

* Model: DistilBERT (HuggingFace Transformers)
* Framework: PyTorch
* Dataset: Flipkart Product Reviews (~200k rows)
* Frontend: Streamlit (for real-time interaction)

## Key Features

* Fine-tuned DistilBERT model for multi-class sentiment classification
* Handles class imbalance using random oversampling
* Uses mixed precision training to improve training speed
* Provides real-time predictions with confidence scores
* Converts sentiment output into a star rating format

## Results

| Metric              | Score  |
| ------------------- | ------ |
| Train Accuracy      | 94.72% |
| Validation Accuracy | 96.27% |
| Epochs              | 2      |

## Usage

```python
analyzer = SentimentAnalyzer(model, tokenizer, label_encoder)
sentiment, rating, confidence = analyzer.predict_sentiment("Great product!")
```

The model returns the predicted sentiment along with a confidence score and an equivalent rating.
Link for best_bert_sentiment.pt
-https://drive.google.com/file/d/1jORjSuKFrf-O18r-Op1rSxr36n5om1eH/view?usp=sharing

## Running the Project

1. Install dependencies

```
pip install -r requirements.txt
```

2. Run the Streamlit application

```
streamlit run app.py
```

## Project Structure

```
sentiment/
├── app.py
├── sentiment_analysis.py
├── artifacts/
│   ├── best_bert_sentiment.pt
│   ├── label_encoder.pkl
│   ├── distilbert_tokenizer/
├── flipkart_product_copy.csv
├── requirements.txt
```

## Author 
```
Chirag Sharma 
MSIT
```
