# Sentiment-Analysis

# 📦 Flipkart Reviews Sentiment Analysis

This project analyzes customer reviews scraped from Flipkart to determine whether the sentiment is **positive** or **negative**, using Natural Language Processing (NLP) and Logistic Regression.

---

## 🔧 Technologies Used

- **Python**
- **Jupyter Notebook**
- **BeautifulSoup** – for web scraping
- **NLTK** – for text preprocessing
- **Scikit-learn** – for feature extraction and model building

---

## ❓ Why This Project?

Online product reviews significantly influence buyer decisions. However, manually reading all reviews can be time-consuming. This project automates the process of classifying reviews into sentiments, helping sellers or users get quick insights into public opinion.

---

## ⚙️ How It Works

### 1. **Web Scraping**
- Flipkart product reviews are scraped using `requests` and `BeautifulSoup`.
- Extracts review text from specific HTML tags.

### 2. **Text Preprocessing**
- Converts text to lowercase.
- Removes punctuation and stopwords.
- Applies stemming to normalize words.
- Handles negations like “not good” → “NOT_good”.

### 3. **Feature Extraction**
- Uses `TfidfVectorizer` to convert cleaned text into numerical vectors suitable for ML models.

### 4. **Model Training**
- Trains a `LogisticRegression` model using TF-IDF features.
- Data is split into training and testing sets to validate accuracy.

### 5. **Prediction**
- Takes new reviews as input and classifies them as either **positive** or **negative**.

---

## 📈 Output

- Shows model accuracy.
- Predicts sentiment for new reviews.
- Cleaned dataset and vectorized input are visible for transparency.

---

## 🧠 Skills Demonstrated

- Web scraping
- NLP preprocessing
- Text vectorization (TF-IDF)
- Binary classification with Logistic Regression
- Model evaluation and testing

---

## 🚀 Future Improvements

- Add support for multiple product categories.
- Integrate GUI or web interface.
- Use more advanced models like SVM or Transformers for better accuracy.

