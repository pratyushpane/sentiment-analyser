# utils.py
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Initialize only once
sid = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

def analyze_sentiment(text):
    return sid.polarity_scores(text)

def remove_stopwords_and_get_unique(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    tokens = word_tokenize(text.lower())
    return list(set(word for word in tokens if word.isalnum() and word not in stop_words))

