import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

# Ensure nltk resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text)

def remove_stopwords_and_get_unique(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    return list(set(word for word in tokens if word.isalnum() and word not in stop_words))
