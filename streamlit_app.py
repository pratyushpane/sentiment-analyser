
import streamlit as st
import joblib
import pandas as pd
from utils import remove_stopwords_and_get_unique, analyze_sentiment

# ✅ Add this block for NLTK data download
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load models and vectorizer
dt_model = joblib.load("model_dt.pkl")
rf_model = joblib.load("model_rf.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def preprocess_single_sentence(sentence):
    tokens = remove_stopwords_and_get_unique(sentence)
    sentiment_scores = analyze_sentiment(sentence)

    token_feats = vectorizer.transform([' '.join(tokens)])

    features = pd.concat([
        pd.DataFrame(token_feats.toarray(), columns=vectorizer.get_feature_names_out()),
        pd.DataFrame([[len(tokens), sentiment_scores['compound'], sentiment_scores['pos'],
                       sentiment_scores['neg'], sentiment_scores['neu']]],
                     columns=['token_count', 'compound_score', 'positive_score', 'negative_score', 'neutral_score'])
    ], axis=1)

    return features

def categorize_sentence(sentence, model):
    return model.predict(preprocess_single_sentence(sentence))[0]

# Streamlit UI
st.title("💬 Sentiment Classifier")
user_input = st.text_area("Enter your sentence here:")

if st.button("Predict"):
    dt_pred = categorize_sentence(user_input, dt_model)
    rf_pred = categorize_sentence(user_input, rf_model)
    st.success(f"Decision Tree Prediction: **{dt_pred}**")
    st.success(f"Random Forest Prediction: **{rf_pred}**")
