#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from imblearn.over_sampling import SMOTE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from spacy.lang.en import English





def preprocess_single_sentence(sentence):
    # Apply the same preprocessing steps as for the training data
    tokens = remove_stopwords_and_get_unique(sentence)
    pos_tags = get_pos_tags(tokens)
    sentiment_scores = analyze_sentiment(sentence)
    
    # Create features
    token_features = vectorizer.transform([' '.join(tokens)])
    pos_feature_names = pos_features.columns  
    # Handle pos_features correctly
    if pos_tags:
        pos_features_dict = count_pos_tags(pos_tags)
        pos_features_series = pd.Series(pos_features_dict).reindex(pos_feature_names, fill_value=0)
    else:
        pos_features_series = pd.Series(index=pos_feature_names).fillna(0)
    
    # Combine features
    features = pd.concat([
        pd.DataFrame(token_features.toarray(), columns=vectorizer.get_feature_names_out()),
        pd.DataFrame([[len(tokens), sentiment_scores['compound'], sentiment_scores['pos'], 
                       sentiment_scores['neg'], sentiment_scores['neu']]], 
                     columns=['token_count', 'compound_score', 'positive_score', 'negative_score', 'neutral_score']),
        pos_features_series.to_frame().T
    ], axis=1)
    
    return features


def categorize_sentence(sentence, model):
    features = preprocess_single_sentence(sentence)
    prediction = model.predict(features)[i]
    return prediction[0]

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')

# wordcloud function
def generate_wordcloud(words):
    wordcloud = WordCloud(width=800, height=600).generate(" ".join(words))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud of Comments")
    plt.show()

# Read data from CSV file
df = pd.read_csv("train_data.csv")

# Function to perform sentiment analysis
def analyze_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)
    return sentiment_scores

# Tokenization and stop word removal function
def remove_stopwords_and_get_unique(text):
    if pd.isna(text) or not isinstance(text, str):
        return []
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(str(text).lower())
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return list(set(filtered_tokens))

def get_pos_tags(tokens):
    return pos_tag(tokens)

# Preprocess the data
df['text'] = df['text'].fillna("").astype(str)
df['unique_tokens'] = df['text'].apply(remove_stopwords_and_get_unique)
df['token_count'] = df['unique_tokens'].apply(len)
df['pos_tags'] = df['unique_tokens'].apply(get_pos_tags)

# Apply sentiment analysis
df['sentiment_scores'] = df['text'].apply(analyze_sentiment)
df['compound_score'] = df['sentiment_scores'].apply(lambda x: x['compound'])
df['positive_score'] = df['sentiment_scores'].apply(lambda x: x['pos'])
df['negative_score'] = df['sentiment_scores'].apply(lambda x: x['neg'])
df['neutral_score'] = df['sentiment_scores'].apply(lambda x: x['neu'])

# Creating a bar plot
status_count = df["labeled_status"].value_counts()

plt.figure(figsize=(8, 6))
status_count.plot(kind='bar', color=['blue', 'orange', 'green'])
plt.title('Sentiment Count')
plt.xlabel("")
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Creating a word cloud
all_unique_tokens = df['unique_tokens'].explode().dropna().tolist()
all_unique_tokens = [token for token in all_unique_tokens if isinstance(token, str)]
generate_wordcloud(all_unique_tokens)

# Drop unnecessary column
df = df.drop("sentiment_scores", axis=1)

# Save preprocessed data
df.to_csv("kosis_preprocessed.csv", index=False)
print("Preprocessed data saved to kosis_preprocessed.csv")

# Prepare the features (X) and target variable (y)
vectorizer = CountVectorizer(max_features=1000)  # limit to top 1000 tokens
token_features = vectorizer.fit_transform(df['unique_tokens'].apply(lambda x: ' '.join(x)))

def count_pos_tags(tags):
    return dict(Counter(tag for word, tag in tags))

pos_features = df['pos_tags'].apply(count_pos_tags).apply(pd.Series).fillna(0)

# Combine all features
X = pd.concat([
    pd.DataFrame(token_features.toarray(), columns=vectorizer.get_feature_names_out()), 
    df[['token_count', 'compound_score', 'positive_score', 'negative_score', 'neutral_score']], 
    pos_features
], axis=1)

y = df['labeled_status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Original dataset shape:", Counter(y_train))
print("Resampled dataset shape:", Counter(y_train_resampled))

# Create and train the decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_resampled, y_train_resampled)

# Create and train the random forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set for both models
dt_pred = dt_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluate both models
print("\nDecision Tree Results:")
dt_accuracy = accuracy_score(y_test, dt_pred)
print(f"Model Accuracy: {dt_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, dt_pred))

print("\nRandom Forest Results:")
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"Model Accuracy: {rf_accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

# Feature importance for both models
dt_feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt_model.feature_importances_})
dt_feature_importance = dt_feature_importance.sort_values('importance', ascending=False)

rf_feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf_model.feature_importances_})
rf_feature_importance = rf_feature_importance.sort_values('importance', ascending=False)

print("\nTop 10 Feature Importances (Decision Tree):")
print(dt_feature_importance.head(10))

print("\nTop 10 Feature Importances (Random Forest):")
print(rf_feature_importance.head(10))
# Example usage of the categorize_sentence function
sample_sentence = "This is the most worst and bad part of the video."
dt_category = categorize_sentence(sample_sentence, dt_model)
rf_category = categorize_sentence(sample_sentence, rf_model)

print("\nSample sentence:", sample_sentence)
print("Decision Tree categorization:", dt_category)
print("Random Forest categorization:", rf_category)



# In[13]:whi


df.columns


