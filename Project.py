import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import nltk
import re

nltk.download('stopwords')
nltk.download('punkt')

# Load and Clean Data

df = pd.read_csv("Reviews.csv")
df = df[['Text', 'Score']].dropna()
df = df[df['Score'].between(1, 5)]
df = df.sample(10000, random_state=42)

# Basic clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join([word for word in tokens if word not in stopwords.words('english')])

df['Cleaned_Text'] = df['Text'].apply(clean_text)


# Sentiment Label (from rating)

def map_sentiment(score):
    if score >= 4:
        return 'Positive'
    elif score == 3:
        return 'Neutral'
    else:
        return 'Negative'

df['Sentiment_Label'] = df['Score'].apply(map_sentiment)


# TF-IDF for Cleaned Reviews

tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
X_text = tfidf.fit_transform(df['Cleaned_Text'])

# Sentiment Encoding
sent_le = LabelEncoder()
sent_encoded = sent_le.fit_transform(df['Sentiment_Label']).reshape(-1, 1)

# Combine text TF-IDF + sentiment for rating model
from scipy.sparse import hstack
X_rating = hstack([X_text, sent_encoded])


# Rating Prediction

y_rating = df['Score'] - 1  # Classes: 0–4
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_rating, y_rating, test_size=0.2, random_state=42)

rating_clf = RandomForestClassifier(n_estimators=200, max_depth=25, class_weight='balanced', random_state=42)
rating_clf.fit(X_train_r, y_train_r)
y_pred_rating = rating_clf.predict(X_test_r)
print("\n📊 Improved Rating Classifier (with Sentiment)")
print("Accuracy:", accuracy_score(y_test_r, y_pred_rating))
print(classification_report(y_test_r, y_pred_rating))

# Sentiment Prediction (Supervised)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_text, sent_encoded.ravel(), test_size=0.2, random_state=42)
sentiment_clf = LogisticRegression(max_iter=600, class_weight='balanced')
sentiment_clf.fit(X_train_s, y_train_s)
y_pred_sentiment = sentiment_clf.predict(X_test_s)
print("\n📊 Sentiment Classifier (Supervised)")
print("Accuracy:", accuracy_score(y_test_s, y_pred_sentiment))
print(classification_report(y_test_s, y_pred_sentiment, target_names=sent_le.classes_))


# Topic Modeling with BERTopic

print("\n⏳ Extracting Topics with BERTopic...")
docs = df['Cleaned_Text'].tolist()
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
topic_model = BERTopic(embedding_model=embedding_model, verbose=False)
topics, _ = topic_model.fit_transform(docs)

# Show top 5 topics
print("\n📌 Improved Topics (BERTopic):")
top_topics = topic_model.get_topic_info().head(6)  # Top 5 + -1 (outliers)
print(top_topics[['Topic', 'Count', 'Name']])

# Show top words per topic
for topic_num in top_topics['Topic'].tolist()[1:6]:
    print(f"\n🔹 Topic {topic_num}:")
    for word, score in topic_model.get_topic(topic_num):
        print(f"   {word}: {score:.4f}")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ---------- Confusion Matrix for Rating ----------
cm_rating = confusion_matrix(y_test_r, y_pred_rating)
plt.figure(figsize=(6,5))
sns.heatmap(cm_rating, annot=True, fmt='d', cmap='Blues')
plt.title("📊 Confusion Matrix: Rating Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---------- Confusion Matrix for Sentiment ----------
cm_sentiment = confusion_matrix(y_test_s, y_pred_sentiment)
plt.figure(figsize=(6,5))
sns.heatmap(cm_sentiment, annot=True, fmt='d', cmap='Greens',
            xticklabels=sent_le.classes_, yticklabels=sent_le.classes_)
plt.title("📊 Confusion Matrix: Sentiment Classifier")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# BERTopic interactive 2D topic clustering
print("\n📈 Generating topic cluster visualization...")
topic_model.visualize_topics().show()

import joblib

# Save rating and sentiment classifiers
joblib.dump(rating_clf, "rating_model.pkl")
joblib.dump(sentiment_clf, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(sent_le, "sentiment_label_encoder.pkl")

# Save BERTopic
topic_model.save("bertopic_model")

