import streamlit as st
import numpy as np
import joblib
from textblob import TextBlob
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from bertopic import BERTopic


nltk.download('punkt')
nltk.download('stopwords')

# ---------------- Load Models ----------------
rating_model = joblib.load("rating_model.pkl")
sentiment_model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
sent_le = joblib.load("sentiment_label_encoder.pkl")
topic_model = BERTopic.load("bertopic_model")

# ---------------- Helpers ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join([w for w in tokens if w not in stopwords.words('english')])

# def analyze_review(review_text):
#     cleaned = clean_text(review_text)
#     tfidf_vec = tfidf.transform([cleaned])

#     # Predict sentiment first (needed as feature for rating model)
#     sentiment_idx = sentiment_model.predict(tfidf_vec)[0]
#     sentiment_feature = np.array([[sentiment_idx]])  # shape: (1,1)

#     # Combine tf-idf + sentiment feature
#     from scipy.sparse import hstack
#     rating_input = hstack([tfidf_vec, sentiment_feature])

#     # Predict rating (0–4 → 1–5)
#     rating = rating_model.predict(rating_input)[0] + 1

#     # Decode sentiment
#     sentiment = sent_le.inverse_transform([sentiment_idx])[0]

#     # Topic
#     topic_idx = topic_model.transform([cleaned])  # this returns a list
#     topic_id = topic_idx[0]  # extract the first element (int)
#     topic_words = topic_model.get_topic(topic_id)

#     topic_words = topic_model.get_topic(topic_idx)

#     return rating, sentiment, topic_idx, topic_words

def analyze_review(review_text):
    cleaned = clean_text(review_text)
    tfidf_vec = tfidf.transform([cleaned])

    # Predict sentiment
    sentiment_idx = sentiment_model.predict(tfidf_vec)[0]
    sentiment_feature = np.array([[sentiment_idx]])
    from scipy.sparse import hstack
    rating_input = hstack([tfidf_vec, sentiment_feature])

    # Predict rating
    rating = rating_model.predict(rating_input)[0] + 1
    sentiment = sent_le.inverse_transform([sentiment_idx])[0]

    # Predict topic safely
    topic_list = topic_model.transform([cleaned])  # might be [[3]] or [3]
    if isinstance(topic_list[0], list):
        topic_id = topic_list[0][0]  # flatten
    else:
        topic_id = topic_list[0]

    topic_words = topic_model.get_topic(topic_id)

    # Get topic name
    topic_name = topic_model.get_topic_info()
    topic_name = topic_name[topic_name["Topic"] == topic_id]["Name"].values[0]


    return rating, sentiment, topic_name, topic_words

# ---------------- UI ----------------
st.set_page_config(page_title="Review Analyzer", layout="centered")
st.title("📝 Amazon Review Analyzer")
st.write("Paste a product review to predict its **rating**, **sentiment**, and **topic**.")

user_input = st.text_area("Enter your review text:", height=200)

if st.button("🔍 Analyze Review"):
    if user_input.strip():
        with st.spinner("Analyzing..."):
            rating, sentiment, topic_name, topic_words = analyze_review(user_input)


        st.success("✅ Analysis Complete!")

        st.subheader("📊 Predicted Rating:")
        st.write(f"⭐️ **{rating} out of 5**")

        st.subheader("😊 Sentiment:")
        st.write(f"**{sentiment}**")

        st.subheader("🧠 Topic Detected:")
        st.write(f"**{topic_name}**")
        st.write("Keywords include: " + ", ".join([word for word, _ in topic_words]))
    else:
        st.warning("Please enter a review first.")
