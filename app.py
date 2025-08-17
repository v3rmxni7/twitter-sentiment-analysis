import streamlit as st
import pickle
import re

# Load model & vectorizer
model_path = "src/artifacts/sentiment_model.pkl"
vectorizer_path = "src/artifacts/tfidf_vectorizer.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Simple preprocessing (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)  # remove links
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    return text

# Streamlit UI
st.set_page_config(page_title="Twitter Sentiment Analysis", page_icon="🐦")
st.title("🐦 Twitter Sentiment Analysis")
st.write("Enter a tweet or text to analyze its sentiment:")

user_input = st.text_area("Your Tweet/Text")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        cleaned = clean_text(user_input)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        # Handle dataset labels (0 = Negative, 4 = Positive)
        if prediction == 0:
            sentiment = "😡 Negative"
        elif prediction == 4:
            sentiment = "😊 Positive"
        else:
            sentiment = f"🤔 Unknown (label={prediction})"

        st.subheader("Prediction:")
        st.success(sentiment)
