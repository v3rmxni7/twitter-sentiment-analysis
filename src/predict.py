import pickle
from preprocessing import clean_tweet
from feature_extraction import load_vectorizer

# File paths (use artifacts folder)
MODEL_PATH = "artifacts/sentiment_model.pkl"
VECTORIZER_PATH = "artifacts/tfidf_vectorizer.pkl"

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load vectorizer
vectorizer = load_vectorizer(VECTORIZER_PATH)

def predict_sentiment(text):
    # Clean and transform
    cleaned_text = clean_tweet(text)
    features = vectorizer.transform([cleaned_text])
    # Predict
    return model.predict(features)[0]

if __name__ == "__main__":
    sample_tweets = [
        "I love this product! It's amazing",
        "Worst experience ever, I want a refund.",
        "The weather is nice today."
    ]

    for tweet in sample_tweets:
        sentiment = predict_sentiment(tweet)
        print(f"Tweet: {tweet}")
        print(f"Predicted Sentiment: {sentiment}\n")
