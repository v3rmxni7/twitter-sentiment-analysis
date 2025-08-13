import os
import pickle
import pandas as pd

class PredictPipeline:
    def __init__(self):
        # Go one level up from src/ → project root
        base_dir = os.path.dirname(os.path.dirname(__file__))

        model_path = os.path.join(base_dir, "models", "sentiment_model.pkl")
        vectorizer_path = os.path.join(base_dir, "models", "tfidf_vectorizer.pkl")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"Vectorizer file not found at {vectorizer_path}")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text):
        data = pd.Series([text])
        transformed_data = self.vectorizer.transform(data)
        prediction = self.model.predict(transformed_data)
        return prediction[0]

if __name__ == "__main__":
    pipeline = PredictPipeline()
    text_input = input("Enter a tweet: ")
    sentiment = pipeline.predict(text_input)
    print(f"Predicted Sentiment: {sentiment}")
