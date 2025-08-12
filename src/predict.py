import os
import pickle
import pandas as pd

class PredictPipeline:
    def __init__(self):
        model_path = os.path.join("artifacts", "model.pkl")
        vectorizer_path = os.path.join("artifacts", "vectorizer.pkl")

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

    def predict(self, text):
        data = pd.Series([text])
        transformed_data = self.vectorizer.transform(data)
        prediction = self.model.predict(transformed_data)
        return prediction[0]
