import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import load_dataset, preprocess_data
from feature_extraction import extract_features, save_vectorizer
from evaluate import evaluate_model
import os

def train_model(X, y):
    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X, y)
    return model

if __name__ == "__main__":
    dataset_path = "../data/training.1600000.processed.noemoticon.csv"

    # Load & preprocess
    df = load_dataset(dataset_path)
    df = preprocess_data(df)

    # Train/Test split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], df['target'], test_size=0.2, random_state=42
    )

    # Feature extraction
    X_train, X_test, vectorizer = extract_features(X_train_text, X_test_text, 5000)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save model & vectorizer inside artifacts/
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    save_vectorizer(vectorizer, "artifacts/tfidf_vectorizer.pkl")

    print("✅ Model and vectorizer saved in 'artifacts/' folder")
