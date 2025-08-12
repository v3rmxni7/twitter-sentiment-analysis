import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import load_dataset, preprocess_data
from feature_extraction import extract_features

def train_model(X, y):
    model = LogisticRegression(max_iter=1000,solver= 'liblinear')
    model.fit(X,y)
    return model

if __name__ == "__main__":
    dataset_path = "data/training.1600000.processed.noemoticon.csv"

    # Load & preprocess
    df = load_dataset(dataset_path)
    df = preprocess_data(df)

    # Extract features
    X, vectorizer = extract_features(df)

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['target'], test_size=0.2, random_state=42, stratify=df['target']
    )

    # Train model
    model = train_model(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Save model & vectorizer
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print(" Model and vectorizer saved in 'models/' folder")