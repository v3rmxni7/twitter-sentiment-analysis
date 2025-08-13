import pickle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import load_dataset, preprocess_data

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
TEST_DATA_PATH = "data/test.csv"

def evaluate_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECTORIZER_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    df_test = load_dataset(TEST_DATA_PATH)
    df_test = preprocess_data(df_test)
    X_test = vectorizer.transform(df_test["clean_text"])
    y_test = df_test["target"]
    y_pred = model.predict(X_test)
    print("="*50)
    print("✅ Model Evaluation Results")
    print("="*50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("="*50)

if __name__ == "__main__":
    evaluate_model()
