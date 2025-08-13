from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def extract_features(train_texts, test_texts, max_features=5000):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # unigrams + bigrams
        stop_words='english'
    )

    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_test, vectorizer

def save_vectorizer(vectorizer, file_path):
    with open(file_path, 'wb') as  f:
        pickle.dump(vectorizer, f)

def load_vectorizer(file_path):
    """
    Load a TF-IDF vectorizer from file.
    """
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    from preprocessing import load_dataset, preprocess_data
    from sklearn.model_selection import train_test_split

    # Load and preprocess dataset
    dataset_path = "../data/training.1600000.processed.noemoticon.csv"
    df = load_dataset(dataset_path)
    df = preprocess_data(df)

    # Split data
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['clean_text'], df['target'], test_size=0.2, random_state=42
    )

    # Extract features
    X_train, X_test, vectorizer = extract_features(X_train_text, X_test_text)

    print("Shape of training data:", X_train.shape)
    print("Shape of testing data:", X_test.shape)

    import os
    os.makedirs("artifacts", exist_ok=True)  # creates folder if not exists
    
    save_vectorizer(vectorizer, "artifacts/tfidf_vectorizer.pkl")
