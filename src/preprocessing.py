import re
import pandas as pd

def load_dataset(path):
    cols = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv(path,encoding='latin-1',names = cols)
    return df

def clean_tweet(text):
    #remove urls, mentions, hashtags, punctuation
    #convert text to lowercase

    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # URLs
    text = re.sub(r'@\w+', '', text)  # mentions
    text = re.sub(r'#', '', text)  # hashtags symbol
    text = re.sub(r'[^A-Za-z\s]', '', text)  # punctuation/numbers
    text = text.lower()  # lowercase
    text = re.sub(r'\s+', ' ', text).strip()  # extra spaces
    return text

def preprocess_data(df):
    df['target'] = df['target'].replace({0:'Negative',4:'Positive'})
    df['clean_text'] = df['text'].apply(clean_tweet)
    return df

if __name__ == "__main__":
    dataset_path = "data/training.1600000.processed.noemoticon.csv"
    df = load_dataset(dataset_path)
    df = preprocess_data(df)
    print(df.head())

