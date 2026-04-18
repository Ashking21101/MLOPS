import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# UPDATED: This now points to the project root (DVC/) instead of the src/ folder
BASE_DIR = Path(__file__).resolve().parent.parent

# fetch the data from data/raw (Relative to project root)
train_data = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'train.csv')
test_data = pd.read_csv(BASE_DIR / 'data' / 'raw' / 'test.csv')

# transform the data
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(y) for y in text]
    return " ".join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):
    text = text.split()
    text = [y.lower() for y in text]
    return " ".join(text)

def removing_punctuations(text):
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    # Ensure we use the correct column name 'content' as per your logic
    df.content = df.content.apply(lambda content: lower_case(content))
    df.content = df.content.apply(lambda content: remove_stop_words(content))
    df.content = df.content.apply(lambda content: removing_numbers(content))
    df.content = df.content.apply(lambda content: removing_punctuations(content))
    df.content = df.content.apply(lambda content: removing_urls(content))
    df.content = df.content.apply(lambda content: lemmatization(content))
    return df

# Process the data
train_processed_data = normalize_text(train_data)
test_processed_data = normalize_text(test_data)

# Store the data inside data/processed (Relative to project root)
data_path = BASE_DIR / "data" / "processed"

os.makedirs(data_path, exist_ok=True)

train_processed_data.to_csv(data_path / "train_processed.csv", index=False)
test_processed_data.to_csv(data_path / "test_processed.csv", index=False)

print(f"Processed data saved to: {data_path}")