# 01_data_preprocessing.ipynb
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load datasets
fake_df = pd.read_csv("Fake.csv")
real_df = pd.read_csv("True.csv")
fake_df["label"] = 0  # Fake
real_df["label"] = 1  # Real
df = pd.concat([fake_df, real_df], axis=0).sample(frac=1).reset_index(drop=True)

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

df['clean_text'] = df['text'].apply(preprocess)

# Save for model training
df.to_csv("clean_news.csv", index=False)
df.head()
