# 03_web_app_streamlit.py
import streamlit as st
import pickle
import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load models
model = pickle.load(open("model_nb.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

# NLTK setup
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Streamlit UI
st.title("📰 Fake News Detection App")

user_input = st.text_area("Enter a news article or paragraph:")

if st.button("Predict"):
    processed = preprocess(user_input)
    vect = vectorizer.transform([processed])
    prediction = model.predict(vect)[0]
    if prediction == 0:
        st.error("🚨 The news appears to be **Fake**!")
    else:
        st.success("✅ The news appears to be **Real**.")
