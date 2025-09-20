# app.py
import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure stopwords are downloaded
nltk.download('stopwords', quiet=True)

# Preprocessing (same as notebook)
port_stem = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text: str) -> str:
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower().split()
    final_words = [port_stem.stem(word) for word in text if word not in stop_words]
    return ' '.join(final_words)

# Load models (cached)
@st.cache_resource
def load_model_and_vectorizer():
    model_path = 'model.pkl'
    vec_path = 'vectorizer.pkl'
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model_and_vectorizer()

st.title("📰 Fake News Detection")
st.write("Enter a headline or article text. The app preprocesses the text the same way as the notebook (stemming + stopwords) before prediction.")

if model is None or vectorizer is None:
    st.error("Model or vectorizer not found. Make sure 'model.pkl' and 'vectorizer.pkl' are in the project folder.")
    st.stop()

user_input = st.text_area("News text", height=200, placeholder="Paste a news headline or article here...")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        processed = preprocess(user_input)
        vect = vectorizer.transform([processed])

        pred = model.predict(vect)[0]
        # confidence (for models that support predict_proba)
        prob = None
        try:
            prob = model.predict_proba(vect)[0][int(pred)]
        except Exception:
            prob = None

        if int(pred) == 0:
            msg = "✅ The News is Real"
            if prob is not None:
                st.success(f"{msg}  — Confidence: {prob:.2%}")
            else:
                st.success(msg)
        else:
            msg = "❌ The News is Fake"
            if prob is not None:
                st.error(f"{msg}  — Confidence: {prob:.2%}")
            else:
                st.error(msg)
