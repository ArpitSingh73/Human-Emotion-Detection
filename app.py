# ========================import packages=========================================================
import streamlit as st
import numpy as np
import re
from nltk.stem import PorterStemmer
import pickle
import nltk

# Download NLTK stopwords
nltk.download("stopwords")
stopwords = set(nltk.corpus.stopwords.words("english"))

# ========================loading the saved files==================================================
lg = pickle.load(open("logistic_regresion.pkl", "rb"))
tfidf_vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
lb = pickle.load(open("label_encoder.pkl", "rb"))


# =========================repeating the same functions==========================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(lg.predict(input_vectorized))

    return predicted_emotion, label


# ==================================creating app====================================
# App
st.title("Human Emotions Detection App")
st.text("")
st.text("")
st.text("")
# st.write("=================================================")
# st.write("['Joy,'Fear','Anger','Love','Sadness','Surprise']")
# st.write("=================================================")


var = st.sidebar.radio("Navigation", ["Home", "About"])
if var == "Home":
    # taking input from user
    user_input = st.text_input("Enter your text here:")
    if st.button("Predict") and user_input:
        predicted_emotion, label = predict_emotion(user_input)
        st.text("")
        st.text("")
        st.write(
                    "Emotion of input sentence appears to be :",
                    predicted_emotion.upper(),
                )
    # else:
    #    st.warning("Please enter something...")    
elif var == "About":
    st.subheader(
        "Hi there, this is a human emotion detection app. This app can understand and classify six basic human emotions viz. `Anger`, `Joy`, `Surprise`, `Sadness`, `Fear`, `Love`  with the help of `Logistic Regression`."
    )
# st.write("Probability:", label)
