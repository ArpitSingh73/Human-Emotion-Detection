
# ml packages
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import pickle
import nltk
import re
from nltk.stem import PorterStemmer

import seaborn as sns
import matplotlib.pyplot as plt

from wordcloud import WordCloud


train_data = pd.read_csv(
    "data.txt", header=None, sep=";", names=["Comment", "Emotion"], encoding="utf-8"
)
# get all words length in comment
train_data["length"] = [len(x) for x in train_data["Comment"]]


# train_data.shape

# train_data.isnull().sum()

# train_data.duplicated().sum()

# EDA
sns.countplot(x=train_data["Emotion"])
plt.show()


# data distribution
df2 = train_data.copy()
df2["length"] = [len(x) for x in df2["Comment"]]

# Convert the 'length' column to a numpy array
length_values = df2["length"].values

# Use sns.histplot instead of sns.kdeplot for simplicity
sns.histplot(data=df2, x="length", hue="Emotion", multiple="stack")

plt.show()


# Words cloud for each emotions
def words_cloud(wordcloud, df):
    plt.figure(figsize=(10, 10))
    plt.title(df + " Word Cloud", size=16)
    plt.imshow(wordcloud)
    # No axis details
    plt.axis("off")


emotions_list = train_data["Emotion"].unique()
for emotion in emotions_list:
    text = " ".join(
        [
            sentence
            for sentence in train_data.loc[train_data["Emotion"] == emotion, "Comment"]
        ]
    )
    wordcloud = WordCloud(width=600, height=600).generate(text)
    words_cloud(wordcloud, emotion)


lb = LabelEncoder()
train_data["Emotion"] = lb.fit_transform(train_data["Emotion"])


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


df = (
    train_data.copy()
)  # copy df from train_data because we will use this for deep learing next


# Data cleaning and preprocessing
# Download NLTK stopwords
nltk.download("stopwords")
stopwords = set(nltk.corpus.stopwords.words("english"))


def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)


df["cleaned_comment"] = df["Comment"].apply(clean_text)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["cleaned_comment"], df["Emotion"], test_size=0.2, random_state=42
)
# Vectorization using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# Multi-class classification using different algorithms
classifiers = {
    "Multinomial Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
}

for name, clf in classifiers.items():
    print(f"\n===== {name} =====")
    clf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = clf.predict(X_test_tfidf)
    accuracy_tfidf = accuracy_score(y_test, y_pred_tfidf)
    print(f"\nAccuracy using TF-IDF: {accuracy_tfidf}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_tfidf))


# selecting model
lg = LogisticRegression()
lg.fit(X_train_tfidf, y_train)
lg_y_pred = lg.predict(X_test_tfidf)


def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])

    # Predict emotion
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    label = np.max(lg.predict(input_vectorized))

    return predicted_emotion, label


# Example usage
sentences = [
    "i didnt feel humiliated",
    "i feel strong and good overall",
    "im grabbing a minute to post i feel greedy wrong",
    "He was speechles when he found out he was accepted to this new job",
    "This is outrageous, how can you talk like that?",
    "I feel like im all alone in this world",
    "He is really sweet and caring",
    "You made me very crazy",
    "i am ever feeling nostalgic about the fireplace i will know that it is still on the property",
    "i am feeling grouchy",
    "He hates you",
]
for sentence in sentences:
    print(sentence)
    pred_emotion, label = predict_emotion(sentence)
    print("Prediction :", pred_emotion)
    print("Label :", label)
    print("================================================================")


# save files
import pickle

pickle.dump(lg, open("logistic_regresion.pkl", "wb"))
pickle.dump(lb, open("label_encoder.pkl", "wb"))
pickle.dump(tfidf_vectorizer, open("tfidf_vectorizer.pkl", "wb"))
