import numpy as np
import pandas as pd
import re
import nltk
import string
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

app = Flask(__name__)

# Load the model and vectorizer
try:
    with open('./model/spam_model.pkl', 'rb') as file:
        model, vectorizer = pickle.load(file)
except FileNotFoundError:
    print("Error: Model file not found. Please train and save the model first.")
    exit()

# Preprocessing Function (same as before)
def preprocess_text(text):
    text = str(text).lower()  # Lowercase and handle NaN
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in nltk.corpus.stopwords.words('english')]  # Remove stopwords
    stemmer = nltk.stem.PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming
    return " ".join(tokens)

# Extract Subject from text column
def extract_subject(text):
    text = str(text)
    if text.startswith("Subject:"):
        return text[8:].split('\n')[0]  # Extract subject and remove newline
    else:
        return ""  # Return empty string if no subject

# Extract From (Sender) from text column
def extract_sender(text):
    text = str(text)
    # Basic extraction - improve as needed
    match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    if match:
        return match.group(0)
    else:
        return ""


def classify_email(email_subject, email_body, email_sender):
    processed_body = preprocess_text(email_body)
    vectorized_body = vectorizer.transform([processed_body]).toarray()
    subject_length = len(str(email_subject))
    sender_importance = 1 if 'admin' in str(email_sender).lower() or 'support' in str(email_sender).lower() else 0
    combined_features = np.hstack((vectorized_body, [[subject_length, sender_importance]]))
    prediction = model.predict(combined_features)
    return "Spam" if prediction[0] == 1 else "Not Spam"

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_subject = request.form['subject']
        email_body = request.form['body']
        email_sender = request.form['sender']
        prediction = classify_email(email_subject, email_body, email_sender)
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    # Ensure NLTK data is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

    app.run(debug=True)
