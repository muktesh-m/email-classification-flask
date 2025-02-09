import numpy as np
import pandas as pd
import re
import nltk
import string
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle

# --- NLTK Data Configuration ---
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')  # Use a local directory
if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Function to download NLTK data (with error handling)
def download_nltk_data(resource):
    try:
        nltk.data.find(resource)
        print(f"{resource} already downloaded.")
    except LookupError:
        print(f"Downloading {resource}...")
        nltk.download(resource, download_dir=nltk_data_path)

# Download necessary NLTK data
download_nltk_data('punkt')
download_nltk_data('stopwords')

# Load stopwords and stemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stemmer = nltk.stem.PorterStemmer()

# --- Data Loading ---
try:
    data = pd.read_csv('emails.csv')
    
    # Ensure required columns exist
    if 'text' not in data.columns or 'spam' not in data.columns:
        raise KeyError("Error: 'text' or 'spam' column missing in CSV.")
    
    # Replace NaN values with empty strings
    data['text'] = data['text'].fillna('')
except FileNotFoundError:
    print("Error: emails.csv not found. Ensure it's in the same directory as the script.")
    exit()
except Exception as e:
    print(f"Error loading emails.csv: {e}")
    exit()

# --- Text Preprocessing ---
def preprocess_text(text):
    try:
        text = str(text).lower()  # Convert to lowercase
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation

        tokens = nltk.word_tokenize(text)
        tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
        tokens = [stemmer.stem(word) for word in tokens]  # Stemming

        return " ".join(tokens) if tokens else "empty_text"  # Avoid empty strings
    except Exception as e:
        print(f"Error during text preprocessing: {e}")
        return "error_text"

# Extract Subject from text column
def extract_subject(text):
    try:
        match = re.search(r"(?i)^Subject:\s*(.*)", text)
        return match.group(1).split('\n')[0].strip() if match else ""
    except Exception as e:
        print(f"Error extracting subject: {e}")
        return ""

# Extract From (Sender) from text column
def extract_sender(text):
    try:
        match = re.search(r"[\w\.-]+@[\w\.-]+\.[\w]+", text)
        return match.group(0) if match else ""
    except Exception as e:
        print(f"Error extracting sender: {e}")
        return ""

# Apply preprocessing and feature extraction
try:
    data['Processed_Body'] = data['text'].apply(preprocess_text)
    data['Subject'] = data['text'].apply(extract_subject).fillna('')
    data['Subject_Length'] = data['Subject'].apply(lambda x: len(str(x)))
    data['From'] = data['text'].apply(extract_sender).fillna('')
    data['Sender_Importance'] = data['From'].apply(lambda x: 1 if 'admin' in str(x).lower() or 'support' in str(x).lower() else 0)
except Exception as e:
    print(f"Error during data processing: {e}")
    exit()

# Ensure some valid text exists before vectorization
if data['Processed_Body'].str.strip().eq("empty_text").all():
    print("Error: No meaningful content remains after preprocessing. Check data format.")
    exit()

# --- Feature Extraction using TF-IDF ---
try:
    vectorizer = TfidfVectorizer(max_features=3000)
    X = vectorizer.fit_transform(data['Processed_Body']).toarray()
except Exception as e:
    print(f"Error during TF-IDF vectorization: {e}")
    exit()

# --- Combine Features ---
try:
    X_combined = np.hstack((X, data[['Subject_Length', 'Sender_Importance']].values))
except Exception as e:
    print(f"Error combining features: {e}")
    exit()

# --- Model Training ---
try:
    y = data['spam']
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.25, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error during model training: {e}")
    exit()

# --- Save Model and Vectorizer ---
try:
    model_dir = os.path.abspath('model')
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'spam_model.pkl'), 'wb') as file:
        pickle.dump((model, vectorizer), file)
    
    print("Model trained and saved successfully!")
except Exception as e:
    print(f"Error saving model: {e}")
    exit()
