import nltk
import os
import ssl

# --- Configuration ---
nltk_data_path = os.path.join(os.getcwd(), 'nltk_data')  # Local directory
os.makedirs(nltk_data_path, exist_ok=True)  # Ensure directory exists
os.environ['NLTK_DATA'] = nltk_data_path  # Set environment variable

# --- Fix SSL issues (if any) ---
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# --- Function to download with error handling ---
def download_nltk_resource(resource_name):
    try:
        nltk.data.find(resource_name)
        print(f"Resource '{resource_name}' already present.")
    except LookupError:
        print(f"Downloading resource '{resource_name}'...")
        try:
            nltk.download(resource_name.split('/')[0], download_dir=nltk_data_path)
            print(f"Resource '{resource_name}' downloaded successfully.")
        except Exception as e:
            print(f"Error downloading '{resource_name}': {e}")

# --- Download Resources ---
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('corpora/stopwords')
download_nltk_resource('tokenizers/punkt_tab')

# --- Print NLTK Path for Verification ---
print(f"NLTK data path: {nltk.data.path}")
