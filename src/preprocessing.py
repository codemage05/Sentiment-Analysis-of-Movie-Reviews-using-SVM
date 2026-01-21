import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data (Safe check)
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

def preprocess_text(text):
    """
    Cleans text: Lowercase, removes HTML/URLs/Special chars, removes stopwords, and stems.
    """
    if not isinstance(text, str):
        return ""

    # Initialize tools
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    # 1. Convert to lowercase
    text = text.lower()

    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 3. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 4. Remove special characters (keep only alphabets)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 5. Tokenization & Stemming
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    return ' '.join(words)