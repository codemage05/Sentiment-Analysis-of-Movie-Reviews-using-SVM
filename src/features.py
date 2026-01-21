from sklearn.feature_extraction.text import TfidfVectorizer

def get_vectorizer(max_features=5000):
    """
    Returns the TF-IDF Vectorizer configuration from the notebook.
    """
    return TfidfVectorizer(
        max_features=max_features,  # From Cell 16
        ngram_range=(1, 2)          # Common best practice for sentiment
    )