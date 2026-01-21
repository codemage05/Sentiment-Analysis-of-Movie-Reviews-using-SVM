import pandas as pd
import os

def load_data(filepath):
    """
    Loads the movie reviews dataset.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found at {filepath}")
        
    df = pd.read_csv(filepath)
    
    # Basic validation
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'sentiment' columns")
        
    return df