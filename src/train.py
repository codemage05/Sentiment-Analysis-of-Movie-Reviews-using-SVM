import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Import local modules
from src.data_loader import load_data
from src.preprocessing import preprocess_text
from src.features import get_vectorizer
from src.evaluate import evaluate_model

# Config
DATA_PATH = 'data/raw/movie_reviews.csv'
MODELS_DIR = 'models'

def main():
    # 1. Load Data
    print("Loading data...")
    df = load_data(DATA_PATH)
    
    # 2. Preprocess
    print("Preprocessing reviews (this takes time)...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    # Encode Labels (Cell 14 logic)
    df['label'] = df['sentiment'].map({'negative': 0, 'positive': 1})
    
    # 3. Split Data (Cell 15 logic)
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['label'], test_size=0.2, random_state=42
    )
    
    # 4. Vectorize (Cell 16 logic)
    print("Vectorizing...")
    vectorizer = get_vectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 5. Train SVM (Cell 19 & 21 logic)
    print("Training SVM Model...")
    svm = SVC(kernel='linear', C=1.0, probability=True)
    svm.fit(X_train_vec, y_train)
    
    # 6. Evaluate
    print("Evaluating...")
    y_pred = svm.predict(X_test_vec)
    evaluate_model(y_test, y_pred)
    
    # 7. Save Models
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    with open(f'{MODELS_DIR}/svm_model.pkl', 'wb') as f:
        pickle.dump(svm, f)
        
    with open(f'{MODELS_DIR}/tfidf_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print(f"Models saved to {MODELS_DIR}/")

if __name__ == "__main__":
    main()