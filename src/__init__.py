# Core libraries
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# NLTK for text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Scikit-learn for ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# Download NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)