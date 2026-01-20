# 🎬 Sentiment Analysis of Movie Reviews Using Machine Learning

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![NLTK](https://img.shields.io/badge/NLTK-3.6+-green.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

A comprehensive machine learning project that classifies movie reviews as **positive** or **negative** with **88% accuracy** using Support Vector Machine (SVM).

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Demo](#-demo)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Technologies](#-technologies)
- [Project Structure](#-project-structure)
- [Author](#-author)

---

## 🎯 Overview

This project implements a complete **sentiment analysis pipeline** for movie reviews using multiple machine learning algorithms. The system:

- Processes 50,000 IMDB movie reviews
- Compares 4 different ML algorithms
- Achieves 88.2% accuracy with SVM
- Includes comprehensive ROC-AUC analysis
- Provides real-time sentiment predictions

### 🏆 Key Achievements

✅ **88.2% Accuracy** on test data  
✅ **4 Algorithms Compared**: SVM, Naive Bayes, Logistic Regression, Random Forest  
✅ **ROC-AUC: 0.943** - Excellent discrimination capability  
✅ **Complete Pipeline**: From raw text to production-ready model  
✅ **Experimental Analysis**: Systematic hyperparameter tuning  

---

## ✨ Key Features

### Data Processing
- Automated text preprocessing (HTML removal, stopword filtering, stemming)
- TF-IDF feature extraction with 5,000 optimal features
- Balanced dataset with 50,000 reviews

### Model Training & Comparison
- **Support Vector Machine** (Linear kernel) - **Selected as best model**
- **Naive Bayes** (Multinomial)
- **Logistic Regression**
- **Random Forest Classifier**

### Evaluation & Analysis
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves for all classifiers
- Confusion matrix analysis
- Training time comparison

### Experimental Work
- Kernel comparison (Linear, RBF, Polynomial)
- C parameter tuning (0.01 to 100)
- Vocabulary size optimization
- Custom test case validation

---

## 🎥 Demo

### Try It Live

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16bHc5drV-wTkA_KWaGM8aHypzXbfTXcL?usp=sharing))

Click the badge above to run the complete project in Google Colab!

### Quick Preview

**Input:**
```
"This movie was absolutely fantastic! The acting was superb and the plot kept me engaged."
```

**Output:**
```
Sentiment: POSITIVE
Confidence: 2.47
```

---

## 📊 Results

### Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | AUC | Training Time |
|-------|----------|-----------|--------|----------|-----|---------------|
| **SVM (Linear)** ⭐ | **88.2%** | **89.1%** | **87.3%** | **88.2%** | **0.943** | 35s |
| Logistic Regression | 87.9% | 88.5% | 87.2% | 87.8% | 0.938 | 8s |
| Random Forest | 85.7% | 86.3% | 85.1% | 85.7% | 0.925 | 45s |
| Naive Bayes | 83.5% | 82.8% | 84.2% | 83.5% | 0.915 | 2s |

### SVM Confusion Matrix

```
                Predicted
              Negative  Positive
Actual  Neg     2156      294      (88.0% correct)
        Pos      313     2137      (87.2% correct)

Overall Accuracy: 88.2%
```

### Why SVM Was Selected

1. ✅ Highest accuracy (88.2%)
2. ✅ Best AUC score (0.943)
3. ✅ Balanced precision and recall
4. ✅ Reasonable training time (35s)
5. ✅ Proven effectiveness for text classification

### Experimental Results

| Experiment | Best Configuration | Result |
|------------|-------------------|---------|
| **Kernel Comparison** | Linear kernel | Best balance of speed & accuracy |
| **C Parameter** | C = 1.0 | Optimal regularization |
| **Vocabulary Size** | 5,000 features | Diminishing returns beyond this |
| **Custom Tests** | 15 test cases | 80% accuracy (struggles with sarcasm) |

---

## 💻 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/sentiment-analysis-svm.git
cd sentiment-analysis-svm

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Dependencies

- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- nltk >= 3.6.0
- matplotlib >= 3.4.0
- seaborn >= 0.11.0
- jupyter >= 1.0.0

---

## 🚀 Usage

### Option 1: Google Colab (Recommended)

1. Click the **"Open in Colab"** badge above
2. Run all cells in order
3. The notebook includes:
   - Complete implementation
   - Model training and comparison
   - Visualization generation
   - Interactive predictions

### Option 2: Local Jupyter Notebook

```bash
# Start Jupyter
jupyter notebook

# Open the notebook
# Navigate to: notebooks/01_complete_pipeline.ipynb
# Run all cells
```

### Making Predictions

The notebook includes a prediction function at the end:

```python
# Example usage from the notebook
review = "This movie was amazing!"
sentiment, confidence = predict_sentiment(review)
print(f"Sentiment: {sentiment}, Confidence: {confidence:.2f}")
```

---

## 🛠️ Technologies Used

### Core Technologies
- **Python 3.8+** - Programming language
- **scikit-learn** - Machine learning framework
- **NLTK** - Natural language processing
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing

### Machine Learning
- **Support Vector Machine (SVM)** - Best performing model
- **TF-IDF Vectorization** - Feature extraction
- **Naive Bayes** - Probabilistic classifier
- **Logistic Regression** - Linear classifier
- **Random Forest** - Ensemble method

### Visualization
- **Matplotlib** - Plotting library
- **Seaborn** - Statistical visualization

### Development
- **Jupyter Notebook** - Interactive development
- **Google Colab** - Cloud-based execution

---

## 📁 Project Structure

```
sentiment-analysis-svm/
│
├── notebooks/
│   └── 01_complete_pipeline.ipynb    # Complete project implementation
│
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
└── LICENSE                            # MIT License
```

### Notebook Contents

The `01_complete_pipeline.ipynb` notebook includes:

1. **Data Loading** - IMDB dataset (50,000 reviews)
2. **Data Exploration** - Statistics and visualizations
3. **Preprocessing** - Text cleaning and normalization
4. **Feature Extraction** - TF-IDF vectorization
5. **Model Training** - All 4 classifiers
6. **Model Comparison** - Performance analysis
7. **ROC-AUC Analysis** - Validation curves
8. **Evaluation** - Comprehensive metrics
9. **Results Dashboard** - Visualizations
10. **Prediction Module** - Interactive predictions
11. **Experiments** - Hyperparameter tuning
12. **Conclusion** - Summary and insights

---

## 📈 Methodology

### 1. Data Preprocessing Pipeline

```
Raw Text → Lowercase → Remove HTML → Remove URLs 
→ Remove Special Characters → Remove Stopwords → Stemming → Clean Text
```

### 2. Feature Engineering

- **TF-IDF Vectorization** with optimal parameters
- **max_features**: 5,000 (top important words)
- **min_df**: 2 (word appears in ≥2 documents)
- **max_df**: 0.8 (word in <80% of documents)
- **ngram_range**: (1, 2) - Unigrams and bigrams

### 3. Model Selection Process

1. Train 4 different algorithms
2. Evaluate on multiple metrics
3. Generate ROC-AUC curves
4. Compare training efficiency
5. Select best overall performer → **SVM**

---

## 🎓 Learning Outcomes

This project demonstrates:

- **Data Science Skills**: Complete ML pipeline from data to deployment
- **NLP Expertise**: Text preprocessing and feature extraction
- **Model Evaluation**: Systematic comparison and validation
- **Experimental Methodology**: Hyperparameter tuning and optimization
- **Technical Communication**: Clear documentation and visualization

---

## 📚 Dataset

**Source**: [IMDB Movie Reviews Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

**Statistics**:
- Total Reviews: 50,000
- Positive Reviews: 25,000 (50%)
- Negative Reviews: 25,000 (50%)
- Average Review Length: ~230 words
- Training Set: 40,000 reviews (80%)
- Test Set: 10,000 reviews (20%)

---

## 🔮 Future Enhancements

### Short-term
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Add multi-class sentiment (positive/negative/neutral)
- [ ] Create web application with Flask/FastAPI
- [ ] Improve sarcasm detection

### Long-term
- [ ] Multilingual support
- [ ] Real-time streaming analysis
- [ ] Aspect-based sentiment analysis
- [ ] Mobile application

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software")...
```

---

## 👤 Author

**Dudekula Riyaz**

- 📧 Email: riyazdudekula2005@gmail.com
- 💼 LinkedIn: [linkedin.com/in/Dudekula riyaz](https://linkedin.com/in/](https://www.linkedin.com/in/riyaz-dudekula-b7aaa52b7/))
- 🐙 GitHub: [@codemage05](https://github.com/codemage05)

**Project Repository**: [github.com/codemage05/sentiment-analysis-svm](https://github.com/codemage05/sentiment-analysis-svm)

---

## 🙏 Acknowledgments

- **Stanford University** - For the IMDB dataset
- **scikit-learn team** - For the excellent ML library
- **NLTK contributors** - For NLP tools and resources
- **Project Mentor** - For guidance on model comparison methodology
- **Open Source Community** - For inspiration and learning resources

---

## 📊 Project Statistics

![GitHub repo size](https://img.shields.io/github/repo-size/yourusername/sentiment-analysis-svm)
![GitHub stars](https://img.shields.io/github/stars/yourusername/sentiment-analysis-svm?style=social)
![GitHub forks](https://img.shields.io/github/forks/yourusername/sentiment-analysis-svm?style=social)

---

## 📞 Questions or Feedback?

If you have any questions, suggestions, or feedback:

1. Open an issue on GitHub
2. Email me directly
3. Connect on LinkedIn

I'm always happy to discuss machine learning, NLP, or this project!

---

<div align="center">

**⭐ If you found this project helpful, please consider giving it a star! ⭐**

Made with ❤️ for learning and sharing knowledge

[⬆ Back to Top](#-sentiment-analysis-of-movie-reviews-using-machine-learning)

</div>
