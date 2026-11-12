# Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An NLP-based machine learning system to automatically detect fake news articles using TF-IDF feature extraction and multiple classification algorithms.

## 🚀 Quick Start

**Try it in under a minute, no download needed** — the repo includes a
40-row sample dataset to prove the pipeline runs end to end:

```bash
# 1. Clone repository
git clone https://github.com/bassam-111/fake-news-detection.git
cd fake-news-detection

# 2. Install dependencies (Python 3.10+)
pip install -r requirements.txt

# 3. Train on the bundled sample dataset (seconds)
python model/train_model.py --sample

# 4. Run the tests
pip install -r requirements-dev.txt
python -m pytest tests/ -v

# 5. Run web app
python app.py
# Visit: http://localhost:5000
```

Sample metrics are **not** representative of real performance — the sample
only exists to verify the pipeline. For real training, download the Kaggle
dataset (see [Dataset & Results](#-dataset--results)) and run
`python model/train_model.py` (full training takes ~15–25 minutes).

## 📋 Overview

This project implements a complete pipeline for fake news detection:
- **Text Preprocessing**: Lowercasing and English stopword removal (built into the TF-IDF vectorizer), title + text combined into one field
- **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency)
- **Machine Learning Models**: PassiveAggressiveClassifier, LogisticRegression, Naive Bayes
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Web Interface**: Flask-based UI for real-time predictions

## 🎯 Problem Statement

Fake news spreads rapidly on social media and misleads the public. This system automatically classifies news articles as **FAKE** or **REAL** using natural language processing and machine learning techniques.

## 🧰 Tech Stack

- **Python 3.10+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Scikit-learn** - ML algorithms and utilities
- **Flask** - Web framework
- **Joblib** - Model serialization

## 📂 Folder Structure

```
fake-news-detection/
├── dataset/
│   ├── README.md                # Dataset instructions
│   ├── sample_news.csv          # 40-row sample dataset (bundled)
│   └── news.csv                 # Full dataset file (download required)
├── model/
│   ├── train_model.py           # Training script
│   └── *.pkl                    # Trained models (after training)
├── templates/
│   └── index.html               # Web UI
├── tests/
│   ├── test_app.py              # Flask API tests
│   └── test_train.py            # Pipeline, sample data and predictor tests
├── app.py                       # Flask application
├── predict.py                   # Standalone prediction script
├── requirements.txt             # Python dependencies (pinned)
├── requirements-dev.txt         # Test dependencies
└── README.md                    # This file
```

## 📊 Dataset & Results

> **Note:** The dataset files and trained model artifacts are not committed in this repository, so the accuracy metrics below are reported from the original training run and are not independently verifiable from this repo alone.

**Source**: [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) (Kaggle)

**Dataset Size**: 44,898 articles
- Fake: 23,481
- Real: 21,417

### ✅ Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **Passive Aggressive** 🏆 | **99.32%** | 99.36% | 99.34% | 99.35% |
| Logistic Regression | 98.72% | 99.18% | 98.36% | 98.77% |
| Naive Bayes | 93.02% | 92.84% | 93.89% | 93.36% |

**Best Model (reported)**: PassiveAggressiveClassifier with **99.32% accuracy** in the original training run.

## 🔍 Data Preprocessing

1. **Remove Missing Values**: Eliminate rows with null titles, text, or labels
2. **Combine Features**: Merge title and text for better context
3. **Label Encoding**: FAKE → 1, REAL → 0
4. **Text Cleaning** (inside the TF-IDF vectorizer):
   - Lowercase conversion
   - English stopword removal

## 🤖 Model Architecture

### TF-IDF + Classification Pipeline

```python
Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=5,
        max_features=5000
    )),
    ('clf', ClassifierAlgorithm())
])
```

### Classifiers Tested

1. **PassiveAggressiveClassifier** ⭐ (Recommended)
   - Best for online learning
   - Fast and efficient

2. **Logistic Regression**
   - Probabilistic classifier
   - Good interpretability

3. **Naive Bayes**
   - Fast and lightweight

## 🧪 Training & Evaluation

### Running the Training Pipeline

```bash
# Install dependencies
pip install -r requirements.txt

# Download dataset from Kaggle, merge Fake.csv + True.csv into
# dataset/news.csv (see dataset/README.md), then train all models
python model/train_model.py
```

### Results Achieved

The models were reportedly trained on 44,898 articles from Kaggle during the original run:
- **Training set**: 35,918 articles (80%)
- **Testing set**: 8,980 articles (20%)
- **Training time**: ~20 minutes on Core i7 8th gen, 16GB RAM

**Confusion Matrix (Best Model - Passive Aggressive)**:
```
                 Predicted
              REAL      FAKE
Actual REAL    4254      30
       FAKE      31    4665
```

**Key Metrics** (reported from the original run, see the note above):
- ✅ Only 30 false positives (real news flagged as fake)
- ✅ Only 31 false negatives (fake news passed as real)

## 🌐 Web Interface (Optional)

### Running the Flask App

```bash
# After training the model
python app.py
```

Then navigate to `http://localhost:5000` in your browser.

### Features

- ✅ Paste or type news content
- ✅ Real-time prediction
- ✅ Decision score for each prediction (raw model margin, not a probability)
- ✅ Responsive design
- ✅ Mobile-friendly UI

### API Endpoints

`score` and `confidence` are the raw decision-function margin (and its
absolute value). They are **not** probabilities: the sign points at the
predicted class and a larger absolute value means a more confident margin.

**POST `/predict`**
```json
Request:
{
    "text": "News article content here..."
}

Response:
{
    "text": "News article content...",
    "prediction": "FAKE",
    "score": 2.45,
    "confidence": 2.45
}
```

**GET `/health`**
```json
Response:
{
    "status": "ready",
    "model_loaded": true
}
```

## 🧪 Demo (verified run)

Trained on the bundled 40-row sample and queried live — sample numbers
only, not real performance:

```bash
$ python model/train_model.py --sample
Sample mode: training on the 40-row bundled sample dataset.
Training set size: 32
Testing set size: 8
passive_aggressive        | Accuracy: 0.8750 | F1: 0.8889
logistic_regression       | Accuracy: 0.8750 | F1: 0.8889
naive_bayes               | Accuracy: 1.0000 | F1: 1.0000

$ curl -s http://localhost:5000/health
{"model_loaded":true,"status":"ready"}

$ curl -s -X POST http://localhost:5000/predict \
    -H 'Content-Type: application/json' \
    -d '{"text": "Scientists confirm drinking cola cures cancer in 48 hours"}'
{"confidence":0.47321269849638203,"prediction":"FAKE","score":0.47321269849638203,"text":"Scientists confirm drinking cola cures cancer in 48 hours"}

$ curl -s -X POST http://localhost:5000/predict \
    -H 'Content-Type: application/json' \
    -d '{"text": "City council approves budget for road repairs"}'
{"confidence":0.7482182318521048,"prediction":"REAL","score":-0.7482182318521048,"text":"City council approves budget for road repairs"}

$ python predict.py "Secret government conspiracy revealed by leaked documents"
✓ Prediction: FAKE
  Decision score: 0.63 (raw model margin, not a probability)
```

No UI screenshot is included: this environment has no usable headless
browser (the Playwright Chromium download failed and the system Chrome
is blocked by a domain policy), so the demo above is a real terminal
transcript instead.

## 📝 Key Features

✔ **Structured Code**
- Modular structure
- Error handling
- Logging capability

✔ **Research-Oriented**
- Multiple model comparison
- Detailed evaluation metrics
- Extensible architecture

✔ **Scalable Pipeline**
- Scikit-learn Pipeline for reproducibility
- Easy model serialization
- Cross-validation ready

✔ **Interview/Portfolio Friendly**
- Clean code structure
- Comprehensive documentation
- Real-world problem
- Shows NLP + ML skills

## 🚀 How to Use

### 1. Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd fake-news-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Go to [Kaggle Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
2. Download the dataset
3. Extract `Fake.csv` and `True.csv` into the `dataset/` folder
4. Merge them into `dataset/news.csv` (snippet in `dataset/README.md`)

Or skip the download and try the bundled sample first:
`python model/train_model.py --sample`

### 3. Train Model

```bash
python model/train_model.py
```

Output will show:
- Data preprocessing stats
- Model training progress
- Evaluation metrics for all 3 models
- Model comparison
- Saved model files in `model/` directory

### 4. Use the Model

**Option A: Programmatically**
```python
import joblib

model = joblib.load('model/passive_aggressive_model.pkl')
prediction = model.predict(['Your news text here'])[0]
label = "FAKE" if prediction == 1 else "REAL"
print(f"Prediction: {label}")
```

**Option B: Web Interface**
```bash
python app.py
# Visit http://localhost:5000
```

## 🔥 Easy Extensions

1. **BERT-based Classifier**
   - Replace TF-IDF with BERT embeddings
   - Use Hugging Face transformers
   - Improve accuracy to 95%+

2. **Browser Extension**
   - Highlight suspicious news on social media
   - Real-time fact-checking

3. **News API Integration**
   - Fetch live news articles
   - Batch processing
   - Daily reports

4. **Explainability**
   - LIME (Local Interpretable Model-agnostic Explanations)
   - Show which words contributed to prediction
   - Feature importance visualization

5. **Database Integration**
   - Store predictions
   - Track accuracy over time
   - Build user feedback loop

## 📚 Learning Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [NLP with NLTK](https://www.nltk.org/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 📊 Project Metrics

- **Lines of Code**: ~500 (app + training + predictor), ~700 including tests
- **Training Time**: ~15–25 minutes on the full 44,898-article dataset; seconds on the bundled sample
- **Model Size**: ~50–100 MB (full dataset); sample models ~20 KB
- **Prediction Time**: <100ms per article

## 🎓 Why This Project is Strong

✅ **Technical Skills Demonstrated**
- Natural Language Processing
- Machine Learning Pipeline Design
- Model Evaluation & Comparison
- Web Development (Flask)
- Data Preprocessing

✅ **Real-World Application**
- Addresses actual problem (misinformation)
- Clean, structured code
- End-to-end pipeline: data → training → API → UI

✅ **Portfolio Value**
- Impressive for internship applications
- Strong FYP (Final Year Project) foundation
- Interview-ready implementation

✅ **Research Potential**
- Compare multiple algorithms
- A/B testing framework
- Easy to add state-of-the-art models (BERT, GPT)

## 💡 Interview Talking Points

1. **Why TF-IDF?**
   - Captures term importance
   - Efficient and interpretable
   - Works well for text classification

2. **Model Selection**
   - Compared 3 different classifiers
   - Used F1-score for imbalanced data
   - Trade-off between accuracy and speed

3. **Pipeline Design**
   - Prevents data leakage
   - Reproducible results
   - Easy to deploy

4. **Future Improvements**
   - BERT embeddings for contextual understanding
   - Ensemble methods
   - Active learning for annotation efficiency

## 📄 License

MIT License - Feel free to use this project for learning and research.

## 👨‍💻 Author

Created for NLP + ML learning and portfolio building.

---

**Happy Learning! 🚀**

For questions or improvements, feel free to create an issue or pull request.
