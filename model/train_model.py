"""Train the fake news detection models.

Usage:
    # Full training on the Kaggle dataset (place news.csv in dataset/ first)
    python model/train_model.py

    # Quick smoke test on the small bundled sample dataset (40 rows).
    # Metrics from a sample run are NOT representative of real performance.
    python model/train_model.py --sample

    # Train on a dataset at a custom path
    python model/train_model.py --dataset path/to/news.csv

The dataset CSV needs columns: title, text, label (FAKE or REAL).
Trained pipelines are saved to model/ as <model>_model.pkl.
"""
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'news.csv')
SAMPLE_PATH = os.path.join(BASE_DIR, 'dataset', 'sample_news.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')


def load_data(path=DATASET_PATH):
    """Load the dataset from CSV file."""
    print("Loading dataset...")
    if not os.path.exists(path):
        print(f"Error: Dataset not found at {path}")
        print("Download the Fake and Real News Dataset from Kaggle, place")
        print("Fake.csv and True.csv in dataset/, and merge them into news.csv")
        print("(see dataset/README.md). Or run with --sample for a quick test.")
        return None

    df = pd.read_csv(path)
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df


def preprocess_data(df):
    """
    Preprocess the dataset.
    - Remove missing values
    - Combine title and text
    - Map labels
    """
    print("\nPreprocessing data...")

    # Handle missing values (copy so later assignments never warn)
    df = df.dropna(subset=['title', 'text', 'label']).copy()

    # Combine title and text for better context
    df['content'] = df['title'] + ' ' + df['text']

    # Map labels: FAKE -> 1, REAL -> 0
    label_mapping = {'FAKE': 1, 'REAL': 0, 1: 1, 0: 0}
    df['label'] = df['label'].map(label_mapping)

    # Remove rows with unmapped labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)

    print(f"Data shape after preprocessing: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    return df[['content', 'label']]


def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    print(f"\nSplitting data (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


def build_pipeline(model_type='passive_aggressive', min_df=5):
    """
    Build ML pipeline with TF-IDF vectorizer and classifier.

    Args:
        model_type: 'passive_aggressive', 'logistic_regression', or 'naive_bayes'
        min_df: ignore terms in fewer than this many documents. The default
            (5) suits the full 45k-article dataset; pass 1 for the tiny
            bundled sample dataset, where almost no term appears 5 times.
    """
    print(f"\nBuilding pipeline with {model_type} classifier...")

    if model_type == 'passive_aggressive':
        clf = PassiveAggressiveClassifier(max_iter=50, random_state=42)
    elif model_type == 'logistic_regression':
        clf = LogisticRegression(max_iter=200, random_state=42, n_jobs=-1)
    elif model_type == 'naive_bayes':
        clf = MultinomialNB()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7, min_df=min_df, max_features=5000)),
        ('clf', clf)
    ])

    return pipeline


def train_model(pipeline, X_train, y_train):
    """Train the model."""
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)
    print("Model training completed!")
    return pipeline


def evaluate_model(pipeline, X_test, y_test):
    """Evaluate model performance."""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)

    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

    print("\nConfusion Matrix:")
    print(f"  True Negatives  (REAL):  {cm[0, 0]}")
    print(f"  False Positives (REAL):  {cm[0, 1]}")
    print(f"  False Negatives (FAKE):  {cm[1, 0]}")
    print(f"  True Positives  (FAKE):  {cm[1, 1]}")

    print("=" * 50)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm
    }


def save_model(pipeline, model_name='fake_news_model.pkl'):
    """Save trained model to disk."""
    model_path = os.path.join(MODEL_DIR, model_name)
    joblib.dump(pipeline, model_path)
    print(f"\nModel saved to: {model_path}")
    return model_path


def predict(pipeline, text):
    """Make a prediction on new text.

    Returns (label, score): label is "FAKE" or "REAL", score is the raw
    decision-function margin (None when the model has no decision
    function). The score is a margin, not a probability.
    """
    prediction = pipeline.predict([text])[0]
    score = None
    steps = getattr(pipeline, 'named_steps', None)
    clf = steps.get('clf') if isinstance(steps, dict) else None
    if clf is not None and hasattr(clf, 'decision_function'):
        score = float(pipeline.decision_function([text])[0])

    label = "FAKE" if prediction == 1 else "REAL"
    return label, score


def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train fake news detection models.")
    parser.add_argument('--dataset', default=DATASET_PATH,
                        help='Path to news.csv (title, text, label columns)')
    parser.add_argument('--sample', action='store_true',
                        help='Train on the small bundled sample dataset '
                             '(smoke test only; metrics are not representative)')
    args = parser.parse_args()

    if args.sample:
        dataset_path = SAMPLE_PATH
        print("Sample mode: training on the 40-row bundled sample dataset.")
        print("Metrics from this run are NOT representative of real performance.")
    else:
        dataset_path = args.dataset

    # Load data
    df = load_data(dataset_path)
    if df is None:
        return

    # Preprocess
    df = preprocess_data(df)

    X = df['content'].values
    y = df['label'].values

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train models (small sample data needs min_df=1; the default of 5
    # only makes sense on the full 45k-article dataset)
    min_df = 1 if args.sample else 5
    models = {
        'passive_aggressive': build_pipeline('passive_aggressive', min_df=min_df),
        'logistic_regression': build_pipeline('logistic_regression', min_df=min_df),
        'naive_bayes': build_pipeline('naive_bayes', min_df=min_df)
    }

    results = {}
    for model_name, pipeline in models.items():
        print(f"\n\n{'=' * 50}")
        print(f"Training {model_name.upper()}")
        print('=' * 50)

        # Train
        trained_pipeline = train_model(pipeline, X_train, y_train)

        # Evaluate
        metrics = evaluate_model(trained_pipeline, X_test, y_test)
        results[model_name] = metrics

        # Save
        save_model(trained_pipeline, f'{model_name}_model.pkl')

    # Print comparison
    print(f"\n\n{'=' * 50}")
    print("MODEL COMPARISON")
    print('=' * 50)
    for model_name, metrics in results.items():
        print(f"{model_name:25} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")

    # Recommend best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest model (F1-score): {best_model[0]} with F1={best_model[1]['f1']:.4f}")


if __name__ == "__main__":
    main()
