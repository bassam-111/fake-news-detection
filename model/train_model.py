import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import joblib
import os
import nltk
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Set paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, 'dataset', 'news.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'model')


def load_data():
    """Load the dataset from CSV file."""
    print("Loading dataset...")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        print("Please download the Fake and Real News Dataset from Kaggle and place it in the dataset folder.")
        return None
    
    df = pd.read_csv(DATASET_PATH)
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
    
    # Handle missing values
    df = df.dropna(subset=['title', 'text', 'label'])
    
    # Combine title and text for better context
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    
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


def build_pipeline(model_type='passive_aggressive'):
    """
    Build ML pipeline with TF-IDF vectorizer and classifier.
    
    Args:
        model_type: 'passive_aggressive', 'logistic_regression', or 'naive_bayes'
    """
    print(f"\nBuilding pipeline with {model_type} classifier...")
    
    if model_type == 'passive_aggressive':
        clf = PassiveAggressiveClassifier(max_iter=50, random_state=42, n_jobs=-1)
    elif model_type == 'logistic_regression':
        clf = LogisticRegression(max_iter=200, random_state=42, n_jobs=-1)
    elif model_type == 'naive_bayes':
        clf = MultinomialNB()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.7, min_df=5, max_features=5000)),
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
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    y_pred = pipeline.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"\nAccuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives  (REAL):  {cm[0,0]}")
    print(f"  False Positives (REAL):  {cm[0,1]}")
    print(f"  False Negatives (FAKE):  {cm[1,0]}")
    print(f"  True Positives  (FAKE):  {cm[1,1]}")
    
    print("="*50)
    
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
    """Make a prediction on new text."""
    prediction = pipeline.predict([text])[0]
    probability = pipeline.decision_function([text])[0] if hasattr(pipeline.named_steps['clf'], 'decision_function') else None
    
    label = "FAKE" if prediction == 1 else "REAL"
    return label, prediction


def main():
    """Main training pipeline."""
    # Load data
    df = load_data()
    if df is None:
        return
    
    # Preprocess
    df = preprocess_data(df)
    
    X = df['content'].values
    y = df['label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    models = {
        'passive_aggressive': build_pipeline('passive_aggressive'),
        'logistic_regression': build_pipeline('logistic_regression'),
        'naive_bayes': build_pipeline('naive_bayes')
    }
    
    results = {}
    for model_name, pipeline in models.items():
        print(f"\n\n{'='*50}")
        print(f"Training {model_name.upper()}")
        print('='*50)
        
        # Train
        trained_pipeline = train_model(pipeline, X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(trained_pipeline, X_test, y_test)
        results[model_name] = metrics
        
        # Save
        save_model(trained_pipeline, f'{model_name}_model.pkl')
    
    # Print comparison
    print(f"\n\n{'='*50}")
    print("MODEL COMPARISON")
    print('='*50)
    for model_name, metrics in results.items():
        print(f"{model_name:25} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1']:.4f}")
    
    # Recommend best model
    best_model = max(results.items(), key=lambda x: x[1]['f1'])
    print(f"\nBest model (F1-score): {best_model[0]} with F1={best_model[1]['f1']:.4f}")


if __name__ == "__main__":
    main()
