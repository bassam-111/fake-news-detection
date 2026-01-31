"""
Standalone prediction script for the fake news detection model.
Use this to make predictions without running the web server.
"""

import joblib
import os
import sys


def load_model(model_name='passive_aggressive_model.pkl'):
    """Load a trained model from disk."""
    model_path = os.path.join('model', model_name)
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run 'python model/train_model.py' first to train the model.")
        sys.exit(1)
    
    model = joblib.load(model_path)
    print(f"Model loaded: {model_name}\n")
    return model


def predict_news(model, text):
    """Predict if news is FAKE or REAL."""
    prediction = model.predict([text])[0]
    label = "FAKE" if prediction == 1 else "REAL"
    
    # Get confidence score if available
    confidence = None
    if hasattr(model.named_steps['clf'], 'decision_function'):
        confidence = float(model.decision_function([text])[0])
    
    return label, confidence


def interactive_mode(model):
    """Run in interactive mode for multiple predictions."""
    print("="*60)
    print("FAKE NEWS DETECTION - INTERACTIVE MODE")
    print("="*60)
    print("Enter news text to classify. Type 'quit' to exit.\n")
    
    while True:
        print("-" * 60)
        text = input("Enter news text (or 'quit' to exit):\n> ").strip()
        
        if text.lower() == 'quit':
            print("\nGoodbye!")
            break
        
        if not text:
            print("Error: Please enter some text.")
            continue
        
        label, confidence = predict_news(model, text)
        
        print(f"\n✓ Prediction: {label}")
        if confidence is not None:
            conf_percent = abs(confidence) * 100
            print(f"  Confidence: {conf_percent:.2f}%")
        print()


def batch_mode(model, texts):
    """Predict for multiple texts."""
    print("="*60)
    print("BATCH PREDICTION RESULTS")
    print("="*60 + "\n")
    
    for i, text in enumerate(texts, 1):
        label, confidence = predict_news(model, text)
        
        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"{i}. Text: {preview}")
        print(f"   Prediction: {label}")
        if confidence is not None:
            print(f"   Confidence: {abs(confidence)*100:.2f}%")
        print()


def main():
    """Main function."""
    # Load the best model (PassiveAggressiveClassifier)
    model = load_model('passive_aggressive_model.pkl')
    
    # Example texts for testing
    example_texts = [
        "President announces new policy to combat climate change with strict regulations",
        "Breaking: Secret government conspiracy revealed by leaked documents",
        "Scientists discover cure for common cold after decades of research"
    ]
    
    if len(sys.argv) > 1:
        # Command line argument provided
        text = ' '.join(sys.argv[1:])
        label, confidence = predict_news(model, text)
        print(f"Text: {text}\n")
        print(f"Prediction: {label}")
        if confidence is not None:
            print(f"Confidence: {abs(confidence)*100:.2f}%")
    else:
        # Run in interactive mode
        interactive_mode(model)


if __name__ == "__main__":
    main()
