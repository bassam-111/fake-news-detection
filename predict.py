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
        print("For a quick test without the full dataset, run:")
        print("  python model/train_model.py --sample")
        sys.exit(1)

    model = joblib.load(model_path)
    print(f"Model loaded: {model_name}\n")
    return model


def predict_news(model, text):
    """Predict if news is FAKE or REAL.

    Returns (label, score): label is "FAKE" or "REAL", score is the raw
    decision-function margin (None when the model has no decision
    function). The score is a margin, not a probability: its sign points
    at the predicted class and a larger absolute value means a more
    confident margin.
    """
    prediction = model.predict([text])[0]
    label = "FAKE" if prediction == 1 else "REAL"

    # Get decision-function margin if available
    score = None
    steps = getattr(model, 'named_steps', None)
    clf = steps.get('clf') if isinstance(steps, dict) else None
    if clf is not None and hasattr(clf, 'decision_function'):
        score = float(model.decision_function([text])[0])

    return label, score


def _show_prediction(label, score):
    print(f"\n✓ Prediction: {label}")
    if score is not None:
        print(f"  Decision score: {score:.2f} (raw model margin, not a probability)")


def interactive_mode(model):
    """Run in interactive mode for multiple predictions."""
    print("=" * 60)
    print("FAKE NEWS DETECTION - INTERACTIVE MODE")
    print("=" * 60)
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

        label, score = predict_news(model, text)
        _show_prediction(label, score)
        print()


def batch_mode(model, texts):
    """Predict for multiple texts."""
    print("=" * 60)
    print("BATCH PREDICTION RESULTS")
    print("=" * 60 + "\n")

    for i, text in enumerate(texts, 1):
        label, score = predict_news(model, text)

        preview = text[:50] + "..." if len(text) > 50 else text
        print(f"{i}. Text: {preview}")
        print(f"   Prediction: {label}")
        if score is not None:
            print(f"   Decision score: {score:.2f}")
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
        label, score = predict_news(model, text)
        print(f"Text: {text}\n")
        _show_prediction(label, score)
    else:
        # Run in interactive mode
        interactive_mode(model)


if __name__ == "__main__":
    main()
