import os
import joblib
from flask import Flask, render_template, request, jsonify
from pathlib import Path

app = Flask(__name__)

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
TEMPLATES_DIR = os.path.join(BASE_DIR, 'templates')

# Create templates directory if it doesn't exist
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Load model
MODEL_PATH = os.path.join(MODEL_DIR, 'passive_aggressive_model.pkl')
model = None
DEBUG = os.getenv('FLASK_DEBUG', 'false').lower() in {'1', 'true', 'yes', 'on'}

def load_model():
    """Load the trained model."""
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            print(f"✓ Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            print(f"⚠ Warning: Model not found at {MODEL_PATH}")
            print("Please run: python model/train_model.py")
            return False
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False


def _decision_score(text):
    """Raw decision-function margin for one text, or None.

    The value is a raw model margin, NOT a probability: its sign points at
    the predicted class and a larger absolute value means a more confident
    margin. Returns None when the loaded model has no decision function
    (for example, a plain classifier instead of a pipeline).
    """
    try:
        steps = getattr(model, 'named_steps', None)
        clf = steps.get('clf') if isinstance(steps, dict) else None
        if clf is not None and hasattr(clf, 'decision_function'):
            return float(model.decision_function([text])[0])
    except Exception:
        return None
    return None


# Load at import time too, so the app also works under `flask run` and
# WSGI servers (not only `python app.py`). Missing model is not fatal:
# /health reports not_ready and /predict explains how to train.
load_model()


@app.route('/')
def home():
    """Render the home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions."""
    if model is None:
        return jsonify({
            'error': 'Model not loaded. Please train the model first.',
            'instructions': 'Run: python model/train_model.py'
        }), 400
    
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing "text" field in request'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Text cannot be empty'}), 400
        
        # Limit text length
        if len(text) > 10000:
            return jsonify({'error': 'Text too long. Maximum 10,000 characters.'}), 400
        
        # Make prediction
        prediction = model.predict([text])[0]

        # Raw decision-function margin (not a probability). See
        # _decision_score: sign points at the class, |score| is the margin.
        score = _decision_score(text)

        label = "FAKE" if prediction == 1 else "REAL"
        confidence = abs(score) if score is not None else None
        
        return jsonify({
            'text': text[:100] + '...' if len(text) > 100 else text,
            'prediction': label,
            'score': score,
            'confidence': confidence
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    status = 'ready' if model is not None else 'not_ready'
    return jsonify({'status': status, 'model_loaded': model is not None})


if __name__ == '__main__':
    load_model()
    app.run(debug=DEBUG, port=5000)
