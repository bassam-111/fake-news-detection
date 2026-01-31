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
        
        # Get decision function score if available
        score = None
        if hasattr(model.named_steps['clf'], 'decision_function'):
            score = float(model.decision_function([text])[0])
        
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
    app.run(debug=True, port=5000)
