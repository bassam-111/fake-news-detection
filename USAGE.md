# How to Use the Fake News Detection System

This guide will help you set up and use the fake news detection system.

## 📋 Prerequisites

- Python 3.10 or higher
- 16GB RAM recommended (8GB minimum)
- 5GB free disk space
- Internet connection (for dataset download)

## 🛠️ Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/bassam-111/fake-news-detection.git
cd fake-news-detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## 📊 Getting the Dataset

### Method 1: Manual Download

1. Go to https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
2. Download the dataset ZIP file
3. Extract `Fake.csv` and `True.csv`
4. Place them in the `dataset/` folder
5. Merge them into `news.csv` — the exact snippet is in `dataset/README.md`

### Method 2: Kaggle CLI

```bash
# 1. Install the Kaggle CLI and put your API token in ~/.kaggle/kaggle.json
pip install kaggle
# 2. Download and unzip
kaggle competitions download -c fake-and-real-news-dataset
unzip fake-and-real-news-dataset.zip -d dataset/
# 3. Merge Fake.csv + True.csv into news.csv (snippet in dataset/README.md)
```

### Method 3: Bundled sample (no download)

```bash
# Trains on the 40-row sample in dataset/sample_news.csv — smoke test only
python model/train_model.py --sample
```

See `dataset/README.md` for detailed instructions.

## 🎓 Training the Model

Once you have the dataset (`news.csv` in `dataset/` folder):

```bash
python model/train_model.py
```

**Training time**: 15-25 minutes on Core i7 8th gen with 16GB RAM (seconds on the bundled sample with `--sample`)

**Expected output**:
- 3 trained models saved in `model/` folder
- Accuracy: 99.32% reported from the original training run (not independently verifiable from this repo alone — dataset and model files are not committed)
- Models: `passive_aggressive_model.pkl`, `logistic_regression_model.pkl`, `naive_bayes_model.pkl`

## 🌐 Running the Web Application

### Start the Flask Server

```bash
python app.py
```

**Access the app**: http://localhost:5000

### Using the Web Interface

1. Open your browser to http://localhost:5000
2. Paste or type news article text
3. Click "Analyze News"
4. See prediction: FAKE or REAL
5. View the decision score (raw model margin, not a probability)

## 💻 Using the Standalone Predictor

For command-line predictions without the web server:

### Interactive Mode

```bash
python predict.py
```

Then enter news text when prompted.

### Single Prediction

```bash
python predict.py "Your news text here..."
```

### Programmatic Usage

```python
import joblib

# Load model
model = joblib.load('model/passive_aggressive_model.pkl')

# Make prediction
text = "Breaking news article text here..."
prediction = model.predict([text])[0]

result = "FAKE" if prediction == 1 else "REAL"
print(f"Prediction: {result}")
```

## 📈 Model Performance

Best Model: **PassiveAggressiveClassifier** (reported from the original training run)

- **Accuracy**: 99.32%
- **Precision**: 99.36%
- **Recall**: 99.34%
- **F1-Score**: 99.35%

Tested on 8,980 articles (20% test split from 44,898 total articles).
These numbers are reported, not independently verifiable from this repo
alone — the dataset and trained models are not committed here. To verify
them, download the dataset and re-run `python model/train_model.py`.

## 🔧 Troubleshooting

### Model not found error

**Problem**: `Model not found at model/passive_aggressive_model.pkl`

**Solution**: Run `python model/train_model.py` to train the model first

### Dataset not found error

**Problem**: `Dataset not found at dataset/news.csv`

**Solution**: Download the dataset and merge `Fake.csv` + `True.csv` into
`news.csv` (snippet in `dataset/README.md`) — or run
`python model/train_model.py --sample` for a quick test without the dataset

### Memory error during training

**Problem**: System runs out of memory

**Solution**: 
- Close other applications
- Increase virtual memory/swap space
- Use a machine with more RAM (16GB recommended)

### Flask app not starting

**Problem**: Port 5000 already in use

**Solution**: 
```bash
# Use different port
flask run --port 8080
# Or modify app.py: app.run(port=8080)
```

## 📝 API Endpoints

### POST /predict

Predict if news text is fake or real. `score` and `confidence` are the
raw decision-function margin (and its absolute value) — not probabilities.

**Request**:
```json
{
    "text": "News article content here..."
}
```

**Response**:
```json
{
    "text": "News article content...",
    "prediction": "FAKE",
    "score": 2.45,
    "confidence": 2.45
}
```

### GET /health

Check if the server and model are loaded.

**Response**:
```json
{
    "status": "ready",
    "model_loaded": true
}
```

## 🚀 Next Steps

- Try different news articles
- Experiment with other ML models
- Add BERT for improved accuracy
- Create a browser extension
- Deploy to cloud (Heroku, AWS, etc.)

## 📚 Additional Resources

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Kaggle Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)

## 💡 Tips

- **For best results**: Use complete article text (title + content)
- **Minimum text**: At least 50 characters for reliable prediction
- **Maximum text**: 10,000 characters limit in web interface
- **Language**: Model trained on English news only

## 📧 Need Help?

Create an issue on GitHub: https://github.com/bassam-111/fake-news-detection/issues
