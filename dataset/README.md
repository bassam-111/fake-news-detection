# Dataset Folder

This folder contains the news dataset files for training the fake news detection model.

## Required Files

To train the model, you need:
- `news.csv` - Combined dataset (fake + real news)

OR

- `Fake.csv` - Fake news articles
- `True.csv` - Real news articles

## How to Get the Dataset

### Option 1: Download from Kaggle (Recommended)

1. Visit: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
2. Download the dataset
3. Extract and place `Fake.csv` and `True.csv` in this folder
4. Run: `python merge_dataset.py` to create `news.csv`

### Option 2: Use Kaggle API

```bash
# Setup Kaggle credentials first (see DOWNLOAD_DATASET.md)
python download_dataset.py
```

### Option 3: Use Sample Data (Testing Only)

```bash
# Creates a small sample dataset for testing
python create_sample_dataset.py
```

## Dataset Structure

The `news.csv` file should have these columns:
- `title` - Article headline
- `text` - Article content
- `label` - FAKE or REAL

## File Sizes

- `Fake.csv`: ~63 MB
- `True.csv`: ~54 MB
- `news.csv`: ~110 MB (after merging)

## Notes

⚠️ **Important**: Large dataset files (>100MB) are excluded from Git via `.gitignore`

✅ Each user must download the dataset separately to train the model

✅ Trained model files (.pkl) are also excluded from Git
