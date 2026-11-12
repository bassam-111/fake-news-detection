# Dataset Folder

This folder contains the news dataset files for training the fake news detection model.

## Quick test: bundled sample dataset

`sample_news.csv` is a small dataset of 40 original, hand-written examples
(20 FAKE, 20 REAL) included in the repo so the full pipeline can be tried
without downloading anything:

```bash
python model/train_model.py --sample
python app.py
```

Sample metrics are **not** representative of real performance — the sample
only exists to prove the pipeline runs end to end.

## Full dataset (for real training)

To train the model properly, you need the Kaggle Fake and Real News
Dataset (44,898 articles):

1. Download it from https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
2. Extract `Fake.csv` and `True.csv` into this folder
3. Merge them into `news.csv` with:

```python
import pandas as pd

fake = pd.read_csv('dataset/Fake.csv')
real = pd.read_csv('dataset/True.csv')
fake['label'] = 'FAKE'
real['label'] = 'REAL'
news = pd.concat([fake, real], ignore_index=True)[['title', 'text', 'label']]
news.to_csv('dataset/news.csv', index=False)
```

4. Train:
```bash
python model/train_model.py
```

## Dataset structure

The `news.csv` file should have these columns:
- `title` - Article headline
- `text` - Article content
- `label` - FAKE or REAL

## File sizes

- `Fake.csv`: ~63 MB
- `True.csv`: ~54 MB
- `news.csv`: ~110 MB (after merging)

## Notes

⚠️ **Important**: Large dataset files (`news.csv`, `Fake.csv`, `True.csv`)
and trained model files (`.pkl`) are excluded from Git via `.gitignore`
— each user downloads the dataset and trains locally.
