# How to Download Kaggle Dataset

## Method 1: Using Kaggle CLI (Easiest)

### Step 1: Get Kaggle API Credentials
1. Go to: https://www.kaggle.com/settings/account
2. Scroll down and click **"Create New API Token"**
3. A file `kaggle.json` will download

### Step 2: Set Up Kaggle Credentials
```powershell
# Create .kaggle folder
mkdir $env:USERPROFILE\.kaggle

# Move the downloaded kaggle.json
Move-Item -Path "$env:USERPROFILE\Downloads\kaggle.json" -Destination "$env:USERPROFILE\.kaggle\"

# Set permissions (important for Windows)
icacls "$env:USERPROFILE\.kaggle\kaggle.json" /grant:r "$env:USERNAME`:(F)"
```

### Step 3: Install Kaggle CLI
```powershell
pip install kaggle
```

### Step 4: Run Download Script
```powershell
python download_dataset.py
```

---

## Method 2: Manual Download (No CLI)

1. **Visit Dataset Page**: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

2. **Download Files**:
   - Click "Download" button
   - Download `Fake.csv` and `True.csv`

3. **Place Files**:
   - Extract and move to: `dataset/` folder
   - You should have:
     - `dataset/Fake.csv`
     - `dataset/True.csv`

4. **Run Merge Script**:
   ```powershell
   python merge_dataset.py
   ```

---

## Method 3: Direct Download via Browser

1. Download from: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
2. Extract ZIP file
3. Copy `Fake.csv` and `True.csv` to `dataset/` folder
4. Create `news.csv` by running:
   ```powershell
   python merge_dataset.py
   ```

---

## Verify Dataset

After downloading, check if the file exists:
```powershell
ls dataset/
```

You should see: `news.csv` (~500MB)

---

## Train on Full Dataset

Once you have `dataset/news.csv`:

```powershell
python model/train_model.py
```

Expected output:
- **Accuracy**: 92-94%
- **Training time**: 15-25 minutes
- **Models saved**: 3 trained models in `model/` folder

---

## Need Help?

If downloads fail:
1. Check internet connection
2. Verify Kaggle account is active
3. Try Manual Download (Method 2)
4. Check `dataset/` folder has space (~2GB)
